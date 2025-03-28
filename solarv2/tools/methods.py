######### Imports #########

import glob
import json
import os
import pickle
import warnings
from ast import literal_eval

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as signal
from itables import init_notebook_mode
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
from scipy.optimize import curve_fit
from skimage.measure import LineModelND, ransac
from sklearn.cluster import DBSCAN
from sklearn.linear_model import RANSACRegressor
from skspatial.objects import Cylinder, Line, Plane, Point, Triangle
from tqdm.auto import tqdm

if __package__:
    from . import params
else:
    import params

####### Methods #######

sipm_map = None
match_dict = {}


def sipm_to_xy(sn, ch):
    global sipm_map
    if sipm_map is None:
        with open(params.sipm_map_file, "r") as f:
            sipm_map = json.load(f)

    xy = sipm_map.get(str(sn), {}).get(str(ch), None)
    if xy is None:
        return None
    else:
        x = xy[0] + 64
        y = xy[1] - 16
        return (x, y)


# Check if SiPMs on anode area
def get_sipm_mask(sn, ch):
    xy = sipm_to_xy(sn, ch)
    # return True
    if xy is None:
        return False
    else:
        return (
            xy[0] > -params.detector_x / 2
            and xy[0] < params.detector_x / 2
            and xy[1] < params.detector_y / 2
            and xy[1] > -params.detector_y / 2
        )


# Cylinder parameters for dQ/dx
def get_dh(unit_vector, length):
    if params.force_dh is not None:
        return params.force_dh

    dl_vector = np.array([params.xy_epsilon, params.xy_epsilon, params.z_epsilon])
    dl_projection = np.dot(abs(unit_vector), dl_vector)
    ratio = round(length / dl_projection)
    dh = length / max(round(ratio), 1)

    return dh


def get_dr(rmse):
    if params.force_dr is not None:
        return params.force_dr

    dl_vector = np.array([params.xy_epsilon, params.xy_epsilon, params.z_epsilon])
    min_dr = np.linalg.norm(dl_vector) / 4
    dr = max(rmse, min_dr)

    return dr


def prepare_event(event, charge_df, light_df=None, match_dict=None):

    if event not in charge_df.index:
        tqdm.write(f"Event {event} not found in {params.output_folder}")
        return None, None, None

    light_event = None
    if light_df is not None:
        light_indices = light_df["event"].copy()

        if event in match_dict:
            light_event = match_dict.get(event)[0]
            light_matches = light_indices[light_indices == light_event].index
            light_event = light_df.loc[light_matches].dropna(subset=params.light_variable)
        else:
            print(f"No light event found for event {event} in {params.output_folder}")

    charge_event = pd.DataFrame(
        charge_df.rename(
            {
                "hit_x": "event_hits_x",
                "hit_y": "event_hits_y",
                "hit_z": "event_hits_z",
                "hit_q": "event_hits_q",
            },
            axis=1,
        )
        .loc[
            event,
            [
                "event_hits_x",
                "event_hits_y",
                "event_hits_z",
                "event_hits_q",
            ],
        ]
        .to_list(),
        index=["x", "y", "z", "q"],
    ).T
    if "event_hits_channelid" in charge_df:
        channel_ids = pd.Series(
            charge_df.loc[
                event,
                "event_hits_channelid",
            ]
        )
        non_zero_mask = (charge_event["x"] != 0) * (charge_event["y"] != 0)  # Remove (0,0) entries

        noisy_channels_mask = ~channel_ids.isin([ch[0] for ch in params.channel_disable_list])  # Disable channel 7

        mask = non_zero_mask * noisy_channels_mask  # Full hits mask

        # Apply boolean indexing to x, y, and z arrays
        charge_event = charge_event[mask]
    else:
        mask = np.full(len(charge_event["q"]), True)

    charge_event["q"] = charge_event["q"] * params.charge_gain  # Convert mV to ke

    return charge_event, light_event, mask


def match_events(charge_df, light_df, window=10, return_dt=False):
    match_dict = {}
    dt_dict = {}

    charge_events = charge_df[["event_unix_ts", "event_start_t"]].drop_duplicates()
    light_events = light_df[["tai_ns", "event"]].drop_duplicates()

    for event, row in tqdm(charge_events.iterrows(), total=len(charge_events), desc="Matching events"):
        charge_ts = (float(row["event_unix_ts"]) * 1e6) + (float(row["event_start_t"]) * 1e-1)
        dt = light_events["tai_ns"].astype(float) * 1e-3 - 36000000 - charge_ts
        light_matches = light_events.where(abs(dt) <= window).dropna()

        if return_dt:
            dt_dict[event] = dt[abs(dt) == min(abs(dt))].values[0]

        if not light_matches.empty:
            if event in match_dict:
                match_dict[event].append(light_matches["event"].unique().astype(int).tolist())
            else:
                match_dict[event] = light_matches["event"].unique().astype(int).tolist()

    if return_dt:
        return match_dict, dt_dict

    return match_dict


# Create a list of fake data
def generate_dead_area(z_range, buffer=1):
    # Dead area on SiPMs
    fake_x = []
    fake_y = []
    fake_z = []

    border = buffer - params.pixel_pitch / 2

    for k in range((params.detector_x // params.quadrant_size)):
        for l in range((params.detector_y // params.quadrant_size)):
            # Dead area on chips 33, 44, 54, 64
            if (l + params.first_chip[0], k + params.first_chip[1]) in [
                (3, 3),
                (4, 4),
                (5, 4),
                (6, 4),
            ]:
                # continue
                temp_x, temp_y, temp_z = np.meshgrid(
                    np.linspace(
                        border,
                        params.quadrant_size - border,
                        int(params.quadrant_size / buffer),
                    )
                    - params.detector_x / 2
                    + params.quadrant_size * (k),
                    -np.linspace(
                        border,
                        params.quadrant_size - border,
                        int(params.quadrant_size / buffer),
                    )
                    + params.detector_y / 2
                    - params.quadrant_size * (l),
                    z_range,
                )

                fake_x.extend(temp_x.flatten())
                fake_y.extend(temp_y.flatten())
                fake_z.extend(temp_z.flatten())
            # Dead area on chip 42
            elif k + params.first_chip[1] == 2 and l + params.first_chip[0] == 4:
                temp_x, temp_y, temp_z = np.meshgrid(
                    np.linspace(
                        border,
                        params.quadrant_size - border,
                        int(params.quadrant_size / buffer),
                    ),
                    np.linspace(
                        border,
                        params.quadrant_size - border,
                        int(params.quadrant_size / buffer),
                    ),
                    z_range,
                )

                mask1 = (temp_y - params.quadrant_size / 2) - (temp_x - params.quadrant_size / 2) >= 0
                mask2 = (temp_y <= params.quadrant_size / 2 + buffer) & (temp_y >= params.quadrant_size / 2 - buffer)
                mask3 = (temp_x <= params.quadrant_size / 2 + buffer) & (temp_x >= params.quadrant_size / 2 - buffer)
                mask = mask1 | (mask2 & mask3)
                temp_x = temp_x[mask] - params.detector_x / 2 + params.quadrant_size * (k)
                temp_y = -temp_y[mask] + params.detector_y / 2 - params.quadrant_size * (l)
                temp_z = temp_z[mask]

                fake_x.extend(temp_x)
                fake_y.extend(temp_y)
                fake_z.extend(temp_z)
            # Dead area on SiPMs
            else:
                sipm_points = int((params.sipm_size + params.pixel_pitch) / (buffer))
                if sipm_points > 1:
                    x1 = np.linspace(
                        -(params.sipm_size + params.pixel_pitch) / 2 + buffer,
                        +(params.sipm_size + params.pixel_pitch) / 2 - buffer,
                        sipm_points,
                    )
                else:
                    x1 = np.array([0])

                x1 = params.quadrant_size / 2 + x1

                temp_x, temp_y, temp_z = np.meshgrid(
                    x1 - params.detector_x / 2 + params.quadrant_size * k,
                    -x1 + params.detector_y / 2 - params.quadrant_size * l,
                    z_range,
                )

                # Removing channel 7
                disable_x, disable_y, disable_z = [], [], []

                for channel in params.channel_disable_list:
                    coords = channel[1]
                    temp_x2, temp_y2, temp_z2 = np.meshgrid(
                        np.array(
                            [
                                -params.detector_x / 2
                                + (coords[0] * params.pixel_pitch)
                                + (params.pixel_pitch - params.pixel_size)
                            ]
                        )
                        + params.quadrant_size * k,
                        np.array(
                            [
                                params.detector_y / 2
                                - (coords[1] * params.pixel_pitch)
                                - (params.pixel_pitch - params.pixel_size)
                            ]
                        )
                        - params.quadrant_size * l,
                        z_range,
                    )
                    disable_x.extend(temp_x2)
                    disable_y.extend(temp_y2)
                    disable_z.extend(temp_z2)

                fake_x.extend(np.append(temp_x, disable_x))
                fake_y.extend(np.append(temp_y, disable_y))
                fake_z.extend(np.append(temp_z, disable_z))

    fake_x = np.array(fake_x)
    fake_y = np.array(fake_y)
    fake_z = np.array(fake_z)

    fake_data = np.c_[np.power(-1, params.flip_x) * fake_x, fake_y, fake_z]

    return fake_data


def lifetime_correction(z):
    # z in mm, velocity in mm/ms, lifetime in ms
    velocity = params.drift_velocity * 1000
    if params.lifetime > 0:
        return np.exp(z / (velocity * params.lifetime))
    else:
        return 1


# Apply DBSCAN clustering
def cluster_hits(hitArray):
    # First stage clustering
    z_intervals = []
    first_stage = DBSCAN(eps=params.xy_epsilon, min_samples=params.min_samples).fit(hitArray[:, 0:2])
    for label in first_stage.labels_:
        if label > -1:
            mask = first_stage.labels_ == label
            z = hitArray[mask, 2]
            z_intervals.append((min(z), max(z)))

    # Sort the intervals based on their start points
    sorted_intervals = sorted(z_intervals, key=lambda interval: interval[0])

    # Initialize a list to store the intervals representing the empty space
    empty_space_ranges = []

    # Iterate through the sorted intervals to find the gaps
    for i in range(len(sorted_intervals) - 1):
        current_interval = sorted_intervals[i]
        next_interval = sorted_intervals[i + 1]

        # Calculate the gap between the current interval and the next interval
        gap_start = current_interval[1]
        gap_end = next_interval[0]

        # Check if there is a gap (empty space) between intervals
        if gap_end > gap_start and gap_end < gap_start + 40:
            empty_space_ranges.append(np.arange(gap_start, gap_end, params.z_epsilon))

    if not empty_space_ranges:
        if np.std(hitArray[:, 2]) > 0:
            z_range = np.arange(
                np.mean(hitArray[:, 2]) - np.std(hitArray[:, 2]),
                np.mean(hitArray[:, 2]) + np.std(hitArray[:, 2]),
                params.z_epsilon,
            )

        else:
            z_range = [np.mean(hitArray[:, 2])]

    else:
        z_range = np.concatenate(empty_space_ranges)

    # Create a list of holes
    fake_data = generate_dead_area(z_range, buffer=(params.xy_epsilon - 1))
    fake_data_count = len(fake_data)

    # Second stage clustering
    # Combine fake to true data
    second_stage_data = np.concatenate([hitArray, fake_data])
    second_stage = DBSCAN(eps=params.xy_epsilon, min_samples=1).fit(second_stage_data[:, 0:2])

    # Third stage clustering
    # Create a new array with z and labels
    third_stage_z = np.c_[second_stage.labels_ * 1e3, second_stage_data[:, 2]]
    labels = second_stage.labels_.copy()
    flag = labels > -1

    third_stage_data = third_stage_z[flag].copy()
    third_stage = DBSCAN(eps=params.z_epsilon, min_samples=params.min_samples, metric="chebyshev").fit(third_stage_data)

    # Shift labels by 1 so that negative values are reserved for outliers
    labels[flag] = third_stage.labels_ + 1

    # Remove fake data
    if fake_data_count > 0:
        labels = labels[:-fake_data_count]

    return labels


# Apply Ransac Fit
def ransacFit(
    hitArray,
    weightArray=None,
    min_samples=None,
    residual_threshold=None,
):
    # Suppress the UndefinedMetricWarning
    warnings.filterwarnings("ignore", category=Warning, module="sklearn")

    if weightArray is not None:
        estimator = RANSACRegressor(
            min_samples=min_samples,
            max_trials=params.ransac_max_trials,
            residual_threshold=residual_threshold,
        )
        last_column = len(hitArray[0]) - 1
        inliers = estimator.fit(
            hitArray[:, 0:last_column],
            hitArray[:, last_column],
            sample_weight=weightArray,
        ).inlier_mask_

        # Check it enouth inliers
        if sum(inliers) > params.ransac_min_samples:
            score = estimator.score(hitArray[:, 0:last_column], hitArray[:, last_column])
        else:
            score = np.nan
    else:
        model_robust, inliers = ransac(
            hitArray,
            LineModelND,
            min_samples=min_samples,
            residual_threshold=residual_threshold,
            max_trials=params.ransac_max_trials,
        )

        # Check it enouth inliers
        if sum(inliers) > params.ransac_min_samples:
            score = model_robust.residuals(hitArray)
        else:
            score = np.nan

    outliers = inliers == False

    # Reset the warning filter
    warnings.filterwarnings("default", category=Warning, module="sklearn")

    return inliers, outliers, score


# Apply best line fit
def lineFit(hitArray):
    line = Line.best_fit(hitArray)
    max_point = Point(np.max(hitArray, axis=0))
    min_point = Point(np.min(hitArray, axis=0))
    centre_point = (max_point + min_point) / 2
    line.point = line.project_point(centre_point)
    residuals = np.array([line.distance_point(point) for point in hitArray])

    # Calculate chi-squared
    mse = np.sum(residuals**2) / len(residuals)

    return line, mse


# Calculate dQ/dx from a line fit
def dqdx(hitArray, q, line_fit, target_dh, dr, h, ax=None):
    # Cylinder steps for dQ/dx
    steps = (
        np.arange(-2 * target_dh, h + 2 * target_dh, target_dh) + target_dh / 2
    )  # centering the steps in the middle of the cylinder
    projected_pitch = np.dot(
        np.array(
            [
                params.pixel_pitch,
                params.pixel_pitch,
                params.integration_window * params.drift_velocity,
            ]
        ),
        abs(line_fit.direction.unit()),
    )
    limit_pitch = np.dot(
        np.array([params.pixel_pitch, params.pixel_pitch, target_dh]),
        abs(line_fit.direction.unit()),
    )

    # Mask of points that have been accounted for
    counted = np.zeros(len(q), dtype=bool)

    # Array of dQ values for each step
    dq_i = pd.Series(np.zeros(len(steps), dtype=float), index=steps)
    dh_i = pd.Series(np.zeros(len(steps), dtype=float), index=steps)

    for step_idx, step in enumerate(steps):
        cyl_origin = line_fit.to_point(step - target_dh / 2 - h / 2)  # centering the step in the base of the cylinder
        cyl_height = line_fit.direction.unit() * target_dh
        cylinder_fit = Cylinder(
            cyl_origin,
            cyl_height,
            dr,
        )
        if ax is not None:
            cylinder_fit.plot_3d(ax)

        # Initialize variables to store the minimum and maximum points
        point_distances = [line_fit.transform_points([cyl_origin])]
        for point_idx, point in enumerate(hitArray):
            if not counted[point_idx] and cylinder_fit.is_point_within(point):
                counted[point_idx] = True
                dq_i.loc[step] += q[point_idx]

                point_distances.append(line_fit.transform_points([point]))

        point_distances.append(point_distances[0] + cyl_height.norm())

        # Calculate dh_i based on the distance between points
        max_distance = 0
        if len(point_distances) > 2:
            point_distances = np.unique(np.array(point_distances))
            intervals = np.diff(point_distances)

            # Sum up the live areas (intervals <= limit_pitch) and correct by the pixel pitch
            live_intervals = intervals[intervals <= limit_pitch]
            dead_intervals_count = np.sum(intervals > limit_pitch)
            boundary_intervals = ((intervals[0] > limit_pitch) + (intervals[-1] > limit_pitch)) / 2
            total_live_distance = (
                np.sum(live_intervals) + (dead_intervals_count * projected_pitch) - boundary_intervals * projected_pitch
            )

            # Calculate max_distance based on live areas
            max_distance = min(abs(point_distances[-1] - point_distances[0]), total_live_distance)

            # Calculate step_length based on conditions
        step_length = max_distance if max_distance > 0 else (projected_pitch if dq_i.loc[step] > 0 else 0)

        # Assign the minimum of step_length and target_dh to dh_i.loc[step]
        dh_i.loc[step] = min(step_length, target_dh)

    return dq_i, dh_i


# Fit clusters with Ransac method
def fit_hit_clusters(
    hitArray,
    q,
    labels,
    ax2d=None,
    ax3d=None,
    plot_cyl=False,
    refit_outliers=True,
):
    metrics = {}
    # Fit clusters
    idx = 0
    condition = lambda: idx < len(np.unique(labels))
    while condition():
        label = np.unique(labels)[idx]
        mask = labels == label
        if label > 0 and mask.sum() > params.min_samples:
            xyz_c = hitArray[mask]
            q_c = np.array(q)[mask]

            norm = np.linalg.norm(np.max(xyz_c, axis=0) - np.min(xyz_c, axis=0))

            # Fit the model
            inliers, outliers, score = ransacFit(
                xyz_c,
                weightArray=q_c - min(q_c) + 1,
                min_samples=params.ransac_min_samples,
                residual_threshold=params.ransac_residual_threshold,
            )

            # Refit outliers
            level_1 = np.where(mask)[0]
            level_2 = np.where(outliers)[0]
            level_3 = level_1[level_2]

            if refit_outliers and sum(outliers) > params.min_samples:
                outlier_labels = cluster_hits(xyz_c[outliers])
                last_label = max(labels) + 1
                # Assign positive labels to clustered outliers and negative labels to unlclustered outliers
                for i, j in enumerate(level_3):
                    labels[j] = (outlier_labels[i] + last_label) * (1 if outlier_labels[i] > 0 else -1)
            else:
                # Assign negative labels to outliers
                for j in level_3:
                    labels[j] = -labels[j]

            if sum(inliers) > params.min_samples:
                line_fit, mse = lineFit(xyz_c[inliers])

                if ax2d is not None:
                    # 2D plot
                    line_fit.plot_2d(
                        ax2d,
                        t_1=-norm / 2,
                        t_2=norm / 2,
                        c="red",
                        label=f"Track {label}",
                        zorder=10,
                    )
                if ax3d is not None:
                    # 3D plot
                    line_fit.plot_3d(
                        ax3d,
                        t_1=-norm / 2,
                        t_2=norm / 2,
                        c="red",
                        label=f"Track {label}",
                    )

                # Calculate dQ/dx
                target_dh = get_dh(line_fit.direction, norm)
                dr = get_dr(np.sqrt(mse))

                dq_i, dh_i = dqdx(
                    xyz_c[inliers],
                    q_c[inliers],
                    line_fit,
                    target_dh=target_dh,
                    dr=dr,
                    h=norm,
                    ax=ax3d if ax3d is not None and plot_cyl else None,
                )

                q_eff = dq_i.sum() / q_c[inliers].sum()
                if dq_i.sum() != 0:
                    dq = dq_i
                    dh = dh_i
                else:
                    dq = 0
                    dh = 0

                metrics[label] = {
                    "Fit_line": line_fit,
                    "Fit_norm": norm,
                    "Fit_mse": mse,
                    "RANSAC_score": score,
                    "q_eff": q_eff,
                    "dQ": dq,
                    "dx": dh,
                    # "target_dx": target_dh,
                }

        idx = np.unique(labels).tolist().index(label) + 1

    metrics["Fit_labels"] = labels

    return metrics


# Light
def voxelize_hits(
    charge_df,
    sipm_df,
    light_variable,
    fit_light=True,
    charge_lines=[],
):
    sipm_voxels_metrics = {}
    xyzl_df = pd.DataFrame()
    for row, sipm in sipm_df.dropna(subset=light_variable).iterrows():
        voxel_mask = (abs(charge_df["x"] - sipm["x"]) <= params.quadrant_size / 2) & (
            abs(charge_df["y"] - sipm["y"]) <= params.quadrant_size / 2
        )
        voxel_charge = charge_df["q"][voxel_mask]
        voxel_z = charge_df["z"][voxel_mask]

        voxel_charge = voxel_charge * lifetime_correction(voxel_z)

        xyzl = sipm[["x", "y", light_variable]].copy()

        if len(voxel_charge) > 0:
            sipm_idx = (sipm["sn"], sipm["ch"])
            sipm_voxels_metrics[sipm_idx] = {}
            voxel_mean_z = np.average(voxel_z, weights=voxel_charge)
            voxel_total_charge = voxel_charge.sum()
            sipm_voxels_metrics[sipm_idx]["charge_q"] = voxel_total_charge
            sipm_voxels_metrics[sipm_idx]["charge_z"] = voxel_mean_z
            sipm_voxels_metrics[sipm_idx][light_variable] = sipm[light_variable]
            xyzl["z"] = voxel_mean_z
        else:
            xyzl["z"] = np.average(charge_df["z"], weights=charge_df["q"])

            if charge_lines:
                point = Point([sipm["x"], sipm["y"], 0])
                plane = Plane(point=point, normal=[0, 0, 1])
                line_distances = [
                    (line.distance_point(point) if not plane.normal.is_parallel(line.direction) else np.inf)
                    for line in charge_lines
                ]
                if line_distances and min(line_distances) < np.inf:
                    charge_line = charge_lines[np.argmin(line_distances)]
                    projection = plane.project_line(charge_line)
                    projected_point = projection.project_point(point)
                    v_line = Line(point=projected_point, direction=[0, 0, 1])
                    intersection = charge_line.intersect_line(v_line, check_coplanar=False)
                    if all(abs(projected_point - point)[:2] <= params.quadrant_size / 2):
                        xyzl["z"] = intersection[2]

        xyzl_df = pd.concat([xyzl_df, xyzl], axis=1)

    if not xyzl_df.empty and fit_light:
        xyzl_df = xyzl_df.T
        xyzl_df = xyzl_df.sort_values(by=light_variable, ascending=False)
        xyzl_df = xyzl_df[xyzl_df[light_variable] > 0]
        if len(xyzl_df) > 3:
            # dbscan = DBSCAN(eps=params.quadrant_size, metric="chebyshev", min_samples=2)
            # labels = dbscan.fit(xyzl_df[["x", "y"]].values).labels_
            # unique, counts = np.unique(labels[labels>-1],return_counts=True)
            # if len(counts)>0 and max(counts)>3:
            #     label = unique[counts.argmax()]

            #     xyz = xyzl_df[labels==label].head(5)
            xyzl = xyzl_df.head(5).astype(float)
            inliers, outliers, score = ransacFit(
                xyzl[["x", "y", "z"]].values,
                weightArray=xyzl[light_variable].values,
                min_samples=3,
                residual_threshold=10,
            )
            if inliers.sum() > 2:
                light_fit, mse = lineFit(xyzl[["x", "y", "z"]][inliers].values)
                sipm_voxels_metrics["Fit_line"] = light_fit
                sipm_voxels_metrics["RANSAC_score"] = score
                sipm_voxels_metrics["Fit_mse"] = mse
                sipm_voxels_metrics["Fit_threshold"] = xyzl[light_variable][inliers].min()

    return sipm_voxels_metrics


def light_geometry(track_line, track_norm, sipm_df, light_variable="integral"):
    metrics = {}
    point1 = track_line.to_point(-track_norm / 2)
    point2 = track_line.to_point(track_norm / 2)
    centre = track_line.point

    iterate_df = sipm_df.dropna(subset=light_variable).copy()
    for row, sipm in iterate_df.iterrows():
        sipm_idx = (sipm["sn"], sipm["ch"])
        point3 = Point([sipm["x"], sipm["y"], 0])
        triangle = Triangle(point1, point2, point3)
        angle = triangle.angle("C")
        distance = point3.distance_point(centre)
        metrics[sipm_idx] = {}
        metrics[sipm_idx]["distance"] = distance
        metrics[sipm_idx]["angle"] = angle
        metrics[sipm_idx][light_variable] = sipm[light_variable]

    return metrics


# Charge data
def get_track_stats(metrics, empty_ratio_lims=(0, 1), min_entries=1):
    track_dQdx = []
    track_length = []
    track_score = []
    track_z = []
    track_points = []
    events = []
    tracks = []

    empty_count = 0
    short_count = 0

    print(rf"Using lifetime correction with tau = {params.lifetime} ms")

    for event, entry in metrics.items():
        for track, values in entry.items():
            if isinstance(track, str) or track <= 0:
                continue

            track_length.append(values["Fit_norm"])
            track_score.append(values["RANSAC_score"])
            track_z.append(values["Fit_line"].point[2])
            events.append(event)
            tracks.append(track)

            dQ = values["dQ"]
            dx = values["dx"]
            non_zero_mask = np.where((dQ > 0) & (dx > 0))[0]

            if len(non_zero_mask) < min_entries:
                short_count += 1
                track_dQdx.append(np.nan)
                track_points.append(np.nan)
                continue

            empty_ratio = sum(dQ.iloc[non_zero_mask[0] : non_zero_mask[-1] + 1] == 0) / (
                non_zero_mask[-1] - non_zero_mask[0] + 1
            )

            if empty_ratio > empty_ratio_lims[1] or empty_ratio < empty_ratio_lims[0]:
                empty_count += 1
                track_dQdx.append(np.nan)
                track_points.append(np.nan)
                continue

            dQdx = (dQ / dx).rename("dQdx")
            dQdx = dQdx.iloc[non_zero_mask[0] : non_zero_mask[-1] + 1]

            position = [values["Fit_line"].to_point(t=-values["Fit_norm"] / 2 + t) for t in dQ.index]
            position = position[non_zero_mask[0] : non_zero_mask[-1] + 1]

            z = np.array([x[2] for x in position])
            dQdx *= lifetime_correction(z)

            track_dQdx.append(dQdx)
            track_points.append(pd.Series(position, index=dQdx.index, name="position"))

    print(f"Tracks with dead area outside {empty_ratio_lims} interval: {empty_count}")
    print(f"Tracks with less than {min_entries} entries: {short_count}")

    track_dQdx = pd.Series(track_dQdx)
    track_points = pd.Series(track_points)
    track_length = pd.Series(track_length)
    track_score = pd.Series(track_score)
    track_z = pd.Series(track_z)
    events = pd.Series(events)
    tracks = pd.Series(tracks)

    mask = track_length.notna() * track_score.notna() * track_z.notna()

    print(f"\nRemaining tracks: {sum(mask)}\n")

    track_dQdx = track_dQdx[mask]
    track_points = track_points[mask]
    track_length = track_length[mask]
    track_score = track_score[mask]
    track_z = track_z[mask]
    events = events[mask]
    tracks = tracks[mask]

    df = pd.DataFrame(
        [
            track_dQdx,
            track_points,
            track_length,
            track_score,
            track_z,
            events,
            tracks,
        ],
        index=[
            "track_dQdx",
            "track_points",
            "track_length",
            "track_score",
            "track_z",
            "event",
            "track",
        ],
    ).T

    return df


# ### Helpers


def load_params(parameters):
    kwargs = {}
    if parameters is not None:
        # Check if parameters are provided in a JSON file
        if len(parameters) == 1 and parameters[0].endswith(".json") and os.path.isfile(parameters[0]):
            with open(parameters[0], "r") as f:
                param = json.load(f)
        else:
            # Convert command line parameters to dictionary
            param = {
                key: value for param in parameters for key, value in [param.split("=") if "=" in param else (param, None)]
            }

        # Now process the parameters in a single for loop
        for key, value in param.items():
            if key in params.__dict__:
                try:
                    params.__dict__[key] = literal_eval(value) if not isinstance(params.__dict__[key], str) else value
                except ValueError:
                    params.__dict__[key] = value
            else:
                try:
                    kwargs[key] = literal_eval(value) if not isinstance(params.__dict__[key], str) else value
                except ValueError:
                    kwargs[key] = value

    return kwargs


def recal_params():
    params.dh_unit = params.xy_unit if params.xy_unit == params.z_unit else "?"
    params.light_unit = "p.e." if params.light_variable == "integral" else f"p.e./{params.time_unit}"
    if params.simulate_dead_area:
        params.detector_x = params.quadrant_size * 4
        params.detector_y = params.quadrant_size * 5

    params.first_chip = (2, 1) if params.detector_y == 160 else (1, 1)

    print(
        "\nCalculated parameters:\n",
        f"dh_unit set to {params.dh_unit}\n",
        f"light_unit set to {params.light_unit}\n",
        f"detector_x set to {params.detector_x}\n",
        f"detector_y set to {params.detector_y}\n",
        f"first_chip set to {params.first_chip}\n",
    )


def max_std(array, ax=None, array_max=None, min_count_ratio=0.9, max_std_ratio=0.5):
    max_std = array.std()
    max_count = len(array)
    if array_max is None:
        array_max = np.percentile(array, min(min_count_ratio * 100 + 1, 100))

    std = []
    count = []
    x_range = range(int(min(array)), (int(array_max) + 1), 1)
    for i in x_range:
        cut = array[array < i]
        std.append(cut.std())
        count.append(len(cut))

    std = np.array(std)
    count = np.array(count)
    condition = ((count / max_count).round(3) >= min_count_ratio) & ((std / max_std).round(3) <= max_std_ratio)
    vline = x_range[(np.where(condition)[0][-1] if np.any(condition) else (count / max_count > min_count_ratio).argmax())]

    print(
        "Max STD ratio",
        max_std_ratio,
        "limited to",
        min_count_ratio * 100,
        "% of events:",
        vline,
        "\n",
    )
    if ax is not None:
        ax.plot(std / max_std, label="STD ratio")
        ax.plot(count / max_count, label="Event count ratio")
        ax.axvline(vline, ls="--", c="r", label=f"{min_count_ratio*100}% of events")
        ax.legend()
        ax.tick_params(axis="both", direction="inout", which="major", top=True, right=True)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.grid(alpha=0.25)

    return vline


def cluster_hot_bins(min_n_ratio, n, x_edges, y_edges, scale=(1, 1), eps=8):
    n[np.isnan(n)] = 0
    min_n = min_n_ratio * n.max()
    filtered_n2 = n[n >= min_n]
    bin_centers_x = 0.5 * (x_edges[1:] + x_edges[:-1])
    bin_centers_y = 0.5 * (y_edges[1:] + y_edges[:-1])
    filtered_centers_x, filtered_centers_y = np.array(np.meshgrid(bin_centers_x, bin_centers_y))
    filtered_centers_x = filtered_centers_x[n.T > min_n]
    filtered_centers_y = filtered_centers_y[n.T > min_n]
    dbscan = DBSCAN(eps=eps, min_samples=int(np.sqrt(len(filtered_centers_y))), metric="chebyshev").fit(
        np.c_[filtered_centers_x / scale[0], filtered_centers_y / scale[1]]
    )
    return filtered_centers_x, filtered_centers_y, dbscan.labels_


# Peak finding algorithm for integration
def integrate_peaks(waveform, buffer_size=10, height=0.1, prominence=0.05):
    # Find peaks in the filtered waveform
    peaks, properties = signal.find_peaks(waveform, height=height, prominence=prominence)

    integration_result = 0
    start_index = 0  # Initialize the start index
    end_index = 0  # Initialize the end index

    for peak in peaks:
        # Determine the potential start and end indices
        potential_start = max(0, peak - buffer_size)
        potential_end = min(len(waveform), peak + buffer_size)

        # If the potential start is within the current peak region, update the end index
        if potential_start <= end_index:
            end_index = potential_end
        else:
            # Integrate the previous peak region and update indices for the new peak
            peak_region = waveform[start_index:end_index]
            integration_result += np.trapz(peak_region)
            start_index = potential_start
            end_index = potential_end

    # Integrate the last peak region
    peak_region = waveform[start_index:end_index]
    integration_result += np.trapz(peak_region)

    return integration_result, properties


# Metrics threatment
def filter_metrics(metrics, **kwargs):
    # Extract parameters with fallback to defaults in params
    min_score = kwargs.get("min_score", params.min_score)
    max_score = kwargs.get("max_score", params.max_score)
    min_track_length = kwargs.get("min_track_length", params.min_track_length)
    max_track_length = kwargs.get("max_track_length", params.max_track_length)
    max_tracks = kwargs.get("max_tracks", params.max_tracks)
    min_light = kwargs.get("min_light", params.min_light)
    max_light = kwargs.get("max_light", params.max_light)
    max_z = kwargs.get("max_z", params.max_z)

    print(
        f"\n min_score = {min_score}\n",
        f"max_score = {max_score}\n",
        f"min_track_length = {min_track_length}\n",
        f"max_track_length = {max_track_length}\n",
        f"max_tracks = {max_tracks}\n",
        f"min_light = {min_light}\n",
        f"max_light = {max_light}\n",
        f"max_z = {max_z}\n",
    )

    filtered_metrics = {}

    for event_idx, metric in metrics.items():
        # Calculate non_track_keys programmatically
        non_track_keys = sum(1 for key in metric if isinstance(key, str))

        # Filter based on the number of tracks and light metrics, if applicable
        if len(metric) <= max_tracks + non_track_keys:
            if ("Total_light" in metric and min_light <= metric["Total_light"] <= max_light) or "Total_light" not in metric:
                candidate_metric = {
                    track_idx: values
                    for track_idx, values in metric.items()
                    if isinstance(track_idx, str)
                    or (
                        track_idx > 0
                        and values["RANSAC_score"] >= min_score
                        and values["RANSAC_score"] <= max_score
                        and values["Fit_norm"] >= min_track_length
                        and values["Fit_norm"] <= max_track_length
                        and values["Fit_line"].point[2] < max_z
                    )
                }

                # Check if the filtered candidate metrics meet the criteria
                if non_track_keys < len(candidate_metric) <= max_tracks + non_track_keys:
                    filtered_metrics[event_idx] = candidate_metric

    print(f"{len(filtered_metrics)} metrics remaining")
    params.filter_label = len(filtered_metrics)

    # Save the filtering parameters to a JSON file
    output_path = os.path.join(params.work_path, params.output_folder)
    os.makedirs(output_path, exist_ok=True)
    with open(
        os.path.join(output_path, f"filter_parameters_{len(filtered_metrics)}.json"),
        "w+",
    ) as f:
        json.dump(
            {
                "min_score": min_score,
                "max_score": max_score,
                "min_track_length": min_track_length,
                "max_track_length": max_track_length,
                "max_tracks": max_tracks,
                "min_light": min_light,
                "max_light": max_light,
                "max_z": max_z,
            },
            f,
        )

    return filtered_metrics


def combine_metrics():
    combined_metrics = {}

    search_path = glob.glob(f"{params.work_path}/**/*metrics*.pkl")
    for file in tqdm(search_path, leave=True, desc="Combining metrics"):
        folder = file.split("/")[-2]
        tqdm.write(folder)
        with open(file, "rb") as f:
            metric = pickle.load(f)
            for key, value in tqdm(metric.items(), leave=False):
                combined_metrics[f"{folder}_{key}"] = value

    output_path = os.path.join(params.work_path, params.output_folder)
    os.makedirs(output_path, exist_ok=True)

    with open(os.path.join(output_path, f"metrics_{params.output_folder}.pkl"), "wb") as o:
        pickle.dump(combined_metrics, o)

    print("Done\n")

    return combined_metrics


# Function to create a square based on center coordinates and side size
def create_square(center, side_size):
    x = [
        center[0] - side_size / 2,
        center[0] + side_size / 2,
        center[0] + side_size / 2,
        center[0] - side_size / 2,
        center[0] - side_size / 2,
    ]
    y = [
        center[1] - side_size / 2,
        center[1] - side_size / 2,
        center[1] + side_size / 2,
        center[1] + side_size / 2,
        center[1] - side_size / 2,
    ]
    z = [0, 0, 0, 0, 0]  # All z-coordinates are set to 0 to align with the xy-plane

    vertices = [(x[i], y[i], z[i]) for i in range(5)]
    square = [[vertices[0], vertices[1], vertices[2], vertices[3]]]
    return square


def apply_lifetime(metrics):
    search_path = os.path.join(params.work_path, f"{params.output_folder}")
    events = pd.Series(metrics.keys())
    if events.str.contains("_").sum() > 0:

        events = (
            events.str.split("_", expand=True)
            .astype(str)
            .apply(lambda x: pd.Series([f"{x[0]}_{x[1]}", x[2]]), axis=1)
            .rename(columns={0: "label", 1: "event"})
        )

        for file in events["label"].unique():
            temp_df = pd.read_pickle(f"{params.work_path}/{file}/charge_df_{file}.pkl")
            for event_idx in events[events["label"] == file]["event"]:
                selection, _, _ = prepare_event(int(event_idx), temp_df)
                total_charge = sum(selection["q"].to_numpy() * lifetime_correction(selection["z"].to_numpy()))
                # print(event_idx, total_charge, selection["q"].to_numpy().sum(), metrics[f"{file}_{event_idx}"]["Total_charge"])
                metrics[f"{file}_{event_idx}"]["Total_charge"] = total_charge

    else:
        temp_df = pd.read_pickle(f"{params.work_path}/charge_df_{params.output_folder}.pkl")
        for event_idx in events[events["label"] == file]["event"]:
            selection, _, _ = prepare_event(int(event_idx), temp_df)
            total_charge = sum(selection["q"].to_numpy() * lifetime_correction(selection["z"].to_numpy()))
            # print(event_idx, total_charge, selection["q"].to_numpy().sum(), metrics[f"{file}_{event_idx}"]["Total_charge"])
            metrics[f"{file}_{event_idx}"]["Total_charge"] = total_charge

    return metrics
