# This file contains all the methods used by the notebooks.
# Import all by using from scripts import *
# Change parameters on your own script by calling params.PARAMETER

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
import pylandau
import scipy.signal as signal
import uproot
from itables import init_notebook_mode
from matplotlib.colors import LinearSegmentedColormap, LogNorm, to_rgba
from matplotlib.ticker import AutoMinorLocator, MaxNLocator, ScalarFormatter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
from skimage.measure import LineModelND, ransac
from sklearn.cluster import DBSCAN, KMeans
from sklearn.linear_model import RANSACRegressor
from skspatial.objects import Cylinder, Line, Plane, Point, Triangle, Vector
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

    dl_vector = np.array([params.xy_epsilon, params.xy_epsilon, params.z_epsilon]) * 2
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


# ## Data handling

# ### Event matching


def match_events(charge_df, light_df, window=10):
    match_dict = {}

    charge_events = charge_df[["event_unix_ts", "event_start_t"]].drop_duplicates()
    light_events = light_df[["tai_ns", "event"]].drop_duplicates()

    for event, row in tqdm(charge_events.iterrows(), total=len(charge_events)):
        charge_ts = (float(row["event_unix_ts"]) * 1e6) + (
            float(row["event_start_t"]) * 1e-1
        )
        light_matches = light_events.where(
            abs(light_events["tai_ns"].astype(float) * 1e-3 - 36000000 - charge_ts)
            <= window
        ).dropna()

        if not light_matches.empty:
            if event in match_dict:
                match_dict[event].append(
                    light_matches["event"].unique().astype(int).tolist()
                )
            else:
                match_dict[event] = light_matches["event"].unique().astype(int).tolist()

    return match_dict


# ### Charge
def get_track_stats(metrics, empty_ratio_lims=(0, 1), min_entries=2):
    track_dQdx = []
    track_length = []
    track_score = []
    track_z = []
    track_points = []
    events = []

    empty_count = 0
    short_count = 0
    for event, entry in metrics.items():
        for track, values in entry.items():
            if isinstance(track, str) or track <= 0:
                continue

            dQ = values["dQ"]
            dx = values["dx"]
            non_zero_mask = np.where(dQ > 0)[0]

            if len(non_zero_mask) < min_entries:
                short_count += 1
                continue

            empty_ratio = sum(
                dQ.iloc[non_zero_mask[0] : non_zero_mask[-1] + 1] == 0
            ) / (non_zero_mask[-1] - non_zero_mask[0] + 1)

            if empty_ratio > empty_ratio_lims[1] or empty_ratio < empty_ratio_lims[0]:
                empty_count += 1
                continue

            dQdx = (dQ / dx).rename("dQdx")
            dQdx = dQdx.iloc[non_zero_mask[0] : non_zero_mask[-1] + 1]
            x_range = dQdx.index
            position = [
                values["Fit_line"].to_point(t=-values["Fit_norm"] / 2 + t)
                for t in dQ.index
            ]
            position = position[non_zero_mask[0] : non_zero_mask[-1] + 1]

            track_dQdx.append(dQdx)
            track_points.append(pd.Series(position, index=dQdx.index, name="position"))
            track_length.append(values["Fit_norm"])
            track_score.append(values["RANSAC_score"])
            track_z.append(values["Fit_line"].point[2])
            events.append(event)

    print(f"Tracks with dead area outside {empty_ratio_lims} interval: {empty_count}")
    print(f"Tracks with less than {min_entries} entries: {short_count}")

    track_dQdx = pd.Series(track_dQdx)
    track_points = pd.Series(track_points)
    track_length = pd.Series(track_length)
    track_score = pd.Series(track_score)
    track_z = pd.Series(track_z)
    events = pd.Series(events)

    mask = (
        track_dQdx.apply(lambda x: x.notna().all())
        * track_length.notna()
        * track_score.notna()
        * track_z.notna()
    )

    print(f"\nRemaining tracks: {sum(mask)}\n")

    track_dQdx = track_dQdx[mask]
    track_points = track_points[mask]
    track_length = track_length[mask]
    track_score = track_score[mask]
    track_z = track_z[mask]
    events = events[mask]

    df = pd.DataFrame(
        [
            track_dQdx,
            track_points,
            track_length,
            track_score,
            track_z,
            events,
        ],
        index=[
            "track_dQdx",
            "track_points",
            "track_length",
            "track_score",
            "track_z",
            "event",
        ],
    ).T

    return df


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

                mask1 = (temp_y - params.quadrant_size / 2) - (
                    temp_x - params.quadrant_size / 2
                ) >= 0
                mask2 = (temp_y <= params.quadrant_size / 2 + buffer) & (
                    temp_y >= params.quadrant_size / 2 - buffer
                )
                mask3 = (temp_x <= params.quadrant_size / 2 + buffer) & (
                    temp_x >= params.quadrant_size / 2 - buffer
                )
                mask = mask1 | (mask2 & mask3)
                temp_x = (
                    temp_x[mask] - params.detector_x / 2 + params.quadrant_size * (k)
                )
                temp_y = (
                    -temp_y[mask] + params.detector_y / 2 - params.quadrant_size * (l)
                )
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


# Apply DBSCAN clustering
def cluster_hits(hitArray):
    # First stage clustering
    z_intervals = []
    first_stage = DBSCAN(eps=params.xy_epsilon, min_samples=params.min_samples).fit(
        hitArray[:, 0:2]
    )
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
    second_stage = DBSCAN(eps=params.xy_epsilon, min_samples=1).fit(
        second_stage_data[:, 0:2]
    )

    # Third stage clustering
    # Create a new array with z and labels
    third_stage_z = np.c_[second_stage.labels_ * 1e3, second_stage_data[:, 2]]
    labels = second_stage.labels_.copy()
    flag = labels > -1

    third_stage_data = third_stage_z[flag].copy()
    third_stage = DBSCAN(
        eps=params.z_epsilon, min_samples=params.min_samples, metric="chebyshev"
    ).fit(third_stage_data)

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
            score = estimator.score(
                hitArray[:, 0:last_column], hitArray[:, last_column]
            )
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
    projected_pitch = (
        np.dot(np.array([1, 1, 0]), abs(line_fit.direction.unit())) * params.pixel_pitch
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
        cyl_origin = line_fit.to_point(
            step - target_dh / 2 - h / 2
        )  # centering the step in the base of the cylinder
        cyl_height = line_fit.direction.unit() * target_dh
        cylinder_fit = Cylinder(
            cyl_origin,
            cyl_height,
            dr,
        )
        if ax is not None:
            cylinder_fit.plot_3d(ax)

        # Initialize variables to store the minimum and maximum points
        point_distances = []
        for point_idx, point in enumerate(hitArray):
            if not counted[point_idx] and cylinder_fit.is_point_within(point):
                counted[point_idx] = True
                dq_i.loc[step] += q[point_idx]

                point_distances.append(line_fit.transform_points([point]))

        # Calculate dh_i based on the distance between points
        max_distance = 0
        if len(point_distances) > 0:
            point_distances = np.unique(np.array(point_distances))
            intervals = np.diff(point_distances)
            total_distance = sum(
                intervals[intervals <= limit_pitch]
            ) + projected_pitch * 2 * (sum(intervals > limit_pitch))
            max_distance = min(
                abs(point_distances[-1] - point_distances[0]), total_distance
            )

        step_length = (
            max_distance + projected_pitch
            if max_distance > 0
            else projected_pitch if dq_i.loc[step] > 0 else 0
        )
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
                    labels[j] = (outlier_labels[i] + last_label) * (
                        1 if outlier_labels[i] > 0 else -1
                    )
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


# ### Light


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
                    (
                        line.distance_point(point)
                        if not plane.normal.is_parallel(line.direction)
                        else np.inf
                    )
                    for line in charge_lines
                ]
                if line_distances and min(line_distances) < np.inf:
                    charge_line = charge_lines[np.argmin(line_distances)]
                    projection = plane.project_line(charge_line)
                    projected_point = projection.project_point(point)
                    v_line = Line(point=projected_point, direction=[0, 0, 1])
                    intersection = charge_line.intersect_line(
                        v_line, check_coplanar=False
                    )
                    if all(
                        abs(projected_point - point)[:2] <= params.quadrant_size / 2
                    ):
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
                sipm_voxels_metrics["Fit_threshold"] = xyzl[light_variable][
                    inliers
                ].min()

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


# ### Helpers


def recal_params():
    params.dh_unit = params.xy_unit if params.xy_unit == params.z_unit else "?"
    params.light_unit = (
        "p.e." if params.light_variable == "integral" else f"p.e./{params.time_unit}"
    )
    if params.simulate_dead_area:
        params.detector_x = params.quadrant_size * 4
        params.detector_y = params.quadrant_size * 5

    params.first_chip = (2, 1) if params.detector_y == 160 else (1, 1)

    print("\nRecalculating parameters:")
    print(
        f"\n dh_unit set to {params.dh_unit}\n",
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
    condition = ((count / max_count).round(3) >= min_count_ratio) & (
        (std / max_std).round(3) <= max_std_ratio
    )
    vline = x_range[
        (
            np.where(condition)[0][-1]
            if np.any(condition)
            else (count / max_count > min_count_ratio).argmax()
        )
    ]

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
        ax.tick_params(
            axis="both", direction="inout", which="major", top=True, right=True
        )
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
    filtered_centers_x, filtered_centers_y = np.array(
        np.meshgrid(bin_centers_x, bin_centers_y)
    )
    filtered_centers_x = filtered_centers_x[n.T > min_n]
    filtered_centers_y = filtered_centers_y[n.T > min_n]
    dbscan = DBSCAN(
        eps=eps, min_samples=int(np.sqrt(len(filtered_centers_y))), metric="chebyshev"
    ).fit(np.c_[filtered_centers_x / scale[0], filtered_centers_y / scale[1]])
    return filtered_centers_x, filtered_centers_y, dbscan.labels_


# Peak finding algorithm for integration
def integrate_peaks(waveform, buffer_size=10, height=0.1, prominence=0.05):
    # Find peaks in the filtered waveform
    peaks, properties = signal.find_peaks(
        waveform, height=height, prominence=prominence
    )

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
            if (
                "Total_light" in metric
                and min_light <= metric["Total_light"] <= max_light
            ) or "Total_light" not in metric:
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
                if (
                    non_track_keys
                    < len(candidate_metric)
                    <= max_tracks + non_track_keys
                ):
                    filtered_metrics[event_idx] = candidate_metric

    print(f"{len(filtered_metrics)} metrics remaining")

    # Save the filtering parameters to a JSON file
    output_path = os.path.join(params.work_path, params.file_label)
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

    for file in tqdm(glob.glob(f"{params.work_path}/**/*metrics*.pkl"), leave=True):
        folder = file.split("/")[0]
        tqdm.write(folder)
        with open(file, "rb") as f:
            metric = pickle.load(f)
            for key, value in tqdm(metric.items(), leave=False):
                combined_metrics[f"{folder}_{key}"] = value

    output_path = os.path.join(params.work_path, "combined")
    os.makedirs(output_path, exist_ok=True)

    with open(os.path.join(output_path, "metrics_combined.pkl"), "wb") as o:
        pickle.dump(combined_metrics, o)

    print("Done")

    return combined_metrics


# ## Uproot functions


# Uproot
def load_charge(file_name, events=None):
    with uproot.open(file_name) as f:
        if "events" in f:
            charge_df = f["events"].arrays(library="pd")
        elif "HitTree" in f:
            charge_df = f["HitTree"].arrays(library="pd")
            charge_df.rename({"eid": "event", "iev": "eventID"}, axis=1, inplace=True)

        charge_df.set_index("eventID", inplace=True)
        if events is not None:
            charge_df = charge_df.loc[events]

    return charge_df


def load_light(file_name, deco=True, events=None, mask=True, keep_rwf=False):
    light_df = pd.DataFrame()
    with uproot.open(file_name) as f:
        if deco:
            tree = f["decowave"]
        else:
            tree = f["rwf_array"]

        scaling_par = 655.340
        if "scaling_par" in f:
            scaling_par = f["scaling_par"].value

        for idx, arrays in enumerate(tree.iterate(library="np")):
            df = pd.DataFrame.from_dict(arrays, orient="index").T
            df.dropna()
            if events is not None:
                df = df[df["event"].isin(events)]

            df[["sn", "ch"]] = df[["sn", "ch"]].astype(int)

            if mask:
                df = df[
                    df[["sn", "ch"]].apply(
                        lambda x: get_sipm_mask(x.iloc[0], x.iloc[1]), axis=1
                    )
                ]

            if df.empty:
                continue

            df[["x", "y"]] = df[["sn", "ch"]].apply(
                lambda x: pd.Series(sipm_to_xy(x.iloc[0], x.iloc[1])), axis=1
            )

            if deco:
                df["rwf"] = df["decwfm"]

            df["rwf"] = df["rwf"].apply(lambda x: x / scaling_par)

            df[["integral", "properties"]] = df["rwf"].apply(
                lambda x: pd.Series(
                    (np.nan, {}) if any(np.isnan(x)) else integrate_peaks(x)
                )
            )
            df["peak"] = df["properties"].apply(
                lambda x: (
                    max(x["peak_heights"])
                    if "peak_heights" in x and len(x["peak_heights"]) > 0
                    else np.nan
                )
            )

            columns = ["event", "tai_ns", "sn", "ch", "peak", "integral", "x", "y"]
            if keep_rwf:
                columns.append("rwf")

            df = df[columns]
            light_df = pd.concat([light_df, df], ignore_index=True)

    return light_df


# ## Plotting
def prepare_event(event, charge_df, light_df=None, match_dict=None):
    if event not in charge_df.index:
        tqdm.write(f"Event {event} not found in {params.file_label}")
        return None, None, None

    light_event = None
    if light_df is not None:
        light_indices = light_df["event"].copy()

        if event in match_dict:
            light_event = match_dict.get(event)[0]
            light_matches = light_indices[light_indices == light_event].index
            light_event = light_df.loc[light_matches].dropna(
                subset=params.light_variable
            )
        else:
            print(f"No light event found for event {event} in {params.file_label}")

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
        non_zero_mask = (charge_event["x"] != 0) * (
            charge_event["y"] != 0
        )  # Remove (0,0) entries

        noisy_channels_mask = ~channel_ids.isin(
            [ch[0] for ch in params.channel_disable_list]
        )  # Disable channel 7

        mask = non_zero_mask * noisy_channels_mask  # Full hits mask

        # Apply boolean indexing to x, y, and z arrays
        charge_event = charge_event[mask]
    else:
        mask = np.full(len(charge_event["q"]), True)

    charge_event["q"] = charge_event["q"] * params.charge_gain  # Convert mV to ke

    return charge_event, light_event, mask


class OOMFormatter(ScalarFormatter):
    def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=True):
        self.oom = order
        self.fformat = fformat
        ScalarFormatter.__init__(self, useOffset=offset, useMathText=mathText)

    def _set_order_of_magnitude(self):
        self.orderOfMagnitude = self.oom

    def _set_format(self, vmin=None, vmax=None):
        self.format = self.fformat
        if self._useMathText:
            self.format = r"$\mathdefault{%s}$" % self.format


def set_common_ax_options(ax=None, cbar=None):
    if ax is not None:
        ax.tick_params(
            axis="both",
            direction="inout",
            which="major",
            top=True,
            right=True,
            labelsize=params.tick_font_size,
        )
        ax.set_axisbelow(True)
        ax.grid(alpha=0.25)
        ax.set_title(ax.get_title(), fontsize=params.title_font_size)
        ax.set_ylabel(ax.get_ylabel(), fontsize=params.label_font_size)
        ax.set_xlabel(ax.get_xlabel(), fontsize=params.label_font_size)
        if hasattr(ax, "get_zlabel"):
            ax.set_zlabel(ax.get_zlabel(), fontsize=params.label_font_size)

        if not ax.get_xscale() == "log":
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            if ax.get_xlim()[1] > 1.1:
                ax.xaxis.set_major_locator(MaxNLocator(integer=(ax.get_xlim()[1] > 2)))
                if ax.get_xlim()[1] > 1e3:
                    ax.xaxis.set_major_formatter(OOMFormatter(3, "%1.1f"))

        if not ax.get_yscale() == "log":
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            if ax.get_ylim()[1] > 1.1:
                ax.yaxis.set_major_locator(MaxNLocator(integer=(ax.get_ylim()[1] > 2)))
                if ax.get_ylim()[1] > 1e3:
                    ax.yaxis.set_major_formatter(OOMFormatter(3, "%1.1f"))

    if cbar is not None:
        cbar.ax.tick_params(labelsize=params.tick_font_size)
        cbar.set_label(cbar.ax.get_ylabel(), fontsize=params.label_font_size)


# ### Event display


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


def create_ed_axes(event_idx, charge, light):
    fig = plt.figure(figsize=(14, 6))
    ax3d = fig.add_subplot(121, projection="3d")
    ax2d = fig.add_subplot(122)
    fig.suptitle(
        f"Event {event_idx} - Charge = {charge} {params.q_unit} - Light = {light} {params.light_unit}"
    )
    grid_color = plt.rcParams["grid.color"]

    # Draw dead areas
    for i, j in [(3, 3), (4, 4), (5, 4), (6, 4), (4, 2)]:
        x = np.array([0, params.quadrant_size])
        y = -np.array([0, params.quadrant_size])
        ax2d.plot(
            np.power(-1, params.flip_x)
            * (
                x
                - params.detector_x / 2
                + params.quadrant_size * (j - params.first_chip[1])
            ),
            (
                y
                + params.detector_y / 2
                - params.quadrant_size * (i - params.first_chip[0])
            ),
            c=grid_color,
            lw=1,
        )
        if i == 4 and j == 2:
            x = np.array([params.quadrant_size / 2, 0])
            y = -np.array([params.quadrant_size, params.quadrant_size / 2])
        ax2d.plot(
            np.power(-1, params.flip_x)
            * (
                x[::-1]
                - params.detector_x / 2
                + params.quadrant_size * (j - params.first_chip[1])
            ),
            (
                y
                + params.detector_y / 2
                - params.quadrant_size * (i - params.first_chip[0])
            ),
            c=grid_color,
            lw=1,
        )

    # Adjust axes
    for ax in [ax3d, ax2d]:
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim([-params.detector_x / 2, params.detector_x / 2])
        ax.set_ylim([-params.detector_y / 2, params.detector_y / 2])
        ax.set_xlabel(f"x [{params.xy_unit}]")
        ax.set_ylabel(f"y [{params.xy_unit}]")
        ax.set_xticks(
            np.linspace(
                -params.detector_x / 2,
                params.detector_x / 2,
                (params.detector_x // params.quadrant_size) + 1,
            )
        )
        ax.set_yticks(
            np.linspace(
                -params.detector_y / 2,
                params.detector_y / 2,
                (params.detector_y // params.quadrant_size) + 1,
            )
        )
        ax.grid()

    ax2d.xaxis.set_minor_locator(AutoMinorLocator(8))
    ax2d.yaxis.set_minor_locator(AutoMinorLocator(8))
    ax2d.tick_params(axis="both", which="both", right=True, top=True)

    # Adjust z-axis
    ax3d.set_zlabel(f"z [{params.z_unit}]")
    # ax3d.zaxis.set_major_locator(MaxNLocator(integer=True))

    return fig, (ax2d, ax3d)


def event_display(
    event_idx,
    charge_df,
    light_df=None,
    plot_cyl=False,
    metrics=None,
):
    if len(charge_df) < 2:
        return None
    if light_df is None:
        light_df = pd.DataFrame(columns=["x", "y", params.light_variable])

    # Plot the hits
    fig, axes = create_ed_axes(
        event_idx,
        round(sum(charge_df["q"])),
        round(sum(light_df[params.light_variable])),
    )
    ax2d = axes[0]
    ax3d = axes[1]

    # Group by x and y coordinates and sum the z values
    unique_points, indices = np.unique(
        charge_df[["x", "y"]], axis=0, return_inverse=True
    )
    q_sum = np.bincount(indices, weights=charge_df["q"])

    # Plot the hits
    plot3d = ax3d.scatter(
        charge_df["x"],
        charge_df["y"],
        charge_df["z"],
        c=charge_df["q"],
        marker="s",
        s=round((30**4) / (params.detector_x * params.detector_y)),
        vmin=q_sum.min(),
        vmax=q_sum.max(),
    )
    plot2d = ax2d.scatter(
        unique_points[:, 0],
        unique_points[:, 1],
        c=q_sum,
        marker="s",
        s=round((30**4) / (params.detector_x * params.detector_y)),
        vmin=q_sum.min(),
        vmax=q_sum.max(),
    )
    cbar = plt.colorbar(plot2d)
    cbar.set_label(f"charge [{params.q_unit}]")

    # Cluster the hits
    labels = cluster_hits(charge_df[["x", "y", "z"]].to_numpy())

    # Fit clusters
    # Fit clusters
    if metrics is None:
        metrics = fit_hit_clusters(
            charge_df[["x", "y", "z"]].to_numpy(),
            charge_df["q"].to_numpy(),
            labels,
            ax2d,
            ax3d,
            plot_cyl,
        )
    else:
        for track_idx, values in metrics.items():
            if isinstance(track_idx, np.int64):
                track = values["Fit_line"]
                track_norm = values["Fit_norm"]
                track.plot_2d(
                    ax2d,
                    t_1=-track_norm / 2,
                    t_2=track_norm / 2,
                    c="red",
                    label=f"Track {track_idx}",
                    zorder=10,
                )
                track.plot_3d(
                    ax3d,
                    t_1=-track_norm / 2,
                    t_2=track_norm / 2,
                    c="red",
                    label=f"Track {track_idx}",
                )

    # Draw missing SiPMs
    grid_color = plt.rcParams["grid.color"]

    # Draw SiPMs
    sipm_plot = ax2d.scatter(
        light_df["x"],
        light_df["y"],
        c=light_df[params.light_variable],
        marker="s",
        s=200,
        linewidths=1.5,
        edgecolors=grid_color,
        zorder=6,
    )

    # Draw SiPMs
    side_size = 6
    vertices_x = np.array([1, 1, -1, -1, 1]) * side_size / 2
    vertices_y = np.array([1, -1, -1, 1, 1]) * side_size / 2
    light_xy = light_df[["x", "y"]].apply(tuple, axis=1)
    for missing_index in range(
        params.detector_x * params.detector_y // (params.quadrant_size**2)
    ):
        col = (
            -params.detector_x / 2
            + params.quadrant_size / 2
            + (missing_index % (params.detector_x // params.quadrant_size)) * 32
        )
        row = (
            params.detector_y / 2
            - params.quadrant_size / 2
            - (missing_index // (params.detector_x // params.quadrant_size)) * 32
        )
        square = create_square((col, row), side_size)
        if not light_xy.isin([(col, row)]).any():
            ax2d.fill(col + vertices_x, vertices_y + row, c=grid_color, zorder=5)
            ax3d.add_collection3d(Poly3DCollection(square, color=grid_color))
        else:
            ax3d.add_collection3d(
                Poly3DCollection(
                    square,
                    facecolors=sipm_plot.to_rgba(
                        light_df[params.light_variable][light_xy == (col, row)]
                    ),
                    linewidths=0.5,
                    edgecolors=grid_color,
                )
            )

    sipm_cbar = plt.colorbar(sipm_plot)
    sipm_cbar.set_label(rf"Light {params.light_variable} [{params.light_unit}]")

    ax3d.set_zlim([0, max(ax3d.get_zlim()[1], charge_df["z"].max())])
    # ax3d.view_init(160, 110, -85)
    ax3d.view_init(30, 20, 100)
    # ax3d.view_init(0, 0, 0)
    # ax3d.view_init(0, 0, 90)
    fig.tight_layout()

    if params.save_figures:
        output_path = os.path.join(params.work_path, params.file_label, str(event_idx))
        os.makedirs(output_path, exist_ok=True)
        fig.savefig(
            os.path.join(output_path, f"event_{event_idx}.pdf"),
            dpi=300,
            bbox_inches="tight",
        )

    return metrics


def plot_fake_data(z_range, buffer=1):
    fake_data = generate_dead_area(z_range, buffer)
    fake_x, fake_y, fake_z = fake_data[:, 0], fake_data[:, 1], fake_data[:, 2]

    fig = plt.figure()

    ax = fig.add_subplot(111)
    ax.set_aspect("equal", adjustable="box")
    ax.scatter(fake_x, fake_y, marker="s", s=20)
    ax.set_xlim([-params.detector_x / 2, params.detector_x / 2])
    ax.set_ylim([-params.detector_y / 2, params.detector_y / 2])
    ax.set_xlabel(f"x [{params.xy_unit}]")
    ax.set_ylabel(f"y [{params.xy_unit}]")
    ax.set_xticks(np.linspace(-params.detector_x / 2, params.detector_x / 2, 5))
    ax.set_yticks(np.linspace(-params.detector_y / 2, params.detector_y / 2, 6))
    ax.xaxis.set_minor_locator(AutoMinorLocator(8))
    ax.yaxis.set_minor_locator(AutoMinorLocator(8))
    ax.grid()
    ax.tick_params(axis="both", which="both", top=True, right=True)
    fig.tight_layout()

    output_path = os.path.join(params.work_path, params.file_label)
    os.makedirs(output_path, exist_ok=True)
    fig.savefig(
        os.path.join(output_path, "fake_data_map.pdf"), dpi=300, bbox_inches="tight"
    )


# ### Tracks


# Plot dQ versus X
def plot_dQ(dQ_series, dx_series, event_idx, track_idx, interpolate=False):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax_twinx = ax.twinx()

    target_dx = dx_series.index.diff()[-1]
    fig.suptitle(
        rf"Event {event_idx} - Track {track_idx} - $dx = {round(target_dx,2)}$ {params.dh_unit}"
    )

    non_zero_indices = np.where(dQ_series > 0)[0]
    mean_dQ = np.mean(dQ_series.iloc[non_zero_indices])
    mean_dx = np.mean(dx_series.iloc[non_zero_indices])
    mean_dQdx = (dQ_series / dx_series).iloc[non_zero_indices].mean()

    # Check if there are non-zero values in dQ_array
    if non_zero_indices.size > 0:
        # Find the first non-zero index and get 1 index before it
        first_index = max(non_zero_indices[0] - 1, 0)

        # Find the last non-zero index and get 2 indices after it
        last_index = min(non_zero_indices[-1] + 2, len(dQ_series))

        new_dQ_series = dQ_series.iloc[first_index:last_index].copy()
        new_dx_series = dx_series.iloc[first_index:last_index].copy()

        if interpolate:
            new_dQ_series[1:-1] = np.where(
                new_dQ_series[1:-1] == 0,
                mean_dQ,
                new_dQ_series[1:-1],
            )
            new_dx_series[1:-1] = np.where(
                new_dx_series[1:-1] == 0,
                mean_dx,
                new_dx_series[1:-1],
            )

        dQ_series = new_dQ_series
        dx_series = new_dx_series

    ax.axhline(
        mean_dQdx,
        ls="--",
        c="red",
        label=rf"Mean = ${round(mean_dQdx,2)}$ {params.q_unit} {params.dh_unit}$^{{-1}}$",
        lw=1,
    )

    ax.step(dQ_series.index, dQ_series / dx_series, where="mid")
    # ax.scatter(dQ_series.index, dQ_series / dx_series)
    ax.set_xlabel(rf"$x$ [{params.dh_unit}]")
    ax.set_ylabel(rf"$dQ/dx$ [{params.q_unit} {params.dh_unit}$^{{-1}}$]")

    ax_twinx.step(dQ_series.index, np.cumsum(dQ_series), color="C1", where="mid")
    ax_twinx.set_ylabel(f"Q [{params.q_unit}]")

    for axes in [ax, ax_twinx]:
        set_common_ax_options(axes)

    h1, l1 = ax.get_legend_handles_labels()
    ax_twinx.legend(h1, l1, loc="lower center")

    ax.legend(loc="lower center")

    fig.tight_layout()
    if params.save_figures:
        output_path = os.path.join(params.work_path, params.file_label, str(event_idx))
        os.makedirs(output_path, exist_ok=True)
        fig.savefig(
            os.path.join(
                output_path, f"dQ_E{event_idx}_T{track_idx}_{round(target_dx,2)}.pdf"
            ),
            dpi=300,
            bbox_inches="tight",
        )


def plot_track_stats(
    metrics,
    limit_xrange=True,
    min_score=0.5,
    empty_ratio_lims=(0, 1),
    min_entries=2,
    lognorm=True,
    profile=False,
    bins=[40, 40],
):
    df = get_track_stats(
        metrics, empty_ratio_lims=empty_ratio_lims, min_entries=min_entries
    )

    track_dQdx = df["track_dQdx"]
    track_length = df["track_length"].astype(float)
    track_score = df["track_score"].astype(float)
    track_z = df["track_z"].astype(float)
    track_cv_dQdx = track_dQdx.apply(lambda x: x.std() / x.mean()).astype(float)
    track_mean_dQdx = track_dQdx.apply(lambda x: x.mean()).astype(float)

    score_mask = (track_score >= min_score).to_numpy()
    score_bool = (1 - score_mask).sum() > 0

    print(f"Tracks with score < {min_score}: {len(track_dQdx)-sum(score_mask)}")
    # print(f"\nRemaining tracks: {sum(score_mask)}\n")

    dQdx_series = pd.concat(track_dQdx.to_list())
    dQdx_series = dQdx_series[dQdx_series > 0].dropna().sort_index()
    cut_dQdx_series = pd.concat(track_dQdx[score_mask].to_list())
    cut_dQdx_series = cut_dQdx_series[cut_dQdx_series > 0].dropna().sort_index()

    # print("\ndQ/dx stats:")
    # TODO if ipython
    # display(dQdx_series.describe())

    # 1D histograms
    fig1 = plt.figure(figsize=(14, 6))

    ax11 = fig1.add_subplot(121)
    ax12 = fig1.add_subplot(122)

    limit = np.percentile(dQdx_series.values, 99) if limit_xrange else np.inf

    n_all11, bins_all11, patches_all11 = ax11.hist(
        dQdx_series[dQdx_series <= limit].values, bins=bins[0], label="All tracks"
    )

    n_all12, bins_all12, patches_all12 = ax12.hist(
        track_length, bins=bins[0], label="All tracks"
    )

    if score_bool:
        n11, edges11, patches11 = ax11.hist(
            cut_dQdx_series[cut_dQdx_series <= limit].values,
            bins=bins_all11,
            label=rf"Score $\geq {min_score}$",
        )
        ax12.hist(
            track_length[score_mask],
            bins=bins_all12,
            label=rf"Score $\geq {min_score}$",
        )

    bin_centers_all11 = (bins_all11[1:] + bins_all11[:-1]) / 2
    p0 = (
        bin_centers_all11[n_all11.argmax()],
        np.std(bin_centers_all11) / 100,
        np.std(bin_centers_all11) / 2,
        max(n_all11),
    )

    popt, pcov = curve_fit(
        pylandau.langau,
        bin_centers_all11[bin_centers_all11 > 3000],
        n_all11[bin_centers_all11 > 3000],
        absolute_sigma=True,
        p0=p0,
        bounds=(
            (
                bin_centers_all11[max(n_all11.argmax() - 3, 0)],
                0,
                0,
                0,
            ),
            (
                bin_centers_all11[
                    min(n_all11.argmax() + 3, len(bin_centers_all11) - 1)
                ],
                np.inf,
                np.inf,
                np.inf,
            ),
        ),
    )

    ax11.plot(
        fit_x := np.linspace(bins_all11[0], bins_all11[-1], 1000),
        pylandau.langau(fit_x, *popt),
        "r-",
        label=r"fit: $\mu$=%5.1f, $\eta$=%5.1f, $\sigma$=%5.1f, A=%5.1f" % tuple(popt),
    )

    ax11.set_xlabel(
        rf"$dQ/dx$ [{params.q_unit} {params.dh_unit}$^{{-1}}$]",
        fontsize=params.label_font_size,
    )
    ax11.set_title(f"{len(track_dQdx)} tracks", fontsize=params.title_font_size)
    ax12.set_title(f"{len(track_length)} tracks", fontsize=params.title_font_size)

    # 2D histograms
    def hist2d(x, y, ax, bins, lognorm, fit="Log", profile=False):
        if profile:
            hist, x_edges, y_edges = np.histogram2d(x, y, bins=bins)

            y_means = [
                np.mean(y[(x >= x_edges[i]) & (x < x_edges[i + 1])])
                for i in range(len(x_edges) - 1)
            ]
            y_stds = [
                np.std(y[(x >= x_edges[i]) & (x < x_edges[i + 1])])
                for i in range(len(x_edges) - 1)
            ]
            x_values = (x_edges[1:] + x_edges[:-1]) / 2
            bin_widths = [
                (x_edges[i + 1] - x_edges[i]) / 2 for i in range(len(x_edges) - 1)
            ]
            ax.errorbar(x_values, y_means, yerr=y_stds, xerr=bin_widths, fmt="o")

        else:
            hist2d = ax.hist2d(
                x,
                y,
                bins=bins,
                cmin=1,
                norm=LogNorm() if lognorm else None,
            )
        if fit == "Log":
            x_fit = np.log(x)
        elif fit == "Linear":
            x_fit = x
        else:
            return

        try:
            fit_p = np.polyfit(x_fit, y, 1)
        except:
            if fit == "Log":
                x_fit = x  # Try linear fit as a fallback
                fit = "Linear"
            elif fit == "Linear":
                x_fit = np.log(x)  # Try log fit as a fallback
                fit = "Log"
            try:
                fit_p = np.polyfit(x_fit, y, 1)
            except:
                return

        p = np.poly1d(fit_p)
        x_plot = np.arange(min(x), max(x), 1)

        if fit == "Log":
            y_plot = p(np.log(x_plot))
        else:
            y_plot = p(x_plot)

        ax.plot(x_plot, y_plot, c="salmon", ls="-", label=f"{fit} fit")

    fig2 = plt.figure(figsize=(14, 6))
    ax21 = fig2.add_subplot(121)
    ax22 = fig2.add_subplot(122)

    fig2.suptitle(f"{len(track_dQdx)} tracks", fontsize=params.title_font_size)
    ax21.set_ylabel(
        rf"Mean $dQ/dx$ [{params.q_unit} {params.dh_unit}$^{{-1}}$]",
        fontsize=params.label_font_size,
    )
    ax21.set_title("Mean dQ/dx vs. Track length", fontsize=params.title_font_size)
    ax22.set_ylabel(rf"$dQ/dx$ CV", fontsize=params.label_font_size)
    ax22.set_title("dQ/dx CV vs. Track length", fontsize=params.title_font_size)

    hist2d21 = hist2d(
        track_length, track_mean_dQdx, ax21, bins, lognorm, fit="Log", profile=profile
    )

    hist2d22 = hist2d(
        track_length, track_cv_dQdx, ax22, bins, lognorm, fit="Linear", profile=profile
    )

    fig4 = plt.figure(figsize=(7, 6))
    ax4 = fig4.add_subplot(111)
    ax4.set_ylabel(f"Fit score")
    ax4.set_title("Fit score vs. Track length")

    hist2d4 = hist2d(
        track_length,
        track_score,
        ax4,
        [bins[0], 40],
        lognorm,
        fit="Log",
        profile=profile,
    )

    fig5 = plt.figure(figsize=(7 + 7 * score_bool, 6))
    ax51 = fig5.add_subplot(111 + 10 * score_bool)
    ax51.set_ylabel(
        rf"$dQ/dx$ [{params.q_unit} {params.dh_unit}$^{{-1}}$]",
        fontsize=params.label_font_size,
    )
    ax51.set_xlabel(
        rf"Residual range [{params.dh_unit}]", fontsize=params.label_font_size
    )
    ax51.set_title(rf"{len(track_dQdx)} tracks", fontsize=params.title_font_size)

    hist2d(
        dQdx_series.index,
        dQdx_series,
        ax51,
        bins,
        lognorm,
        fit="Linear",
        profile=profile,
    )

    fig6 = plt.figure(figsize=(7 + 7 * score_bool, 6))
    ax61 = fig6.add_subplot(111 + 10 * score_bool)
    ax61.set_ylabel(
        rf"Mean $dQ/dx$ [{params.q_unit} {params.dh_unit}$^{{-1}}$]",
        fontsize=params.label_font_size,
    )
    ax61.set_xlabel(
        rf"Mean anode distance [{params.z_unit}]", fontsize=params.label_font_size
    )
    ax61.set_title(rf"{len(track_z)} tracks", fontsize=params.title_font_size)

    hist2d(track_z, track_mean_dQdx, ax61, bins, lognorm, fit="Linear", profile=profile)

    # fig7 = plt.figure(figsize=(7 + 7 * score_bool, 6))
    # ax71 = fig7.add_subplot(111 + 10 * score_bool)
    # ax71.set_ylabel(rf"$dQ/dx$ [{params.q_unit} {params.dh_unit}$^{{-1}}$]", fontsize=label_size)
    # ax71.set_xlabel(rf"Anode distance [{params.z_unit}]", fontsize=label_size)
    # ax71.set_title(rf"{len(track_z)} tracks", fontsize=title_size)

    # dq_z_series = pd.concat(dq_z_list)
    # dq_z_series = dq_z_series[dq_z_series > 0].dropna().sort_index()

    # hist2d(
    #     dq_z_series.index,
    #     dq_z_series,
    #     ax71,
    #     bins,
    #     lognorm,
    #     fit="Linear",
    #     profile=profile,
    # )

    # def exp_decay(x, tau, init):
    #     return init * np.exp(-x / tau)

    # popt, pcov = curve_fit(
    #     exp_decay,
    #     track_z,
    #     track_mean_dQdx,
    #     p0=[23, 5000],
    # )

    # plt.plot(
    #     track_z,
    #     exp_decay(track_z, *popt),
    #     "r-",
    #     label="fit: tau=%5.3f, init=%5.3f" % tuple(popt),
    # )

    # print(popt)

    axes = [ax11, ax12, ax21, ax22, ax4, ax51, ax61]  # , ax71]
    figs = [fig1, fig2, fig4, fig5, fig6]  # , fig7]

    if score_bool:
        # 2D histograms after RANSAC score cut
        fig3 = plt.figure(figsize=(14, 6))
        ax31 = fig3.add_subplot(121)
        ax32 = fig3.add_subplot(122)
        ax31.set_ylabel(rf"Mean $dQ/dx$ [{params.q_unit} {params.dh_unit}$^{{-1}}$]")
        ax31.set_title(rf"Mean dQ/dx vs. Track length")
        ax32.set_ylabel(rf"$dQ/dx$ CV")
        ax32.set_title(rf"dQ/dx CV vs. Track length")
        fig3.suptitle(
            rf"Fit score $\geq {min_score}$ ({round(sum(score_mask)/len(score_mask)*100)}% of tracks)",
            fontsize=params.title_font_size,
        )

        figs.append(fig3)
        axes.extend([ax31, ax32])

        hist2d31 = hist2d(
            track_length[score_mask],
            track_mean_dQdx[score_mask],
            ax31,
            bins,
            lognorm,
            fit="Log",
            profile=profile,
        )

        hist2d32 = hist2d(
            track_length[score_mask],
            track_cv_dQdx[score_mask],
            ax32,
            bins,
            lognorm,
            fit="Linear",
            profile=profile,
        )

        ax52 = fig5.add_subplot(122)
        axes.append(ax52)
        ax52.set_ylabel(
            rf"$dQ/dx$ [{params.q_unit} {params.dh_unit}$^{{-1}}$]",
            fontsize=params.label_font_size,
        )
        ax52.set_xlabel(
            rf"Residual range [{params.dh_unit}]", fontsize=params.label_font_size
        )
        ax52.set_title(
            rf"Fit score $\geq {min_score}$ ({round(sum(score_mask)/len(score_mask)*100)}% of tracks)",
            fontsize=params.title_font_size,
        )
        fig5.suptitle("dQ/dx vs. Residual range", fontsize=params.title_font_size)

        hist2d(
            cut_dQdx_series.index,
            cut_dQdx_series,
            ax52,
            bins,
            lognorm,
            fit="Linear",
            profile=profile,
        )

        ax62 = fig6.add_subplot(122)
        axes.append(ax62)
        ax62.set_ylabel(
            rf"Mean $dQ/dx$ [{params.q_unit} {params.dh_unit}$^{{-1}}$]",
            fontsize=params.label_font_size,
        )
        ax62.set_xlabel(
            rf"Mean anode distance [{params.z_unit}]", fontsize=params.label_font_size
        )
        ax62.set_title(
            rf"Fit score $\geq {min_score}$ ({round(sum(score_mask)/len(score_mask)*100)}% of tracks)",
            fontsize=params.title_font_size,
        )
        fig6.suptitle(
            "Mean dQ/dx vs. Mean anode distance", fontsize=params.title_font_size
        )

        hist2d(
            track_z[score_mask],
            track_mean_dQdx[score_mask],
            ax62,
            bins,
            lognorm,
            fit="Linear",
            profile=profile,
        )

        # ax72 = fig7.add_subplot(122)
        # axes.append(ax72)
        # ax72.set_ylabel(rf"$dQ/dx$ [{params.q_unit} {params.dh_unit}$^{{-1}}$]", fontsize=label_size)
        # ax72.set_xlabel(rf"Anode distance [{params.z_unit}]", fontsize=label_size)
        # ax72.set_title(
        #     rf"Fit score $\geq {min_score}$ ({round(sum(score_mask)/len(score_mask)*100)}% of tracks)", fontsize=title_size
        # )
        # fig7.suptitle("dQ/dx vs. Anode distance", fontsize=title_size)

        # cut_dq_z_series = pd.concat(
        #     [series for i, series in enumerate(dq_z_list) if score_mask[i]]
        # )
        # cut_dq_z_series = cut_dq_z_series[cut_dq_z_series > 0].dropna().sort_index()

        # hist2d(
        #     cut_dq_z_series.index,
        #     cut_dq_z_series,
        #     ax72,
        #     bins,
        #     lognorm,
        #     fit="Linear",
        #     profile=profile,
        # )

    max_track_legth = np.sqrt(
        params.detector_x**2 + params.detector_y**2 + params.detector_z**2
    )
    max_track_legth_xy = np.sqrt(params.detector_x**2 + params.detector_y**2)
    print("Max possible track length", round(max_track_legth, 2), "mm")
    print("Max possible track length on xy plane", round(max_track_legth_xy, 2), "mm")
    print("Max possible vertical track length", params.detector_y, "mm")

    for ax in axes:
        if ax == ax11 or ax == ax12:
            ax.set_ylabel("Counts")
        if ax != ax11:
            if not (
                ax == ax51
                or ax == ax61
                # or ax == ax71
                or (score_bool and (ax == ax52 or ax == ax62))  # or ax == ax72))
            ):
                ax.set_xlabel(f"Track length [{params.dh_unit}]")
            if max(track_length) > params.detector_y:
                ax.axvline(
                    params.detector_y, c="g", ls="--", label="Max vertical length"
                )
            if max(track_length) > max_track_legth_xy:
                ax.axvline(
                    max_track_legth_xy, c="orange", ls="--", label=r"Max length in $xy$"
                )
            if max(track_length) > max_track_legth:
                ax.axvline(max_track_legth, c="r", ls="--", label="Max length")

            if ax != ax12:
                if limit_xrange:
                    xlim = ax.get_xlim()
                    ax.set_xlim(xlim[0], min(max_track_legth + 10, xlim[1]))

                cbar = ax.get_figure().colorbar(ax.collections[0])
                cbar.set_label("Counts" + (" [log]" if lognorm else ""))
                set_common_ax_options(cbar=cbar)
        if not (not score_bool and ax == ax11):
            ax.legend(loc="lower right" if ax == ax4 else "upper right")

        set_common_ax_options(ax=ax)

    for fig in figs:
        fig.tight_layout()

    if params.save_figures:
        entries = len(track_dQdx)
        output_path = os.path.join(params.work_path, params.file_label)
        os.makedirs(output_path, exist_ok=True)

        fig1.savefig(
            os.path.join(
                output_path, f"track_stats_1D_hist_{params.file_label}_{entries}.pdf"
            ),
            dpi=300,
            bbox_inches="tight",
        )
        fig2.savefig(
            os.path.join(
                output_path,
                f"track_stats_2D_hist_{params.file_label}_{entries}{'_profile' if profile else ''}.pdf",
            ),
            dpi=300,
            bbox_inches="tight",
        )
        fig4.savefig(
            os.path.join(
                output_path,
                f"track_stats_score_{params.file_label}_{entries}{'_profile' if profile else ''}.pdf",
            ),
            dpi=300,
            bbox_inches="tight",
        )
        fig5.savefig(
            os.path.join(
                output_path,
                f"track_stats_dQdx_{params.file_label}_{entries}{'_profile' if profile else ''}.pdf",
            ),
            dpi=300,
            bbox_inches="tight",
        )
        fig6.savefig(
            os.path.join(
                output_path,
                f"track_stats_dQdx_z_{params.file_label}_{entries}{'_profile' if profile else ''}.pdf",
            ),
            dpi=300,
            bbox_inches="tight",
        )
        # fig7.savefig(
        #     os.path.join(output_path, f"track_stats_dQ_z_{params.file_label}_{entries}{'_profile' if profile else ''}.pdf"),
        #     dpi=300,
        #     bbox_inches="tight",
        # )
        if score_bool:
            fig3.savefig(
                os.path.join(
                    output_path,
                    f"track_stats_2D_hist_cut_{params.file_label}_{entries}{'_profile' if profile else ''}.pdf",
                ),
                dpi=300,
                bbox_inches="tight",
            )
    return df


# ### Tracks and light


def plot_light_geo_stats(
    metrics,
    limit_xrange=True,
    light_max=None,
    min_count_ratio=0.99,
    max_std_ratio=0.2,
    single_track=True,
    lognorm=True,
):
    sipm_distance = []
    sipm_angle = []
    sipm_light = []

    for metric in metrics.values():
        if single_track and len(metric.keys()) > (
            1 + sum(1 for key in metric if isinstance(key, str))
        ):
            continue
        for track_idx, values in metric.items():
            if not isinstance(track_idx, str) and track_idx > 0:
                sipms = values["SiPM"]
                for light in sipms.values():
                    sipm_distance.append(light["distance"])
                    sipm_angle.append(light["angle"])
                    sipm_light.append(light[params.light_variable])

    sipm_distance = np.array(sipm_distance)
    sipm_angle = np.array(sipm_angle)
    sipm_light = np.array(sipm_light)

    max_distance = np.sqrt(
        params.detector_x**2 + params.detector_y**2 + params.detector_z**2
    )
    print("Max possible distance to track", round(max_distance, 2), "mm")
    print("Drift distance", params.detector_z, "mm")

    sipm_distance = sipm_distance[~np.isnan(sipm_light) & (sipm_light > 0)]
    sipm_angle = sipm_angle[~np.isnan(sipm_light) & (sipm_light > 0)]
    sipm_light = sipm_light[~np.isnan(sipm_light) & (sipm_light > 0)]

    fig1 = plt.figure(figsize=(7, 6))
    ax1 = fig1.add_subplot(111)

    vline = max_std(
        sipm_light,
        ax1,
        array_max=light_max,
        max_std_ratio=max_std_ratio,
        min_count_ratio=min_count_ratio,
    )
    bins = vline

    ax1.set_xlabel("Max light integral")
    ax1.set_ylabel("Normalized value")

    fig1.suptitle("Light integral distribution")

    sipm_distance = sipm_distance[(sipm_light <= vline)]
    sipm_angle = sipm_angle[(sipm_light <= vline)]
    sipm_light = sipm_light[(sipm_light <= vline)]
    sipm_angle = np.degrees(sipm_angle)

    fig2, ax2 = plt.subplots(1, 1, figsize=(7, 6))
    axes = np.array(ax2)

    n2, x_edges2, y_edges2, image2 = ax2.hist2d(
        sipm_distance,
        sipm_angle,
        weights=abs(sipm_light),
        bins=bins,
        cmin=1,
        norm=LogNorm() if lognorm else None,
    )

    def triangle_calc(height, base):
        # Calculate the angle θ
        return np.degrees(2 * np.arctan((base / 2) / height))

    def inverse_triangle_calc(angle, height):
        # Calculate the base
        return 2 * height * np.tan(np.radians(angle / 2))

    filtered_centers_x2, filtered_centers_y2, cluster_labels = cluster_hot_bins(
        0.35, n2, x_edges2, y_edges2, scale=(3, 1), eps=8
    )

    # Create a LinearSegmentedColormap from the gradient
    salmon_cmap = LinearSegmentedColormap.from_list(
        "salmon_cmap",
        [
            to_rgba("darkred"),
            to_rgba("salmon"),
        ],
        N=np.unique(cluster_labels).size - 1,
    )

    x2 = np.arange(min(sipm_distance), max(sipm_distance), 1)
    for cluster_label in np.unique(cluster_labels):
        if cluster_label == -1:
            continue
        fit2 = parameters, cov = curve_fit(
            triangle_calc,
            filtered_centers_x2[cluster_labels == cluster_label],
            filtered_centers_y2[cluster_labels == cluster_label],
            p0=[
                inverse_triangle_calc(sipm_angle.mean(), sipm_distance.mean()),
            ],
        )
        print(f"Fit mean track length: {parameters[0]}")

        ax2.plot(
            x2,
            triangle_calc(x2, *parameters),
            ls="-",
            c=salmon_cmap(cluster_label),
            label=rf"Fit: {parameters[0]:.0f}{params.dh_unit} track length",
        )
    ax2.set_ylabel(f"SiPM opening angle to track centre [deg]")
    cbar2 = plt.colorbar(image2)
    cbar2.set_label(rf"Light {params.light_variable} [{params.light_unit} - log]")

    fig2.suptitle(f"SiPM level light distribution - {len(sipm_light)} entries")

    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 6))
    axes = np.append(axes, axes3)

    hist30 = axes3[0].hist2d(
        sipm_distance,
        sipm_light,
        bins=bins,
        cmin=1,
        norm=LogNorm() if lognorm else None,
    )
    axes3[0].set_ylabel(f"Light_{params.light_variable} [{params.light_unit}]")
    cbar30 = plt.colorbar(hist30[3])
    cbar30.set_label(rf"Counts [Log]")

    hist31 = axes3[1].hist2d(
        sipm_angle, sipm_light, bins=bins, cmin=1, norm=LogNorm() if lognorm else None
    )
    axes3[1].set_xlabel(f"SiPM opening angle to track [deg]")
    axes3[1].set_ylabel(f"Light {params.light_variable} [{params.light_unit}]")
    cbar31 = plt.colorbar(hist31[3])
    cbar31.set_label(rf"Counts [Log]")

    fig3.suptitle(f"SiPM level light distribution - {len(sipm_light)} entries")

    for ax in axes:
        set_common_ax_options(ax)
        if ax == ax2 or ax == axes3[0]:
            if limit_xrange:
                xlim = ax.get_xlim()
                ax.set_xlim(xlim[0], min(max_distance + 10, xlim[1]))
            if max(sipm_distance) > params.detector_z:
                ax.axvline(
                    params.detector_z, c="orange", ls="--", label="Drift distance"
                )
            if max(sipm_distance) > max_distance:
                ax.axvline(max_distance, c="r", ls="--", label="Max distance")

            ax.set_xlabel(f"Distance from track centre [{params.dh_unit}]")

            ax.legend()

    for fig in [fig1, fig2, fig3]:
        fig.tight_layout()

    if params.save_figures:
        output_path = os.path.join(params.work_path, params.file_label)
        os.makedirs(output_path, exist_ok=True)
        entries = len(sipm_light)
        fig1.savefig(
            os.path.join(
                output_path, f"light_geo_optimization_{params.file_label}_{entries}.pdf"
            ),
            dpi=300,
            bbox_inches="tight",
        )
        fig2.savefig(
            os.path.join(
                output_path, f"light_geo_2D_hist_{params.file_label}_{entries}.pdf"
            ),
            dpi=300,
            bbox_inches="tight",
        )
        fig3.savefig(
            os.path.join(
                output_path, f"light_geo_1D_hist_{params.file_label}_{entries}.pdf"
            ),
            dpi=300,
            bbox_inches="tight",
        )


def plot_light_fit_stats(metrics):
    cosine_df = pd.DataFrame(columns=["cosine", "threshold", "Light", "Charge"])
    for event, metric in metrics.items():
        if "Fit_line" not in metric["SiPM"]:
            continue
        light_track = metric["SiPM"]["Fit_line"]
        if len(metric.keys()) == (sum(1 for key in metric if isinstance(key, str)) + 1):
            for idx, track in metric.items():
                if isinstance(idx, str):
                    continue
                charge_track = track["Fit_line"]
                cross = charge_track.direction.cross(light_track.direction)
                cosine = abs(
                    charge_track.direction.cosine_similarity(light_track.direction)
                )
                cosine_df.loc[event] = [
                    cosine,
                    metric["SiPM"]["Fit_threshold"],
                    charge_track.direction,
                    light_track.direction,
                ]
    entries = len(cosine_df)
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    for i in range(0, int(cosine_df["threshold"].max()), 10):
        data = cosine_df[cosine_df["threshold"] > i]
        if len(data) > 0.005 * len(cosine_df):
            data.hist(
                "cosine",
                ax=ax,
                bins=np.linspace(0, 1, 11),
                label=f"Threshold: {i} {params.light_unit} - {len(cosine_df[cosine_df['threshold']>i])} entries",
            )
    ax.set(
        title="Cosine similarity between charge and light tracks",
        xlabel="Cosine similarity",
        ylabel="Counts",
    )
    set_common_ax_options(ax)
    ax.legend()
    fig.tight_layout()
    if params.save_figures:
        output_path = os.path.join(params.work_path, params.file_label)
        os.makedirs(output_path, exist_ok=True)
        fig.savefig(
            os.path.join(output_path, f"light_fit_{params.file_label}_{entries}.pdf"),
            dpi=300,
            bbox_inches="tight",
        )


def plot_voxel_data(metrics, bins=50, log=(False, False, False), lognorm=False):
    z = []
    q = []
    l = []
    for i, metric in metrics.items():
        # if not metric["SiPM"]:
        #     continue
        for key, sipm in metric["SiPM"].items():
            if isinstance(key, tuple):
                q.append(sipm["charge_q"])
                z.append(sipm["charge_z"])
                l.append(sipm["integral"])

    z = np.array(z)
    q = np.array(q)
    l = np.array(l)

    max_light = max_std(
        l,
        ax=None,
        min_count_ratio=0.98,
        max_std_ratio=0.1,
    )

    max_charge = np.percentile(q, 99)
    max_z = np.percentile(z, 99)

    mask = (l < max_light) & (l > 0) & (q < max_charge) & (q > 0) & (z < max_z)

    z = z[mask]
    q = q[mask]
    l = l[mask]

    if log[0]:
        bins_z = np.exp(np.linspace(0, np.log(max(z)), bins))
    else:
        bins_z = bins
    if log[1]:
        bins_q = np.exp(np.linspace(np.log(min(q)), np.log(max(q)), bins))
    else:
        bins_q = bins
    if log[2]:
        bins_l = np.exp(np.linspace(np.log(min(l)), np.log(max(l)), bins))
    else:
        bins_l = bins

    fig1 = plt.figure(figsize=(10, 6))
    ax1 = fig1.add_subplot(111)

    hist = ax1.hist2d(
        z, l, bins=[bins_z, bins_l], cmin=1, norm=LogNorm() if lognorm else None
    )
    cbar1 = plt.colorbar(hist[3])
    ax1.set_title("Light vs. z distance")
    cbar1.set_label(rf"Counts - log" if lognorm else rf"Counts")
    set_common_ax_options(cbar=cbar1)

    # def fit_func(x, a, b):
    #     return np.exp(-(x - a) / b)

    # params, cov = curve_fit(fit_func, z, l)

    # print("Exponential fit:", params)

    # x = np.linspace(min(z), max(z), 1000)
    # ax1.plot(x, fit_func(x, *params), c="r", ls="--", label="Exponential fit")
    # ax1.legend()

    fig2, axes2 = plt.subplots(3, 1, figsize=(10, 18))

    figs = [fig1, fig2]
    axes = [ax1, *axes2]

    hist21 = axes2[0].hist2d(
        z, q, bins=[bins_z, bins_q], cmin=1, norm=LogNorm() if lognorm else None
    )
    cbar21 = plt.colorbar(hist21[3])
    axes2[0].set_title(rf"Charge vs. Anode distance")
    cbar21.set_label(rf"Counts - log" if lognorm else rf"Counts")
    set_common_ax_options(cbar=cbar21)

    hist22 = axes2[1].hist2d(
        q, l, bins=[bins_q, bins_l], cmin=1, norm=LogNorm() if lognorm else None
    )
    cbar22 = plt.colorbar(hist22[3])
    axes2[1].set(
        title=rf"Light vs. Charge",
        xlabel=(
            rf"Charge [{params.q_unit} - log]"
            if log[1]
            else rf"Charge [{params.q_unit}]"
        ),
        xscale="log" if log[1] else "linear",
    )
    cbar22.set_label(rf"Counts - log" if lognorm else rf"Counts")
    set_common_ax_options(cbar=cbar22)

    hist23 = axes2[2].hist2d(
        z,
        q,
        weights=l,
        bins=[bins_z, bins_q],
        cmin=1,
        norm=LogNorm() if lognorm else None,
    )
    cbar23 = plt.colorbar(hist23[3])
    axes2[2].set_title(rf"Charge vs. Anode distance with light weights")
    cbar23.set_label(
        (
            rf"Light {params.light_variable} [{params.light_unit} - log]"
            if lognorm
            else rf"Light {params.light_variable} [{params.light_unit}]"
        )
    )
    set_common_ax_options(cbar=cbar23)

    for idx, ax in enumerate(axes):
        if not idx == 2:
            ax.set_xlabel(
                f"Anode distance [{params.z_unit} - log]"
                if log[0]
                else f"Anode distance [{params.z_unit}]"
            )
            ax.set_xscale("log" if log[0] else "linear")
        if idx % 2 == 0:
            ax.set_ylabel(
                rf"Light {params.light_variable} [{params.light_unit} - log]"
                if log[2]
                else rf"Light {params.light_variable} [{params.light_unit}]"
            )
            ax.set_yscale("log" if log[2] else "linear")
        else:
            ax.set_ylabel(
                rf"Charge [{params.q_unit} - log] "
                if log[1]
                else rf"Charge [{params.q_unit}]"
            )
            ax.set_yscale("log" if log[1] else "linear")

        set_common_ax_options(ax=ax)

    for fig in figs:
        fig.tight_layout()

    if params.save_figures:
        output_path = os.path.join(params.work_path, params.file_label)
        os.makedirs(output_path, exist_ok=True)
        events = sum(mask)
        fig1.savefig(
            os.path.join(
                output_path, f"voxel_light_vs_z_{params.file_label}_{events}.pdf"
            ),
            dpi=300,
            bbox_inches="tight",
        )
        fig2.savefig(
            os.path.join(
                output_path, f"voxel_charge_vs_z_hist_{params.file_label}_{events}.pdf"
            ),
            dpi=300,
            bbox_inches="tight",
        )


# ### Light


def light_vs_charge(
    metrics,
    light_max=None,
    min_count_ratio=0.99,
    max_std_ratio=0.5,
    clusters=None,
    bin_density=1,
    log=(True, False),
    p0=True,
):
    if isinstance(log, bool):
        log = [log, log]

    light_array = []
    charge_array = []
    for event, metric in metrics.items():
        light_array.append(metric["Total_light"])
        charge_array.append(metric["Total_charge"])

    light_array = np.array(light_array)
    charge_array = np.array(charge_array)

    mask = (charge_array > 0) & (light_array > 0)
    charge_array = charge_array[mask]
    light_array = light_array[mask]

    fig1 = plt.figure(figsize=(8, 6))
    ax1 = fig1.add_subplot(111)
    ax1.set_xlabel("Max light integral")
    ax1.set_ylabel("Normalized value")
    fig1.suptitle("Light integral distribution")
    vline = max_std(
        light_array,
        ax1,
        array_max=light_max,
        min_count_ratio=min_count_ratio,
        max_std_ratio=max_std_ratio,
    )
    bins = int(vline / 20)

    charge_array = charge_array[(light_array <= vline)]
    light_array = light_array[(light_array <= vline)]

    def hist2d(x, y, ax, bins, log):
        if log:
            log_bins_x = np.exp(np.linspace(np.log(min(x) - 1), np.log(max(x)), bins))
            log_bins_y = np.exp(np.linspace(np.log(min(y)), np.log(max(y)), bins))
            bins = [log_bins_x, log_bins_y]
            ax.set_xscale("log")
            ax.set_yscale("log")

        n, x_edges, y_edges, image = ax.hist2d(x, y, bins=bins, cmin=1)

        # fit peak with curve_fit
        # @latexify.function(use_math_symbols=True)
        def fit_function(xy, amplitude, xo, yo, sigma_x, sigma_y):
            x, y = xy
            gauss = (
                amplitude
                * np.exp(-0.5 * ((x - xo) / sigma_x) ** 2)
                / (sigma_x * np.sqrt(2 * np.pi))
                * np.exp(-0.5 * ((y - yo) / sigma_y) ** 2)
                / (sigma_y * np.sqrt(2 * np.pi))
            )
            return gauss

        try:
            bin_peaks = n.ravel(order="F")
            bin_peaks[np.isnan(bin_peaks)] = 0
            x_bin_centers = 0.5 * (x_edges[1:] + x_edges[:-1])
            y_bin_centers = 0.5 * (y_edges[1:] + y_edges[:-1])
            x_bin_centers, y_bin_centers = np.array(
                np.meshgrid(x_bin_centers, y_bin_centers)
            )
            x_bin_centers = x_bin_centers.ravel()
            y_bin_centers = y_bin_centers.ravel()

            parameters, cov_matrix = curve_fit(
                fit_function,
                (
                    x_bin_centers / max(x_bin_centers),
                    y_bin_centers / max(y_bin_centers),
                ),
                bin_peaks,
                bounds=(0, np.inf),
            )
            plot_mesh = np.array(
                np.meshgrid(
                    np.linspace(min(x) / max(x), 1, len(x_bin_centers) * 5),
                    np.linspace(min(y) / max(y), 1, len(y_bin_centers) * 5),
                )
            )

            z_plot = fit_function(plot_mesh, *parameters)

            # print(latexify.get_latex(fit_function))
            print("Parameters:")
            print(
                "\n".join(
                    [
                        f"{name}: {value}"
                        for name, value in zip(
                            [
                                "amplitude",
                                "mu_x",
                                "mu_y",
                                "sigma_x",
                                "sigma_y",
                                "theta",
                            ],
                            parameters,
                        )
                    ]
                ),
                "\n",
            )
            contour = ax.contour(
                plot_mesh[0] * max(x),
                plot_mesh[1] * max(y),
                z_plot,
                norm="log",
                cmap="autumn",
                linewidths=1,
                levels=[
                    fit_function(
                        (
                            parameters[1] - 3 * parameters[3],
                            parameters[2] - 3 * parameters[4],
                        ),
                        *parameters,
                    ),
                    fit_function(
                        (
                            parameters[1] - 2 * parameters[3],
                            parameters[2] - 2 * parameters[4],
                        ),
                        *parameters,
                    ),
                    fit_function(
                        (
                            parameters[1] - 1 * parameters[3],
                            parameters[2] - 1 * parameters[4],
                        ),
                        *parameters,
                    ),
                ],
            )
            fmt = {}
            strs = [r"$3\sigma$", r"$2\sigma$", r"$1\sigma$"]
            for l, s in zip(contour.levels, strs):
                fmt[l] = s

            ax.clabel(contour, contour.levels, inline=True, fmt=fmt, fontsize=10)
        except:
            print("Fit failed\n")

        ax.set_xlabel(f"Total charge [{params.q_unit}{' - Log' if log else ''}]")
        ax.set_ylabel(
            f"Total Light {params.light_variable} [{params.light_unit}{' - Log' if log else ''}]"
        )
        cbar = plt.colorbar(image)
        cbar.set_label(rf"Counts")
        set_common_ax_options(ax)

        return n, x_edges, y_edges, image

    def hist1d(array, ax, bin_density, log, p0):
        # fit peak with curve_fit
        def fit_function(x, a, mu, sigma, b, c):
            return (
                # gaussian_part
                a
                * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
                / (sigma * np.sqrt(2 * np.pi))
                # exponential_part
                + b * np.exp(-c * x)
            )

        upper_bound = np.percentile(array, 95)
        array = array[array < upper_bound]
        if log:
            ax.set_xscale("log")
            bins = np.exp(
                np.linspace(
                    np.log(min(array) - 1), np.log(upper_bound), int(50 * bin_density)
                )
            )
        else:
            bins = np.linspace(min(array), upper_bound, int(50 * bin_density))

        n, edges, patches = ax.hist(
            array, bins=bins, fill=False, ec="C0", histtype="bar"
        )

        peak_y = n.max()
        peak_x = edges[n.argmax() : n.argmax() + 1].mean()
        mean = array.mean()
        median = np.median(array)

        set_common_ax_options(ax)
        if not log:
            bin_centers = 0.5 * (edges[1:] + edges[:-1])
            bin_centers = bin_centers[n > 0]
            bin_peaks = n[n > 0]

            try:
                if p0 is True:
                    p0 = [
                        peak_y,
                        max(peak_x / min(bin_centers), 1),
                        0.04 * max(bin_centers) / max(peak_x, 1),
                        peak_y,
                        0.01,
                    ]
                parameters, cov_matrix = curve_fit(
                    fit_function,
                    bin_centers / min(bin_centers),
                    bin_peaks,
                    p0=p0,
                    bounds=([0, 0, 0, 0, -np.inf], np.inf),
                )
                x_plot = np.linspace(min(array), max(array), len(bins) * 10)
                y_plot = fit_function(x_plot / min(bin_centers), *parameters)

                print("Parameters:")
                print(
                    "\n".join(
                        [
                            f"{name}: {value}"
                            for name, value in zip(
                                ["a", "mu", "sigma", "b", "c"], parameters
                            )
                        ]
                    ),
                    "\n",
                )
                ax.plot(
                    x_plot,
                    y_plot,
                    "m",
                    label=rf"Fit ($\mu={parameters[1]*min(bin_centers):.2f}$)",
                )
            except:
                print("Fit failed\n")

        ax.axvline(peak_x, c="r", ls="--", label=f"Peak: {peak_x:.2f}")
        ax.axvline(median, c="orange", ls="--", label=f"Median: {median:.2f}")
        ax.axvline(mean, c="g", ls="--", label=f"Mean: {mean:.2f}")

        ax.set_ylabel(f"Counts")
        # ax.set_xlim(min(array) - 2, edges3[(edges3 < upper_bound).argmin()])
        ax.set_ylim(0, peak_y * 1.1)
        ax.legend()

    fig2 = plt.figure(figsize=(8, 6))
    ax2 = plt.subplot(111)
    n2d, xedges2d, yedges2d, image2d = hist2d(
        charge_array, light_array, ax2, bins, log[0]
    )
    fig2.suptitle(f"Event level Light vs. Charge - {len(charge_array)} events")

    fig3 = plt.figure(figsize=(8, 6))
    ax3 = plt.subplot(111)
    ratio = charge_array / light_array
    hist1d(ratio, ax3, bin_density, log[1], p0)
    ax3.set_xlabel(
        f"Event total charge / Light [{params.q_unit}/{params.light_unit}{' - Log' if log[1] else ''}]"
    )
    fig3.suptitle(f"Event level Charge vs. Light - {len(charge_array)} events")

    fig4, axes4 = plt.subplots(1, 2, figsize=(14, 6))

    hist1d(charge_array, axes4[0], bin_density, log[1], p0)
    axes4[0].set_xlabel(
        f"Event total charge [{params.q_unit}{' - Log' if log[1] else ''}]"
    )
    hist1d(light_array, axes4[1], bin_density, log[1], p0)
    axes4[1].set_xlabel(
        f"Event total Light [{params.light_unit}{' - Log' if log[1] else ''}]"
    )
    fig4.suptitle(f"Event level Charge and Light - {len(charge_array)} events")

    figs = [fig1, fig2, fig3, fig4]
    if clusters is None:
        if log[0]:
            temp_x, temp_y, cluster_labels = cluster_hot_bins(
                0.3, n2d, np.log(xedges2d), np.log(yedges2d), eps=1
            )
        else:
            temp_x, temp_y, cluster_labels = cluster_hot_bins(
                0.3,
                n2d,
                xedges2d,
                yedges2d,
                eps=2,
                scale=(np.diff(xedges2d).mean(), np.diff(yedges2d).mean()),
            )
        clusters = np.unique(cluster_labels).size - 1

    if clusters > 1:
        data = pd.DataFrame(np.log(charge_array), np.log(light_array))
        kmeans = KMeans(n_clusters=clusters)
        kmeans.fit(data)
        labels = kmeans.predict(data) + 1
        populations = np.unique(labels)

        fig22, axes22 = plt.subplots(
            len(populations), 1, figsize=(8, 6 * len(populations))
        )

        fig32, axes32 = plt.subplots(
            len(populations), 1, figsize=(10, 6 * len(populations))
        )

        figs.extend([fig22, fig32])

        for idx, label in enumerate(populations):
            hist2d(
                charge_array[labels == label],
                light_array[labels == label],
                axes22[idx],
                bins,
                log[0],
            )
            axes22[idx].set_title(f"Population {label} - {sum(labels == label)} events")

            print(f"population {label}:")

            ratio = charge_array[labels == label] / light_array[labels == label]
            hist1d(ratio, axes32[idx], bin_density, log[1], p0)
            axes32[idx].set_xlabel(
                f"Event total charge / Light [{params.q_unit}/{params.light_unit}{' - Log' if log[1] else ''}]"
            )
            axes32[idx].set_title(f"Population {label} - {sum(labels == label)} events")

    for fig in figs:
        fig.tight_layout()

    if params.save_figures:
        output_path = os.path.join(params.work_path, params.file_label)
        os.makedirs(output_path, exist_ok=True)
        events = len(ratio)
        fig1.savefig(
            os.path.join(
                output_path,
                f"light_vs_charge_optmization_{params.file_label}_{events}.pdf",
            ),
            dpi=300,
            bbox_inches="tight",
        )
        fig2.savefig(
            os.path.join(
                output_path, f"light_vs_charge_2D_hist_{params.file_label}_{events}.pdf"
            ),
            dpi=300,
            bbox_inches="tight",
        )
        fig3.savefig(
            os.path.join(
                output_path, f"light_vs_charge_ratio_{params.file_label}_{events}.pdf"
            ),
            dpi=300,
            bbox_inches="tight",
        )
