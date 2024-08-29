import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pylandau
from IPython.display import display
from matplotlib.colors import LinearSegmentedColormap, LogNorm, to_rgba
from matplotlib.ticker import AutoMinorLocator, MaxNLocator, ScalarFormatter
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans

if __package__:
    from . import params
    from .methods import (
        cluster_hits,
        cluster_hot_bins,
        create_square,
        fit_hit_clusters,
        generate_dead_area,
        get_track_stats,
        max_std,
    )
else:
    import params
    from methods import (
        cluster_hits,
        cluster_hot_bins,
        create_square,
        fit_hit_clusters,
        generate_dead_area,
        get_track_stats,
        max_std,
    )

plt.style.use(params.style)


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
    **kwargs,
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

    ax3d.set_zlim([0, ax3d.get_zlim()[1]])
    # ax3d.view_init(160, 110, -85)
    ax3d.view_init(30, 20, 100)
    # ax3d.view_init(0, 0, 0)
    # ax3d.view_init(0, 0, 90)
    fig.tight_layout()

    if params.save_figures:
        output_path = os.path.join(
            params.work_path, params.output_folder, str(event_idx)
        )
        os.makedirs(output_path, exist_ok=True)
        fig.savefig(
            os.path.join(output_path, f"event_{event_idx}.pdf"),
            dpi=300,
            bbox_inches="tight",
        )

    return metrics


def plot_fake_data(z_range, buffer=1, **kwargs):
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

    output_path = os.path.join(params.work_path, params.output_folder)
    os.makedirs(output_path, exist_ok=True)
    fig.savefig(
        os.path.join(output_path, "fake_data_map.pdf"), dpi=300, bbox_inches="tight"
    )


# ### Tracks


# Plot dQ versus X
def plot_dQ(dQ_series, dx_series, event_idx, track_idx, interpolate=False, **kwargs):
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

    ax.step(dQ_series.index, (dQ_series / dx_series).fillna(0), where="mid")
    # ax.scatter(dQ_series.index, (dQ_series / dx_series).fillna(0))
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
        output_path = os.path.join(
            params.work_path, params.output_folder, str(event_idx)
        )
        os.makedirs(output_path, exist_ok=True)
        fig.savefig(
            os.path.join(
                output_path, f"dQ_E{event_idx}_T{track_idx}_{round(target_dx,2)}.pdf"
            ),
            dpi=300,
            bbox_inches="tight",
        )


def plot_track_angles(metrics, **kwargs):
    cos_x = []
    cos_y = []
    cos_z = []
    vectors = []
    for idx, metric in metrics.items():
        for track_idx, track in metric.items():
            if type(track) is dict:
                if "RANSAC_score" in track and track["RANSAC_score"] < 0.5:
                    continue
                if "Fit_norm" in track and track["Fit_norm"] < 1:
                    continue
                if "Fit_line" in track:
                    cos_x.append(
                        track["Fit_line"].direction.cosine_similarity([1, 0, 0])
                    )
                    cos_y.append(
                        track["Fit_line"].direction.cosine_similarity([0, 1, 0])
                    )
                    cos_z.append(
                        track["Fit_line"].direction.cosine_similarity([0, 0, 1])
                    )
                    vectors.append(track["Fit_line"].direction.to_array())

    vectors = np.array(vectors)
    cos_x = np.array(cos_x)
    cos_y = np.array(cos_y)
    cos_z = np.array(cos_z)

    fig, ax = plt.subplots(2, 3, figsize=(18, 12))

    ax[0, 0].hist(vectors[:, 0], bins=20)
    ax[0, 0].set_xlabel("X vector component")
    ax[0, 1].hist(vectors[:, 1], bins=20)
    ax[0, 1].set_xlabel("Y vector component")
    ax[0, 2].hist(vectors[:, 2], bins=20)
    ax[0, 2].set_xlabel("Z vector component")

    ax[1, 0].hist(abs(cos_x), bins=20)
    ax[1, 0].set_xlabel("Cosine similarity to x-axis")
    ax[1, 1].hist(abs(cos_y), bins=20)
    ax[1, 1].set_xlabel("Cosine similarity to y-axis")
    ax[1, 2].hist(abs(cos_z), bins=20)
    ax[1, 2].set_xlabel("Cosine similarity to z-axis")

    for axes in ax.flatten():
        set_common_ax_options(axes)

    fig.tight_layout()
    if params.save_figures:
        output_path = os.path.join(params.work_path, params.output_folder)
        os.makedirs(output_path, exist_ok=True)
        label = (
            f"_{params.filter_label}"
            if params.filter_label is not None
            else f"_{len(cos_x)}"
        )
        fig.savefig(
            os.path.join(
                output_path, f"track_angles_{params.output_folder}{label}.pdf"
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
    dropna=True,
    **kwargs,
):
    df = get_track_stats(
        metrics, empty_ratio_lims=empty_ratio_lims, min_entries=min_entries
    )
    if dropna:
        df = df.dropna(subset=["track_dQdx"])

    track_dQdx = df["track_dQdx"]
    track_length = df["track_length"].astype(float)
    track_score = df["track_score"].astype(float)
    track_z = df["track_z"].astype(float)
    track_cv_dQdx = (
        track_dQdx.dropna().apply(lambda x: x.std() / x.mean()).astype(float)
    )
    track_mean_dQdx = track_dQdx.dropna().apply(lambda x: x.mean()).astype(float)

    score_mask = (track_score >= min_score).to_numpy()
    nan_mask = track_dQdx.notna().to_numpy()
    score_bool = (1 - score_mask).sum() > 0

    print(f"Tracks with score < {min_score}: {len(track_dQdx)-sum(score_mask)}")
    # print(f"\nRemaining tracks: {sum(score_mask)}\n")

    dQdx_series = pd.concat(track_dQdx.dropna().to_list())
    dQdx_series = dQdx_series[dQdx_series > 0].dropna().sort_index()
    cut_dQdx_series = pd.concat(track_dQdx[score_mask].dropna().to_list())
    cut_dQdx_series = cut_dQdx_series[cut_dQdx_series > 0].dropna().sort_index()

    print("\ndQ/dx stats:")
    try:
        if get_ipython() is not None:
            display(dQdx_series.describe())
        else:
            print(dQdx_series.describe())
    except:
        print(dQdx_series.describe())

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
        np.std(bin_centers_all11) / 15,
        max(n_all11),
    )
    bounds = (
        (
            bin_centers_all11[max(n_all11.argmax() - 3, 0)],
            0,
            0,
            0,
        ),
        (
            bin_centers_all11[min(n_all11.argmax() + 3, len(bin_centers_all11) - 1)],
            np.inf,
            np.inf,
            np.inf,
        ),
    )
    try:
        popt, pcov = curve_fit(
            pylandau.langau,
            bin_centers_all11[bin_centers_all11 > 2000],
            n_all11[bin_centers_all11 > 2000],
            absolute_sigma=True,
            p0=p0,
            bounds=bounds,
        )

        ax11.plot(
            fit_x := np.linspace(bins_all11[0], bins_all11[-1], 1000),
            pylandau.langau(fit_x, *popt),
            "r-",
            label=r"fit: $\mu$=%5.1f, $\eta$=%5.1f, $\sigma$=%5.1f, A=%5.1f"
            % tuple(popt),
        )
    except:
        print("\nCould not fit Landau to dQ/dx")
        print(f"p0: {p0}")
        print(f"bounds: {bounds}")

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
        track_length[nan_mask],
        track_mean_dQdx,
        ax21,
        bins,
        lognorm,
        fit="Log",
        profile=profile,
    )

    hist2d22 = hist2d(
        track_length[nan_mask],
        track_cv_dQdx,
        ax22,
        bins,
        lognorm,
        fit="Linear",
        profile=profile,
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

    hist2d(
        track_z[nan_mask],
        track_mean_dQdx,
        ax61,
        bins,
        lognorm,
        fit="Linear",
        profile=profile,
    )

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
            track_length[score_mask * nan_mask],
            track_mean_dQdx[score_mask[nan_mask]],
            ax31,
            bins,
            lognorm,
            fit="Log",
            profile=profile,
        )

        hist2d32 = hist2d(
            track_length[score_mask * nan_mask],
            track_cv_dQdx[score_mask[nan_mask]],
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
            track_z[score_mask * nan_mask],
            track_mean_dQdx[score_mask[nan_mask]],
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
    print("\nMax possible track length", round(max_track_legth, 2), "mm")
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
        label = (
            f"_{params.filter_label}"
            if params.filter_label is not None
            else f"_{len(track_length)}"
        )
        output_path = os.path.join(params.work_path, params.output_folder)
        os.makedirs(output_path, exist_ok=True)

        fig1.savefig(
            os.path.join(
                output_path, f"track_stats_1D_hist_{params.output_folder}{label}.pdf"
            ),
            dpi=300,
            bbox_inches="tight",
        )
        fig2.savefig(
            os.path.join(
                output_path,
                f"track_stats_2D_hist_{params.output_folder}{label}{'_profile' if profile else ''}.pdf",
            ),
            dpi=300,
            bbox_inches="tight",
        )
        fig4.savefig(
            os.path.join(
                output_path,
                f"track_stats_score_{params.output_folder}{label}{'_profile' if profile else ''}.pdf",
            ),
            dpi=300,
            bbox_inches="tight",
        )
        fig5.savefig(
            os.path.join(
                output_path,
                f"track_stats_dQdx_{params.output_folder}{label}{'_profile' if profile else ''}.pdf",
            ),
            dpi=300,
            bbox_inches="tight",
        )
        fig6.savefig(
            os.path.join(
                output_path,
                f"track_stats_dQdx_z_{params.output_folder}{label}{'_profile' if profile else ''}.pdf",
            ),
            dpi=300,
            bbox_inches="tight",
        )
        # fig7.savefig(
        #     os.path.join(output_path, f"track_stats_dQ_z_{params.output_folder}{label}{'_profile' if profile else ''}.pdf"),
        #     dpi=300,
        #     bbox_inches="tight",
        # )
        if score_bool:
            fig3.savefig(
                os.path.join(
                    output_path,
                    f"track_stats_2D_hist_cut_{params.output_folder}{label}{'_profile' if profile else ''}.pdf",
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
    **kwargs,
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
        print(f"Fit {cluster_label} mean track length: {parameters[0]}")

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
        output_path = os.path.join(params.work_path, params.output_folder)
        os.makedirs(output_path, exist_ok=True)
        label = (
            f"_{params.filter_label}"
            if params.filter_label is not None
            else f"_{len(sipm_light)}"
        )
        fig1.savefig(
            os.path.join(
                output_path,
                f"light_geo_optimization_{params.output_folder}{label}.pdf",
            ),
            dpi=300,
            bbox_inches="tight",
        )
        fig2.savefig(
            os.path.join(
                output_path, f"light_geo_2D_hist_{params.output_folder}{label}.pdf"
            ),
            dpi=300,
            bbox_inches="tight",
        )
        fig3.savefig(
            os.path.join(
                output_path, f"light_geo_1D_hist_{params.output_folder}{label}.pdf"
            ),
            dpi=300,
            bbox_inches="tight",
        )


def plot_light_fit_stats(metrics, **kwargs):
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
        label = (
            f"_{params.filter_label}"
            if params.filter_label is not None
            else f"_{len(cosine_df)}"
        )
        output_path = os.path.join(params.work_path, params.output_folder)
        os.makedirs(output_path, exist_ok=True)
        fig.savefig(
            os.path.join(output_path, f"light_fit_{params.output_folder}{label}.pdf"),
            dpi=300,
            bbox_inches="tight",
        )


def plot_voxel_data(
    metrics, bins=50, log=(False, False, False), lognorm=False, **kwargs
):
    if not isinstance(bins, int):
        bins = 50

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
        output_path = os.path.join(params.work_path, params.output_folder)
        os.makedirs(output_path, exist_ok=True)
        label = (
            f"_{params.filter_label}"
            if params.filter_label is not None
            else f"_{sum(mask)}"
        )
        fig1.savefig(
            os.path.join(
                output_path, f"voxel_light_vs_z_{params.output_folder}{label}.pdf"
            ),
            dpi=300,
            bbox_inches="tight",
        )
        fig2.savefig(
            os.path.join(
                output_path,
                f"voxel_charge_vs_z_hist_{params.output_folder}{label}.pdf",
            ),
            dpi=300,
            bbox_inches="tight",
        )


def plot_light_vs_charge(
    metrics,
    light_max=None,
    min_count_ratio=0.99,
    max_std_ratio=0.5,
    clusters=None,
    bin_density=1,
    log=(True, False),
    p0=True,
    **kwargs,
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
        output_path = os.path.join(params.work_path, params.output_folder)
        os.makedirs(output_path, exist_ok=True)
        label = (
            f"_{params.filter_label}"
            if params.filter_label is not None
            else f"_{len(ratio)}"
        )
        fig1.savefig(
            os.path.join(
                output_path,
                f"light_vs_charge_optmization_{params.output_folder}{label}.pdf",
            ),
            dpi=300,
            bbox_inches="tight",
        )
        fig2.savefig(
            os.path.join(
                output_path,
                f"light_vs_charge_2D_hist_{params.output_folder}{label}.pdf",
            ),
            dpi=300,
            bbox_inches="tight",
        )
        fig3.savefig(
            os.path.join(
                output_path,
                f"light_vs_charge_ratio_{params.output_folder}{label}.pdf",
            ),
            dpi=300,
            bbox_inches="tight",
        )
