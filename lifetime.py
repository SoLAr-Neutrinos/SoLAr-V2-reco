import pickle
import sys
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from solarv2 import params
import pandas as pd
import pylandau


def find_z_values(line, norm, dx):
    return [line.to_point(t=-norm / 2 + t)[2] for t in dx.index]


def exp_decay(x, tau, init):
    return init * np.exp(-x / tau)


def get_lifetime(metrics):
    totaldQdx_slices = [[] for _ in range(6)]
    dz, mediandQdx, std, fits = [], [], [], []

    for entry in tqdm(metrics.values()):
        for track, values in entry.items():
            if isinstance(track, str) or track <= 0:
                continue

            fitted_track = values["Fit_line"]
            fitted_len = values["Fit_norm"]
            dq, dx = values["dQ"], values["dx"]
            dx = dx[(dx > 0) & (dq > 0)]
            dq = dq[dx.index]
            dqdx = list(dq / dx)

            dq_z = find_z_values(fitted_track, fitted_len, dx)
            for i in range(len(dq_z)):
                if dqdx[i] == 0 or np.isnan(dqdx[i]):
                    continue
                bin_idx = min(int(dq_z[i] // 50), 5)
                if bin_idx < 0:
                    continue
                totaldQdx_slices[bin_idx].append(dqdx[i])

    fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharex=True, sharey=True, tight_layout=True)
    axes = axes.flatten()
    for j, data in enumerate(totaldQdx_slices):
        if len(data) >= 10:
            z = (2.5 + j * 5)*10
            dz.append(z)
            limit = np.percentile(data, 99)
            n, edges, patches = axes[j].hist(
                data,
                bins=np.linspace(0, 20000, 100),
            )
            axes[j].set_title(f"Z = {z} mm")
            if j // 3 == 1:
                axes[j].set_xlabel(r"d$Q$/d$x$ [e mm$^{-1}$]", fontsize=12)
            if j % 3 == 0:
                axes[j].set_ylabel("Counts", fontsize=12)

            bin_centers = 0.5 * (edges[1:] + edges[:-1])
            try:
                bounds = (
                    (
                        bin_centers[n.argmax() - 5],
                        0,
                        1000,
                        n.max() * 0.9,
                    ),
                    (
                        bin_centers[n.argmax() + 5],
                        150,
                        2000,
                        n.max() * 1.1,
                    ),
                )
                popt, pcov = curve_fit(
                    pylandau.langau,
                    bin_centers[(bin_centers > 2000) & (bin_centers < limit)],
                    n[(bin_centers > 2000) & (bin_centers < limit)],
                    absolute_sigma=True,
                    p0=[bin_centers[n.argmax()], 125 , 1400, n.max()],
                    bounds=bounds,
                )
                axes[j].plot(
                    fit_x := np.linspace(bin_centers[0], bin_centers[-1], 100),
                    pylandau.langau(fit_x, *popt),
                    "r-",
                    label=f"$\mu={popt[0]:5.1f}$\n$\eta={popt[1]:5.1f}$\n$\sigma={popt[2]:5.1f}$\n$A={popt[3]:5.1f}$",
                )
                axes[j].legend(fontsize=10, title="Fit")
                mediandQdx.append(popt[0])
                perr = np.sqrt(np.diag(pcov))
                std.append(perr[0])
                print(f"fit {z} (μ={popt[0]:5.1f}, η={popt[1]:5.1f}, σ={popt[2]:5.1f}, A={popt[3]:5.1f})")
                fits.append((popt, perr))
                # print(perr[0])
            except Exception as e:
                print("Error occurred while fitting:", e)
                mediandQdx.append(np.median(data))
                std.append(np.std(data))

    fig.savefig(f"dqdx_binnedInZ_fit.pdf", bbox_inches="tight", dpi=300)
    plt.close()

    dt = np.array([(i / (params.drift_velocity * 1000)) for i in dz])
    popt, pcov = curve_fit(exp_decay, dt, mediandQdx, p0=[2.0, 5000])

    plt.close()
    # plt.plot(dz, mediandQdx, "o")
    plt.errorbar(dz, mediandQdx, yerr=std, marker="o", ls="None", label="Data")
    dz = np.linspace(0, 300, 100)
    dt = np.array([(i / (params.drift_velocity * 1000)) for i in dz])
    plt.plot(dz, exp_decay(dt, *popt), label="Exponential decay")
    plt.legend(fontsize=12)
    plt.title(r"d$Q$/d$x$ MPV Across Drift Distance", fontsize=14)
    plt.ylabel(r"d$Q$/d$x$ MPV [e mm$^{-1}$]", fontsize=14)
    plt.xticks(np.arange(0, 350, 50))
    plt.xlabel("Drift Distance [mm]", fontsize=14)
    plt.tick_params(axis="both", which="major", direction="in", size=8, labelsize=12, right=True, top=True)
    plt.tick_params(axis="both", which="minor", direction="in", size=4, right=True, top=True)
    plt.grid(ls=":")
    plt.minorticks_on()
    plt.ylim(plt.ylim()[0] * 0.99, plt.ylim()[1] * 1.01)
    plt.xlim(0, 300)
    plt.savefig("dqdx_binnedInZ.pdf", bbox_inches="tight")
    plt.close()

    return popt, pcov, fits


if __name__ == "__main__":
    metrics_file = sys.argv[1]
    with open(metrics_file, "rb") as input_file:
        metrics = pickle.load(input_file)
    popt, pcov, fits = get_lifetime(metrics)
    fit_tau, fit_init = popt
    uncertainties = np.sqrt(np.diag(pcov))
    print("Measured Lifetime:", fit_tau, "ms", fit_init)
    print("Uncertainties:", uncertainties)

    with open("lifetime_result.txt", "w") as f:
        for i, fit in enumerate(fits):
            f.write(f"Fit {i} parameters: {fit[0]}\n")
            f.write(f"Fit {i} errors: {fit[1]}\n")
        f.write(f"Measured Lifetime: {fit_tau} ms ± {uncertainties[0]} ms\n")
        f.write(f"Initial dQ/dx: {fit_init} e/mm ± {uncertainties[1]} e/mm\n")
    # Save the lifetime result back into the metrics dictionary

    # metrics["lifetime"] = fit_tau
    # with open(metrics_file, "wb") as output_file:
    #     pickle.dump(metrics, output_file)
