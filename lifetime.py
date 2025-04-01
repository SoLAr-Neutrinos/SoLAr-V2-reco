import pickle
import sys
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from solarv2 import params
import pandas as pd


def find_z_values(line, norm, dx):
    return [line.to_point(t=-norm / 2 + t)[2] for t in dx.index]


def exp_decay(x, tau, init):
    return init * np.exp(-x / tau)


def get_lifetime(metrics):
    totaldQdx_slices = [[] for _ in range(6)]
    dz, mediandQdx, std = [], [], []

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

    for j, data in enumerate(totaldQdx_slices):
        if len(data) >= 10:
            dz.append(2.5 + j * 5)
            mediandQdx.append(np.median(data))
            std.append(np.std(data))

    dt = np.array([(i / (params.drift_velocity * 100)) for i in dz])
    popt, pcov = scipy.optimize.curve_fit(exp_decay, dt, mediandQdx, p0=[2.3, 5000])
    fit_tau, fit_init = popt

    plt.plot(dz, mediandQdx, "o")
    dz = np.linspace(0, 30, 100)
    dt = np.array([(i / (params.drift_velocity * 100)) for i in dz])
    plt.plot(dz, exp_decay(dt, *popt), label="Exponential decay")
    plt.legend(fontsize=14)
    plt.title("Median dQ/dx Measured Across Drift Distance", fontsize=14)
    plt.ylabel(r"Median dQ/dx [e mm$^{-1}$]", fontsize=14)
    plt.xticks(np.arange(0, 35, 5))
    plt.xlabel("Drift distance [cm]", fontsize=14)
    plt.tick_params(axis="both", which="major", direction="in", size=8, labelsize=12, right=True, top=True)
    plt.tick_params(axis="both", which="minor", direction="in", size=4, right=True, top=True)
    plt.grid(ls=":")
    plt.minorticks_on()
    plt.savefig("dqdx_binnedInZ.pdf", bbox_inches="tight")
    plt.close()

    return fit_tau, fit_init, pcov


if __name__ == "__main__":
    metrics_file = sys.argv[1]
    with open(metrics_file, "rb") as input_file:
        metrics = pickle.load(input_file)
    fit_tau, fit_init, cov = get_lifetime(metrics)
    uncertainties = np.sqrt(np.diag(cov))
    print("Measured Lifetime:", fit_tau, "ms", fit_init)
    print("Uncertainties:", uncertainties)
