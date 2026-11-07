#Bounds on a single element's transmission as it is rotated between crossed linear polarizers.
#For each design:
#  top:    simulated spectra at every rotation (grey), the band they fill, and the closed-form bounds overlaid:
#          upper bound sin^2(Gamma/2) (dissertation Eq. 7.3), exact lower bound (CB/2)^2 sinc^2(Gamma/2), and
#          the Eq. 7.4 form sin^2(CB/2) for comparison, with CB the circular birefringence from the pseudopolar
#          decomposition (Arteaga and Canillas 2009); the two differ by the factor [sinc(Gamma/2)/sinc(CB/2)]^2
#  bottom: the rotation where each wavelength peaks, simulated (dots) vs (pi/4 - phi/2) mod pi/2 (line), with
#          450, 532 and 633 nm marked as in the dissertation's Figure 7.17
#Edit DESIGNS to try other elements.
import time
start_time = time.time()

import math
import numpy as np
from matplotlib import pyplot as plt

# Make the repo root importable no matter which folder this script is run from
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from Multi_Twist_Main import *
from retarder_analysis import *

DEG = math.pi/180

# Designs: [initial orientation, twist 1, thickness 1, twist 2, thickness 2, ...] in radians and micron
DESIGNS = {
    "3TR HW-B (Komanduri et al. 2013)": np.array([47.3*DEG, 76.4*DEG, 1.1, 0*DEG, 2.27, -76.4*DEG, 1.1]),
    "Example 2TR: +70° / 0° twist, 1.1 / 1.6 µm": np.array([0*DEG, 70*DEG, 1.1, 0*DEG, 1.6]),
}
MARKED_WAVELENGTHS = {450: "tab:blue", 532: "tab:green", 633: "tab:red"}   # nm, as in Figure 7.17

wavelengths = np.arange(400, 801, 1)/1e3              # micron
input_stokes = np.array([1, 1, 0, 0], dtype=float)    # linear horizontal polarizer; crossed (vertical) analyzer
rotations = np.arange(0, 90, 0.5)*DEG                 # a single element repeats every 90 degrees between polarizers

fig, axes = plt.subplots(2, len(DESIGNS), figsize=(6.5*len(DESIGNS), 9), height_ratios=[1.6, 1], layout="constrained")
for column, (name, spec) in enumerate(DESIGNS.items()):
    # Full simulation: every rotation, every wavelength
    T = np.stack([(1 - (full_matrix_specification(rotate_MTR_specification(spec, r), wavelengths) @ input_stokes)[:, 1])/2
                  for r in rotations])
    lower, upper, peak = rotation_transmission_bounds(spec, wavelengths)
    eq74 = eq74_lower_bound(spec, wavelengths)
    nm = wavelengths*1e3

    ax = axes[0, column]
    for k in range(0, rotations.size, 10):            # every 5 degrees
        ax.plot(nm, 100*T[k], color="0.75", lw=0.6, zorder=1)
    ax.fill_between(nm, 100*T.min(0), 100*T.max(0), color="0.92", zorder=0, label="simulated, all rotations")
    ax.plot(nm, 100*upper, color="red", lw=2, label="upper bound  sin²(Γ/2)  (Eq. 7.3)")
    ax.plot(nm, 100*lower, color="darkred", lw=2, label="lower bound  (CB/2)²·sinc²(Γ/2)  (exact)")
    ax.plot(nm, 100*eq74, color="darkred", lw=1.4, ls="--", label="Eq. 7.4 form  sin²(CB/2)  (pseudopolar CB, unwrapped)")
    error_ub, error_lb = np.abs(T.max(0) - upper).max(), np.abs(T.min(0) - lower).max()
    ax.set(title=f"{name}\nbounds vs simulation: max error {100*error_ub:.2f}% (upper), {100*error_lb:.2f}% (lower)",
           xlabel="Wavelength (nm)", ylabel="Transmittance (%)", xlim=(400, 800), ylim=(-2, 102))

    ax = axes[1, column]
    simulated_peak = rotations[T.argmax(0)]
    ax.scatter(nm[::8], np.degrees(simulated_peak[::8]), s=10, color="0.4", label="simulated peak (0.5° steps)", zorder=2)
    # break the predicted line where it wraps from 90 back to 0 degrees
    predicted = np.degrees(peak)
    predicted_line = np.where(np.abs(np.diff(predicted, prepend=predicted[0])) > 45, np.nan, predicted)
    ax.plot(nm, predicted_line, color="k", lw=1.2, label="predicted  (45° − φ/2) mod 90°", zorder=1)
    for wavelength, color in MARKED_WAVELENGTHS.items():
        k = np.argmin(np.abs(nm - wavelength))
        ax.scatter(wavelength, predicted[k], s=140, marker="*", color=color, edgecolors="k", linewidths=0.6, zorder=3)
        ax.annotate(f"{wavelength} nm: {predicted[k]:.0f}°", (wavelength, predicted[k]),
                    xytext=(8, -14) if predicted[k] > 75 else (6, 6),
                    textcoords="offset points", fontsize=8.5)
    ax.set(title="Rotation of maximum transmission", xlabel="Wavelength (nm)", ylabel="Orientation (°)",
           xlim=(400, 800), ylim=(0, 90), yticks=range(0, 91, 15))
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9)

    print(f"{name}: upper bound max error {error_ub:.1e}, lower bound max error {error_lb:.1e}, "
          f"Eq. 7.4 form max error {np.abs(T.min(0) - eq74).max():.2f}")

fig.suptitle("Single element rotated between crossed linear polarizers: transmission bounds and peak orientation")
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="outside lower center", ncols=4, fontsize=9, frameon=False)
fig.savefig("rotation_bounds.png", dpi=150)
print(f"Saved rotation_bounds.png ({time.time() - start_time:.1f} seconds)")
plt.show()
