#Two-film sandwich: pick the useful settings out of a full rotation sweep.
#  - Primaries: the three settings whose colors span the largest triangle in u'v'
#  - Hue wheel: one setting every 30 degrees of CIELUV hue, the most saturated the films can reach at that hue
#Both are found separately for crossed and parallel polarizers. Results are plotted on the u'v' diagram with
#swatches and saved to two_film_color_settings.csv, so each color comes with the rotations that produce it.
#Edit FILM_1 / FILM_2 to try other designs.
import time
start_time = time.time()

import csv
import math
import numpy as np
from matplotlib import pyplot as plt

# Make the repo root importable no matter which folder this script is run from
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from Multi_Twist_Main import *
from color_manipulations import *
from color_selection import *

DEG = math.pi/180

# Film designs: [initial orientation, twist 1, thickness 1, twist 2, thickness 2, ...] in radians and micron
FILM_1 = np.array([47.3*DEG, 76.4*DEG, 1.1, 0*DEG, 2.27, -76.4*DEG, 1.1])   # 3TR HW-B (Komanduri et al. 2013)
FILM_2 = np.array([0*DEG, 0*DEG, 2.0])                                       # single untwisted 2 micron layer

MIN_LIGHTNESS = 40       # L*: ignore settings dimmer than this when picking colors
NUMBER_OF_HUES = 12      # hue wheel samples (every 30 degrees)

wavelengths = np.arange(380, 800, 5)/1e3             # micron
input_stokes = np.array([1, 1, 0, 0], dtype=float)   # linear horizontal polarizer in front
# Every distinct color exactly once: film 1 over 0-90 degrees, film 2 over 0-180 (turning both by 90 repeats colors)
rotations_1, rotations_2 = two_film_rotation_grid(1*DEG)

stokes = two_film_rotation_sweep(FILM_1, FILM_2, wavelengths, rotations_1, rotations_2, input_stokes, flip_second=True)

xBar, yBar, zBar = loadColorMatchingFunctions()
locus_u, locus_v = convertXYZTouv(xBar, yBar, zBar)
keep = np.isfinite(locus_u)
locus_u, locus_v = locus_u[keep], locus_v[keep]

rows = []
fig, axes = plt.subplots(2, 2, figsize=(15, 10.5), height_ratios=[3, 1.1], layout="constrained")

for column, configuration in enumerate(["crossed", "parallel"]):
    X, Y, Z = convertMTRToXYZ(wavelengths, stokes, configuration=configuration)
    u, v = convertXYZTouv(X, Y, Z)
    L, u_star, v_star = convertXYZToCIELUV(X, Y, Z)
    hue, chroma = hue_and_chroma_uv(u_star, v_star)
    display = convertXYZToDisplayRGB(X, Y, Z)

    primaries, primaries_area = pick_primaries(u, v, L, min_lightness=MIN_LIGHTNESS)
    wheel = sample_hue_wheel(u_star, v_star, L, number_of_hues=NUMBER_OF_HUES, min_lightness=MIN_LIGHTNESS)

    picks = [("primary", f"P{n+1}", None, index) for n, index in enumerate(primaries)]
    picks += [("hue", f"{target:.0f}°", target, index) for target, index in wheel]
    for kind, label, target, index in picks:
        if index is None:
            rows.append({"configuration": configuration, "kind": kind, "label": label, "target_hue_deg": target})
            continue
        i, j = np.unravel_index(index, u.shape)
        rows.append({"configuration": configuration, "kind": kind, "label": label, "target_hue_deg": target,
                     "rotation_1_deg": round(rotations_1[i]/DEG, 1), "rotation_2_deg": round(rotations_2[j]/DEG, 1),
                     "u_prime": round(float(u[i, j]), 4), "v_prime": round(float(v[i, j]), 4),
                     "L_star": round(float(L[i, j]), 1), "C_star_uv": round(float(chroma[i, j]), 1),
                     "hue_uv_deg": round(float(hue[i, j]), 1), "_ij": (i, j)})

    # --- u'v' diagram: faint cloud of every setting, the primaries triangle, and the hue wheel picks
    ax = axes[0, column]
    ax.plot(np.r_[locus_u, locus_u[0]], np.r_[locus_v, locus_v[0]], color="0.3", lw=1)
    ax.fill(*SRGB_PRIMARIES_UV.T, fill=False, ls="--", color="0.5", label="sRGB")
    ax.scatter(u.ravel(), v.ravel(), s=3, c=display.reshape(-1, 3), alpha=0.15, edgecolors="none")
    corners = np.array([[u.ravel()[k], v.ravel()[k]] for k in primaries])
    ax.fill(*corners.T, fill=False, color="k", lw=1.5, label=f"primaries (area {primaries_area:.4f})")

    ring_center = np.array([np.nanmean(u), np.nanmean(v)])
    for row in rows:
        if row["configuration"] != configuration or "_ij" not in row:
            continue
        i, j = row["_ij"]
        marker = "s" if row["kind"] == "primary" else "o"
        ax.scatter(u[i, j], v[i, j], s=110 if marker == "s" else 70, marker=marker, color=display[i, j], edgecolors="k", linewidths=1, zorder=3)
        if row["kind"] == "primary":
            #Primary labels go outside the ring, away from its center, with a leader line, so they don't sit on the hue labels
            direction = np.array([u[i, j], v[i, j]]) - ring_center
            direction = direction/np.hypot(*direction)
            ax.annotate(row["label"], (u[i, j], v[i, j]), xytext=tuple(40*direction), textcoords="offset points",
                        ha="right" if direction[0] < 0 else "left", va="center", fontsize=10, fontweight="bold",
                        arrowprops=dict(arrowstyle="-", color="k", lw=0.8, shrinkA=2, shrinkB=6))
        else:
            ax.annotate(row["label"], (u[i, j], v[i, j]), xytext=(6, 4), textcoords="offset points", fontsize=9)

    coverage = gamut_coverage_of_sRGB(u, v, space="uv")
    ax.set(title=f"{configuration.capitalize()} polarizers: primaries (squares) and hue wheel (circles)\n"
                 f"all settings cover {100*coverage:.0f}% of sRGB in u'v'; picks need L* ≥ {MIN_LIGHTNESS}",
           xlabel="u'", ylabel="v'", xlim=(0, 0.65), ylim=(0, 0.62), aspect="equal")
    ax.legend(loc="lower right", fontsize=9)

    # --- swatch strip: each pick at its true brightness, with the rotations that make it
    ax = axes[1, column]
    strip = [row for row in rows if row["configuration"] == configuration]
    for n, row in enumerate(strip):
        if "_ij" in row:
            i, j = row["_ij"]
            ax.add_patch(plt.Rectangle((n, 0.45), 0.9, 0.55, color=display[i, j]))
            text = f"{row['label']}\n{row['rotation_1_deg']:.0f}°\n{row['rotation_2_deg']:.0f}°\nL* {row['L_star']:.0f}"
        else:
            ax.add_patch(plt.Rectangle((n, 0.45), 0.9, 0.55, fill=False, hatch="///", color="0.6"))
            text = f"{row['label']}\nnot\nreachable\n"
        ax.text(n + 0.45, 0.4, text, ha="center", va="top", fontsize=8, linespacing=1.3,
                fontweight="bold" if row["kind"] == "primary" else "normal")
    # Row labels once, to the left of the strip, lined up with the film 1 and film 2 rows of every swatch
    ax.text(-0.12, 0.4, "\nθ₁\nθ₂\n", ha="right", va="top", fontsize=8, linespacing=1.3, fontweight="bold")
    ax.set(xlim=(-0.7, len(strip)), ylim=(-0.45, 1.02), title="Swatches: label, θ₁ = film 1 rotation, θ₂ = film 2 rotation, lightness L*")
    ax.axis("off")

    print(f"\n{configuration.upper()} polarizers: primaries triangle area {primaries_area:.4f} in u'v'")
    print(f"  {'pick':6s} {'film 1':>7s} {'film 2':>7s}   {'u_':>6s} {'v_':>6s}  {'L*':>5s} {'C*':>5s} {'hue':>6s}".replace("_", "'"))
    for row in strip:
        if "_ij" in row:
            print(f"  {row['label']:6s} {row['rotation_1_deg']:6.0f}° {row['rotation_2_deg']:6.0f}°   "
                  f"{row['u_prime']:6.3f} {row['v_prime']:6.3f}  {row['L_star']:5.0f} {row['C_star_uv']:5.0f} {row['hue_uv_deg']:5.0f}°")
        else:
            print(f"  {row['label']:6s} not reachable at L* >= {MIN_LIGHTNESS}")

fig.suptitle("Two-film sandwich: useful settings picked from every distinct setting (θ₁ 0–90°, θ₂ 0–180°, 1° steps)")
fig.savefig("two_film_color_settings.png", dpi=150)

columns = ["configuration", "kind", "label", "target_hue_deg", "rotation_1_deg", "rotation_2_deg",
           "u_prime", "v_prime", "L_star", "C_star_uv", "hue_uv_deg"]
with open("two_film_color_settings.csv", "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)

print(f"\nSaved two_film_color_settings.png and two_film_color_settings.csv ({time.time() - start_time:.1f} seconds)")
plt.show()
