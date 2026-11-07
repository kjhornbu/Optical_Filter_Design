#Two-film sandwich: sweep both film rotations between fixed polarizers and plot every reachable color
#on the CIE 1976 u'v' chromaticity diagram (more perceptually uniform than xy), for the crossed and
#parallel polarizer configurations. Dots are drawn at their true relative brightness, and the labeled
#markers A-D are the same rotation pairs in both panels, so you can see each one flip to its complement.
#Film 2 is flipped (film-to-film sandwich). Edit FILM_1 / FILM_2 below to try other designs.
import time
start_time = time.time()

import math
import numpy as np
from matplotlib import pyplot as plt

# Make the repo root importable no matter which folder this script is run from
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from Multi_Twist_Main import *
from color_manipulations import *

DEG = math.pi/180

# Film designs: [initial orientation, twist 1, thickness 1, twist 2, thickness 2, ...] in radians and micron
# Film 1: 3TR HW-B from Komanduri, Lawler and Escuti, Optics Express 21(1), 2013
FILM_1 = np.array([47.3*DEG, 76.4*DEG, 1.1, 0*DEG, 2.27, -76.4*DEG, 1.1])
# Film 2: a single untwisted 2 micron layer (strongly chromatic), as a simple second design
FILM_2 = np.array([0*DEG, 0*DEG, 2.0])

wavelengths = np.arange(380, 800, 5)/1e3             # micron
input_stokes = np.array([1, 1, 0, 0], dtype=float)   # linear horizontal polarizer in front
# Every distinct color exactly once: film 1 over 0-90 degrees, film 2 over 0-180 (turning both by 90 repeats colors)
rotations_1, rotations_2 = two_film_rotation_grid(2*DEG)

stokes = two_film_rotation_sweep(FILM_1, FILM_2, wavelengths, rotations_1, rotations_2, input_stokes, flip_second=True)

# Spectral locus for the chromaticity diagram outline
xBar, yBar, zBar = loadColorMatchingFunctions()
locus_u, locus_v = convertXYZTouv(xBar, yBar, zBar)
keep = np.isfinite(locus_u)
locus_u, locus_v = locus_u[keep], locus_v[keep]

# Rotation pairs (film 1, film 2) in degrees to label in both panels
labeled_pairs = {"A": (0, 0), "B": (10, 40), "C": (30, 80), "D": (60, 20)}
labeled_index = {name: (int(round(r1/2)), int(round(r2/2))) for name, (r1, r2) in labeled_pairs.items()}

fig, axes = plt.subplots(1, 2, figsize=(12, 6.8), layout="constrained")
for ax, configuration in zip(axes, ["crossed", "parallel"]):
    X, Y, Z = convertMTRToXYZ(wavelengths, stokes, configuration=configuration)
    u, v = convertXYZTouv(X, Y, Z)
    area, coverage = gamut_area_uv(u, v), gamut_coverage_of_sRGB(u, v, space="uv")
    x, y = convertXYZToxy(X, Y, Z)
    print(f"{configuration:8s}: u'v' area {area:.4f}, covers {100*coverage:.1f}% of sRGB in u'v' "
          f"({100*gamut_coverage_of_sRGB(x, y):.1f}% in xy)")
    
    ax.plot(np.r_[locus_u, locus_u[0]], np.r_[locus_v, locus_v[0]], color="0.3", lw=1)
    ax.fill(*SRGB_PRIMARIES_UV.T, fill=False, ls="--", color="0.5", label="sRGB")
    ax.scatter(u.ravel(), v.ravel(), s=6, c=convertXYZToDisplayRGB(X, Y, Z).reshape(-1, 3), edgecolors="none")
    
    L, _, _ = convertXYZToCIELUV(X, Y, Z)
    for name, (i, j) in labeled_index.items():
        ax.scatter(u[i, j], v[i, j], s=90, facecolors="none", edgecolors="k", linewidths=1.2)
        ax.annotate(f"{name}  L*={L[i, j]:.0f}", (u[i, j], v[i, j]), xytext=(7, 5), textcoords="offset points", fontsize=9)
    
    ax.set(title=f"{configuration.capitalize()} polarizers\nu'v' area {area:.3f}, {100*coverage:.0f}% of sRGB",
           xlabel="u'", ylabel="v'", xlim=(0, 0.65), ylim=(0, 0.62), aspect="equal")
    ax.legend(loc="lower right")

pair_text = ", ".join(f"{name} = ({r1}\u00b0, {r2}\u00b0)" for name, (r1, r2) in labeled_pairs.items())
fig.suptitle(f"Two-film sandwich: every distinct setting (film 1 0\u201390\u00b0, film 2 0\u2013180\u00b0, 2\u00b0 steps)\nLabeled rotation pairs (film 1, film 2): {pair_text}")
fig.savefig("two_film_rotation_sweep.png", dpi=150)

print(f"Time to sweep {rotations_1.size*rotations_2.size} rotation pairs: {time.time() - start_time:.2f} seconds")
plt.show()
