#Two-film sandwich color map: x is film 1's rotation, y is film 2's rotation, and each pixel is the color you
#would see at that setting (true relative brightness), for crossed and parallel polarizers.
#The map covers 0-180 degrees for both films so the repeats are visible: turning BOTH films by 90 degrees gives
#the same color, so the left half (film 1 0-90, outlined) already contains every distinct color once, and the
#right half repeats it shifted 90 degrees in film 2. The primaries are marked in the outlined half with their
#repeated copies (primed labels) in the other half. Bands running diagonally mean that turning both films
#together barely changes the color; crossing the bands is what sweeps through the hues.
#Edit FILM_1 / FILM_2 to try other designs.
import time
start_time = time.time()

import math
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

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

MIN_LIGHTNESS = 40       # L*: same cutoff as the color picker, so the marked primaries match its CSV

wavelengths = np.arange(380, 800, 5)/1e3             # micron
input_stokes = np.array([1, 1, 0, 0], dtype=float)   # linear horizontal polarizer in front
step = 1                                             # degrees
angles = np.arange(0, 180, step)                     # full 0-180 for both films, to show the repeats
rotations = angles*DEG

stokes = two_film_rotation_sweep(FILM_1, FILM_2, wavelengths, rotations, rotations, input_stokes, flip_second=True)
half = int(90/step)   # film 1 index where the repeat starts

fig, axes = plt.subplots(1, 2, figsize=(13, 7.6), layout="constrained")
for ax, configuration in zip(axes, ["crossed", "parallel"]):
    X, Y, Z = convertMTRToXYZ(wavelengths, stokes, configuration=configuration)
    image = convertXYZToDisplayRGB(X, Y, Z)          # indexed [film 1, film 2, rgb]

    # pixel (film 1, film 2) -> image column x = film 1, row y = film 2
    ax.imshow(np.transpose(image, (1, 0, 2)), origin="lower", extent=(0, 180, 0, 180), interpolation="nearest")

    # Outline the half that holds every distinct color once
    ax.add_patch(Rectangle((0, 0), 90, 180, fill=False, ls="--", lw=1.8, ec="white"))
    ax.add_patch(Rectangle((0, 0), 90, 180, fill=False, ls=(4, (4, 4)), lw=1.8, ec="black"))

    # Primaries, picked in the distinct half exactly like the color picker, plus their repeated copies
    u, v = convertXYZTouv(X[:half], Y[:half], Z[:half])
    L, _, _ = convertXYZToCIELUV(X[:half], Y[:half], Z[:half])
    primaries, _ = pick_primaries(u, v, L, min_lightness=MIN_LIGHTNESS)
    for n, index in enumerate(primaries):
        i, j = np.unravel_index(index, u.shape)
        r1, r2 = angles[i], angles[j]
        copy = (r1 + 90, (r2 + 90) % 180)
        for (x, y), label, is_copy in [((r1, r2), f"P{n+1}", False), (copy, f"P{n+1}\u2032", True)]:
            #hollow markers so the pixel color underneath stays visible; copies get a thinner grey outline
            ax.scatter(x + step/2, y + step/2, s=130, marker="s", facecolors="none",
                       edgecolors="0.35" if is_copy else "k", linewidths=1.2 if is_copy else 2.2, zorder=3)
            ax.annotate(label, (x + step/2, y + step/2), xytext=(9, 7), textcoords="offset points", fontsize=10,
                        fontweight="normal" if is_copy else "bold", color="k",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    
    ax.set(title=f"{configuration.capitalize()} polarizers\nleft of the dashed line: every distinct color once | right: repeat", xlabel="Film 1 rotation (°)", ylabel="Film 2 rotation (°)",
           xticks=range(0, 181, 30), yticks=range(0, 181, 30), xlim=(0, 180), ylim=(0, 180))

fig.suptitle("Two-film sandwich: color at every pair of film rotations (true brightness, 1° steps)\n"
             "P1–P3: primaries;  P′: the same color repeated with both films turned 90° (film 2 wraps around at 180°)")
fig.savefig("two_film_color_map.png", dpi=150)

print(f"Saved two_film_color_map.png ({angles.size**2} settings, {time.time() - start_time:.1f} seconds)")
plt.show()
