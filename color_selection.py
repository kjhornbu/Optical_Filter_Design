## Picking useful colors out of a rotation sweep
# A two-film sweep gives thousands of (rotation 1, rotation 2) settings. These functions pick the ones
# people actually want: the best set of primaries, an evenly spaced hue wheel, or the setting closest to a
# requested color. They work on arrays of any shape (for example the A1 x A2 grid from a sweep) and return
# indices into those arrays, so the matching rotations are rotations_1[i], rotations_2[j].
import itertools

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull, QhullError

from Multi_Twist_Main import *
from color_manipulations import *

def hue_and_chroma_uv(u_star, v_star):
    #CIELUV hue angle h_uv in degrees (0-360) and chroma C*_uv. Hue is measured around the white point,
    #so equal steps in h_uv are roughly equal perceived hue steps.
    u_star, v_star = np.asarray(u_star, dtype=float), np.asarray(v_star, dtype=float)
    return np.degrees(np.arctan2(v_star, u_star)) % 360, np.hypot(u_star, v_star)

def _candidates(u, v, L, min_lightness):
    #Flat indices of usable points: finite color and at least min_lightness (L*) if given
    u, v = np.ravel(u), np.ravel(v)
    usable = np.isfinite(u) & np.isfinite(v)
    if min_lightness is not None:
        if L is None:
            raise ValueError("min_lightness needs L (CIELUV or CIELAB L*) for every point")
        usable &= np.ravel(L) >= min_lightness
    return np.flatnonzero(usable)

def pick_primaries(u, v, L=None, min_lightness=None):
    #The three points whose u'v' triangle has the largest area: the best red/green/blue-style primaries the
    #element can reach. Settings dimmer than min_lightness (L*) are excluded so a near-black point can't win.
    #Returns (flat indices of the 3 points in counterclockwise order, triangle area in u'v').
    candidates = _candidates(u, v, L, min_lightness)
    if candidates.size < 3:
        raise ValueError(f"Need at least 3 usable points to pick primaries, got {candidates.size}")
    points = np.column_stack([np.ravel(u)[candidates], np.ravel(v)[candidates]])

    #The largest triangle always has its corners on the convex hull, so only hull vertices need checking
    try:
        hull = ConvexHull(points).vertices
    except QhullError:
        raise ValueError("The usable colors all lie on a line, so no primaries triangle exists")
    triples = np.array(list(itertools.combinations(hull, 3)))
    p0, p1, p2 = points[triples[:, 0]], points[triples[:, 1]], points[triples[:, 2]]
    areas = 0.5*np.abs((p1[:, 0]-p0[:, 0])*(p2[:, 1]-p0[:, 1]) - (p1[:, 1]-p0[:, 1])*(p2[:, 0]-p0[:, 0]))
    best = triples[np.argmax(areas)]

    #ConvexHull lists vertices counterclockwise, and best keeps that order
    return candidates[best], float(areas.max())

def sample_hue_wheel(u_star, v_star, L=None, number_of_hues=12, min_lightness=None, hue_tolerance=3.0):
    #Evenly spaced hues around the CIELUV hue circle, starting at 0 degrees. For each target hue, picks the
    #most saturated (highest chroma) point within hue_tolerance degrees of it, breaking ties by closeness in hue.
    #Keep the tolerance small so the samples really are evenly spaced; widen it for coarse sweeps.
    #Returns a list of (target hue, flat index or None if no reachable color has that hue).
    candidates = _candidates(u_star, v_star, L, min_lightness)
    hue, chroma = hue_and_chroma_uv(np.ravel(u_star)[candidates], np.ravel(v_star)[candidates])

    wheel = []
    for target in np.arange(number_of_hues)*360/number_of_hues:
        difference = np.abs((hue - target + 180) % 360 - 180)   #circular hue difference
        nearby = np.flatnonzero(difference <= hue_tolerance)
        if nearby.size == 0:
            wheel.append((float(target), None))
            continue
        best = nearby[np.lexsort((difference[nearby], -np.round(chroma[nearby], 9)))[0]]   #highest chroma, then nearest hue
        wheel.append((float(target), int(candidates[best])))
    return wheel

def _uv_of_setting(film_1, film_2, wavelengths, input_stokes, rotation_1, rotation_2, configuration, flip_second):
    stokes = two_film_muller_matrix(film_1, film_2, wavelengths, rotation_1, rotation_2, flip_second) @ input_stokes
    return np.array(convertXYZTouv(*convertMTRToXYZ(wavelengths, stokes, configuration=configuration)))

def find_rotations_for_color(target_uv, film_1, film_2, wavelengths, input_stokes, configuration="crossed",
                             flip_second=True, grid_step=2*np.pi/180):
    #Rotation pair whose color is closest to target_uv = (u', v'). Searches a grid, then refines the best grid
    #point continuously. Returns (rotation_1 in [0, pi/2), rotation_2 in [0, pi), achieved (u', v'), distance in u'v').
    #For scale: a u'v' distance of about 0.004 is commonly treated as just noticeable.
    target_uv = np.asarray(target_uv, dtype=float)
    rotations_1, rotations_2 = two_film_rotation_grid(grid_step)   #every distinct setting once
    stokes = two_film_rotation_sweep(film_1, film_2, wavelengths, rotations_1, rotations_2, input_stokes, flip_second)
    u, v = convertXYZTouv(*convertMTRToXYZ(wavelengths, stokes, configuration=configuration))
    distance = np.hypot(u - target_uv[0], v - target_uv[1])
    i, j = np.unravel_index(np.nanargmin(distance), distance.shape)

    cost = lambda r: np.hypot(*(_uv_of_setting(film_1, film_2, wavelengths, input_stokes, r[0], r[1], configuration, flip_second) - target_uv))
    start = [rotations_1[i], rotations_2[j]]
    result = minimize(cost, x0=start, method="Nelder-Mead",
                      options={"xatol": 1e-4, "fatol": 1e-7, "initial_simplex": [start,
                               [start[0] + grid_step/2, start[1]], [start[0], start[1] + grid_step/2]]})
    rotation_1, rotation_2 = canonical_two_film_rotations(*(result.x if result.fun <= distance[i, j] else start))
    achieved = _uv_of_setting(film_1, film_2, wavelengths, input_stokes, rotation_1, rotation_2, configuration, flip_second)
    return float(rotation_1), float(rotation_2), achieved, float(np.hypot(*(achieved - target_uv)))
