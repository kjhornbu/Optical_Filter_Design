# Checks for picking primaries, hue-wheel samples and target colors out of a rotation sweep.
# Run with pytest, or without it: python tests/test_color_selection.py
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from color_selection import *

DEG = math.pi/180
WAVELENGTHS = np.arange(400, 800, 10)/1e3
LH = np.array([1, 1, 0, 0], dtype=float)
FILM_1 = np.array([47.3*DEG, 76.4*DEG, 1.1, 0.0, 2.27, -76.4*DEG, 1.1])   # 3TR HW-B
FILM_2 = np.array([0.0, 0.0, 2.0])

def test_primaries_are_the_triangle_corners():
    #sRGB corners plus points inside: the corners must be picked, in counterclockwise order
    rng = np.random.default_rng(0)
    weights = rng.dirichlet([1, 1, 1], size=200)
    inside = weights @ SRGB_PRIMARIES_UV
    u, v = np.r_[inside[:, 0], SRGB_PRIMARIES_UV[:, 0]], np.r_[inside[:, 1], SRGB_PRIMARIES_UV[:, 1]]
    indices, area = pick_primaries(u, v)
    assert sorted(indices) == [200, 201, 202]
    assert math.isclose(area, gamut_area_uv(*SRGB_PRIMARIES_UV.T))
    corners = np.column_stack([u[indices], v[indices]])
    signed = 0.5*np.sum(corners[:, 0]*np.roll(corners[:, 1], -1) - corners[:, 1]*np.roll(corners[:, 0], -1))
    assert signed > 0   # counterclockwise

def test_dim_points_are_not_picked_as_primaries():
    u, v = np.r_[SRGB_PRIMARIES_UV[:, 0], 0.30], np.r_[SRGB_PRIMARIES_UV[:, 1], 0.05]   # far corner point...
    L = np.array([60, 60, 60, 5])                                                       # ...but nearly black
    assert 3 in pick_primaries(u, v)[0]
    assert 3 not in pick_primaries(u, v, L, min_lightness=20)[0]

def test_hue_wheel_picks_most_saturated_point_per_hue():
    #Two rings of colors: chroma 40 and chroma 80. Every hue sample must come from the outer ring at the right hue.
    angles = np.radians(np.arange(0, 360, 5))
    u_star = np.r_[40*np.cos(angles), 80*np.cos(angles)]
    v_star = np.r_[40*np.sin(angles), 80*np.sin(angles)]
    wheel = sample_hue_wheel(u_star, v_star, number_of_hues=12)
    assert [hue for hue, _ in wheel] == list(np.arange(12)*30.0)
    for target, index in wheel:
        hue, chroma = hue_and_chroma_uv(u_star[index], v_star[index])
        assert math.isclose(chroma, 80) and abs((hue - target + 180) % 360 - 180) < 1e-9

def test_hue_wheel_samples_stay_close_to_their_target_hue():
    #A dense ring where chroma grows slightly with hue: the pick must still sit within the tolerance of each target
    angles = np.arange(0, 360, 0.5)
    chroma = 60 + 0.01*angles
    u_star, v_star = chroma*np.cos(np.radians(angles)), chroma*np.sin(np.radians(angles))
    for target, index in sample_hue_wheel(u_star, v_star, number_of_hues=12):
        hue, _ = hue_and_chroma_uv(u_star[index], v_star[index])
        assert abs((hue - target + 180) % 360 - 180) <= 3.0

def test_hue_wheel_reports_unreachable_hues():
    angles = np.radians(np.arange(0, 90, 5))   # only reds/yellows exist
    wheel = dict(sample_hue_wheel(50*np.cos(angles), 50*np.sin(angles), number_of_hues=4))
    assert wheel[0.0] is not None and wheel[180.0] is None and wheel[270.0] is None

def test_find_rotations_recovers_a_reachable_color():
    #Ask for the color of a known off-grid setting: the search must land on that color (any pair that gives it is fine)
    for configuration in ("crossed", "parallel"):
        target = np.array(convertXYZTouv(*convertMTRToXYZ(
            WAVELENGTHS, two_film_muller_matrix(FILM_1, FILM_2, WAVELENGTHS, 33.3*DEG, 71.7*DEG) @ LH, configuration=configuration)))
        r1, r2, achieved, distance = find_rotations_for_color(target, FILM_1, FILM_2, WAVELENGTHS, LH, configuration=configuration)
        assert distance < 1e-3, distance
        assert 0 <= r1 < math.pi/2 and 0 <= r2 < math.pi   # the non-repeating range
        recomputed = convertXYZTouv(*convertMTRToXYZ(
            WAVELENGTHS, two_film_muller_matrix(FILM_1, FILM_2, WAVELENGTHS, r1, r2) @ LH, configuration=configuration))
        assert np.allclose(recomputed, achieved)

def test_find_rotations_reports_distance_for_unreachable_color():
    #Saturated spectral green is far outside what this pair can make: the answer must say how far off it is
    _, _, _, distance = find_rotations_for_color((0.05, 0.58), FILM_1, FILM_2, WAVELENGTHS, LH)
    assert distance > 0.05


if __name__ == "__main__":
    tests = [(name, f) for name, f in sorted(globals().items()) if name.startswith("test_") and callable(f)]
    failed = 0
    for name, test in tests:
        try:
            test()
            print(f"PASS  {name}")
        except Exception as error:
            failed += 1
            print(f"FAIL  {name}: {type(error).__name__}: {error}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
