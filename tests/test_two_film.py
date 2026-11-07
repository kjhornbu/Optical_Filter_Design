# Physics and color checks for the multi-twist model and the two-film sandwich.
# Run with pytest (pip install pytest; pytest tests) or without it: python tests/test_two_film.py
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Multi_Twist_Main import *
from color_manipulations import *

DEG = math.pi/180
WAVELENGTHS = np.arange(400, 800, 5)/1e3          # micron
LH = np.array([1, 1, 0, 0], dtype=float)           # linear horizontal
RNG = np.random.default_rng(1)

# 3TR HW-B from Komanduri, Lawler and Escuti, Optics Express 21(1), 2013
THREE_TR_HWB = np.array([47.3*DEG, 76.4*DEG, 1.1, 0.0, 2.27, -76.4*DEG, 1.1])

def random_spec(layers=3):
    spec = [RNG.uniform(-math.pi/2, math.pi/2)]
    for _ in range(layers):
        spec += [RNG.uniform(-2*math.pi, 2*math.pi), RNG.uniform(0.1, 5)]
    return np.array(spec)

def half_wave_plate(azimuth, wavelength=0.55):
    #Single untwisted layer that is exactly half-wave at `wavelength`
    thickness = wavelength/(2*birefringence_formula(wavelength))
    return np.array([azimuth, 0.0, thickness])


# --- single film --------------------------------------------------------------

def test_3TR_HWB_reproduces_published_design():
    #Right circular in, near-perfect left circular out (S3 within 5% of -1) across 400-800 nm
    _, stokes = full_matrix_specification_multi_wL(THREE_TR_HWB, np.arange(0.4, 0.8, 0.001), np.array([1, 0, 0, 1.0]))
    assert stokes[:, 3].max() < -0.95

def test_mueller_matrices_preserve_fully_polarized_light():
    #Lossless retarders keep the degree of polarization at 1
    _, stokes = full_matrix_specification_multi_wL(random_spec(), WAVELENGTHS, LH)
    assert np.allclose(np.linalg.norm(stokes[:, 1:], axis=1), 1)

def test_rotating_a_spec_equals_rotating_its_mueller_matrix():
    spec, angle = random_spec(), 0.7
    expected = rotate_muller_matrix(full_matrix_specification(spec, 0.55), angle)
    assert np.allclose(full_matrix_specification(rotate_MTR_specification(spec, angle), 0.55), expected)
    assert np.allclose(rotate_muller_stack(full_matrix_specification(spec, WAVELENGTHS), [angle])[0],
                       full_matrix_specification(rotate_MTR_specification(spec, angle), WAVELENGTHS))


# --- flipping -----------------------------------------------------------------

def test_flipped_film_obeys_reverse_propagation_reciprocity():
    #For a reciprocal element, propagating backwards gives M_back = Q M^T Q with Q = diag(1, 1, -1, 1).
    #A film flipped about x is exactly the backwards case, so the flipped spec must reproduce it.
    Q = np.diag([1, 1, -1, 1.0])
    for _ in range(5):
        spec = random_spec()
        M = full_matrix_specification(spec, WAVELENGTHS)
        M_flipped = full_matrix_specification(flip_MTR_specification(spec), WAVELENGTHS)
        assert np.allclose(M_flipped, Q @ np.swapaxes(M, -1, -2) @ Q)

def test_flipping_twice_gives_back_the_original():
    spec = random_spec()
    assert np.allclose(flip_MTR_specification(flip_MTR_specification(spec)), spec)

def test_flipped_waveplate_mirrors_its_axis():
    #An untwisted plate at +30 degrees looks like a plate at -30 degrees from the other side
    assert np.allclose(full_matrix_specification(flip_MTR_specification(half_wave_plate(30*DEG)), WAVELENGTHS),
                       full_matrix_specification(half_wave_plate(-30*DEG), WAVELENGTHS))


# --- two-film sandwich --------------------------------------------------------

def test_empty_second_film_leaves_first_film_unchanged():
    spec_1, empty = random_spec(), np.array([0.3, 0.0, 0.0])
    assert np.allclose(two_film_muller_matrix(spec_1, empty, WAVELENGTHS, 0.4, 1.1),
                       full_matrix_specification(rotate_MTR_specification(spec_1, 0.4), WAVELENGTHS))

def test_two_half_wave_plates_rotate_linear_polarization():
    #Classic result: half-wave plates at a1 then a2 rotate linear light by 2*(a2 - a1).
    #Film 2 is flipped in the sandwich, so a plate coated at 0 degrees sits at its lab rotation.
    a1, a2 = 10*DEG, 35*DEG
    M = two_film_muller_matrix(half_wave_plate(0), half_wave_plate(0), np.array([0.55]), a1, a2)
    stokes = M[0] @ LH
    psi = 2*(a2 - a1)
    assert np.allclose(stokes, [1, math.cos(2*psi), math.sin(2*psi), 0], atol=1e-12)

def test_rotation_sweep_matches_one_at_a_time():
    spec_1, spec_2 = random_spec(), random_spec(2)
    angles_1, angles_2 = np.array([0, 0.3, 1.2]), np.array([-0.5, 0.9])
    for flip in (True, False):
        sweep = two_film_rotation_sweep(spec_1, spec_2, WAVELENGTHS, angles_1, angles_2, LH, flip_second=flip)
        for i, a1 in enumerate(angles_1):
            for j, a2 in enumerate(angles_2):
                expected = two_film_muller_matrix(spec_1, spec_2, WAVELENGTHS, a1, a2, flip_second=flip) @ LH
                assert np.allclose(sweep[i, j], expected)

def test_turning_both_films_90_degrees_repeats_every_color():
    #Same as rotating the whole setup by 90 degrees; lossless films have M01 = M10 = 0 so the swapped polarizers
    #transmit the same. Turning only one film by 90 degrees must NOT repeat the colors for twisted films.
    spec_1, spec_2 = random_spec(), random_spec(2)
    angles = np.arange(0, 180, 10)*DEG
    sweep = two_film_rotation_sweep(spec_1, spec_2, WAVELENGTHS, angles, angles, LH)
    shift = 9   # 90 degrees in 10 degree steps
    for configuration in ("crossed", "parallel"):
        u, v = convertXYZTouv(*convertMTRToXYZ(WAVELENGTHS, sweep, configuration=configuration))
        assert np.allclose(u, np.roll(u, (-shift, -shift), (0, 1))) and np.allclose(v, np.roll(v, (-shift, -shift), (0, 1)))
        assert not np.allclose(u, np.roll(u, -shift, 0))

def test_rotation_grid_and_canonical_settings():
    rotations_1, rotations_2 = two_film_rotation_grid(1*DEG)
    assert rotations_1.size == 90 and rotations_2.size == 180
    assert rotations_1.max() < math.pi/2 and rotations_2.max() < math.pi
    spec_1, spec_2 = random_spec(), random_spec(2)
    for r1, r2 in [(123*DEG, 6*DEG), (-20*DEG, 300*DEG), (250*DEG, -45*DEG), (45*DEG, 100*DEG)]:
        c1, c2 = canonical_two_film_rotations(r1, r2)
        assert 0 <= c1 < math.pi/2 and 0 <= c2 < math.pi
        #Same S1 means the same crossed and parallel transmission (so the same color). S2 and S3 can differ: an odd
        #number of quarter turns gives a different output polarization that looks identical through these analyzers.
        S1 = lambda a, b: (two_film_muller_matrix(spec_1, spec_2, WAVELENGTHS, a, b) @ LH)[:, 1]
        assert np.allclose(S1(r1, r2), S1(c1, c2))

def test_flipped_and_unflipped_sandwiches_differ():
    #Guards against flip_second being silently ignored
    spec_1, spec_2 = random_spec(), random_spec()
    assert not np.allclose(two_film_muller_matrix(spec_1, spec_2, WAVELENGTHS, flip_second=True),
                           two_film_muller_matrix(spec_1, spec_2, WAVELENGTHS, flip_second=False))


# --- color ----------------------------------------------------------------------

def test_crossed_plus_parallel_is_all_the_light():
    _, stokes = full_matrix_specification_multi_wL(random_spec(), WAVELENGTHS, LH)
    _, Yc, _ = convertMTRToXYZ(WAVELENGTHS, stokes, configuration="crossed")
    _, Yp, _ = convertMTRToXYZ(WAVELENGTHS, stokes, configuration="parallel")
    assert math.isclose(Yc + Yp, 1, rel_tol=1e-9)

def test_no_film_is_black_crossed_and_white_parallel():
    stokes = np.broadcast_to(LH, (WAVELENGTHS.size, 4))
    assert np.allclose(convertMTRToXYZ(WAVELENGTHS, stokes, configuration="crossed"), 0)
    L, a, b = convertXYZToCIELAB(*convertMTRToXYZ(WAVELENGTHS, stokes, configuration="parallel"))
    assert np.allclose([L, a, b], [100, 0, 0], atol=1e-6)

def test_batched_color_matches_one_spectrum_at_a_time():
    sweep = two_film_rotation_sweep(random_spec(), random_spec(), WAVELENGTHS, [0, 0.5], [0.2, 0.9, 1.4], LH)
    X, Y, Z = convertMTRToXYZ(WAVELENGTHS, sweep)
    assert X.shape == (2, 3)
    for i in range(2):
        for j in range(3):
            transmission = (1 - sweep[i, j, :, 1])/2
            wl_nm = np.arange(380, 780)
            T = np.interp(wl_nm, WAVELENGTHS*1e3, transmission)
            xbar, ybar, zbar = loadColorMatchingFunctions()
            assert np.allclose([X[i, j], Y[i, j], Z[i, j]],
                               [np.sum(cmf*T)/np.sum(ybar) for cmf in (xbar, ybar, zbar)], rtol=1e-2)

def test_gamut_metrics_on_known_shapes():
    x, y = SRGB_PRIMARIES_XY.T
    assert math.isclose(gamut_area_xy(x, y), 0.1121, abs_tol=1e-4)
    assert math.isclose(gamut_coverage_of_sRGB(x, y), 1.0)
    assert gamut_area_xy([0.3, 0.3, 0.3], [0.3, 0.3, 0.3]) == 0            # one color: no area
    assert gamut_coverage_of_sRGB([0.0, 0.1, 0.0], [0.8, 0.8, 0.9]) == 0   # entirely outside sRGB

def test_rotating_the_3TR_gives_color():
    #A 3TR design between crossed polarizers changes color as it rotates (it is only achromatic at its design azimuth)
    angles = np.arange(0, 180, 5)*DEG
    stokes = np.stack([full_matrix_specification_multi_wL(rotate_MTR_specification(THREE_TR_HWB, a), WAVELENGTHS, LH)[1] for a in angles])
    x, y = convertXYZToxy(*convertMTRToXYZ(WAVELENGTHS, stokes))
    assert np.nanmax(np.hypot(x - np.nanmean(x), y - np.nanmean(y))) > 0.1


def test_uv_of_known_colors():
    #Illuminant E white is (4/19, 9/19); sRGB primaries in u'v' are (0.4507, 0.5229), (0.1250, 0.5625), (0.1754, 0.1579)
    assert np.allclose(convertXYZTouv(1, 1, 1), [4/19, 9/19])
    assert np.allclose(SRGB_PRIMARIES_UV, [[0.4507, 0.5229], [0.1250, 0.5625], [0.1754, 0.1579]], atol=1e-4)

def test_uv_from_XYZ_matches_uv_from_xy():
    _, stokes = full_matrix_specification_multi_wL(random_spec(), WAVELENGTHS, LH)
    X, Y, Z = convertMTRToXYZ(WAVELENGTHS, stokes)
    assert np.allclose(convertXYZTouv(X, Y, Z), convertxyTouv(*convertXYZToxy(X, Y, Z)))

def test_CIELUV_white_and_black():
    assert np.allclose(convertXYZToCIELUV(*(np.array(whitePointXYZ())/100)), [100, 0, 0], atol=1e-6)
    assert np.allclose(convertXYZToCIELUV(0, 0, 0), [0, 0, 0])

def test_gamut_metrics_in_uv():
    u, v = SRGB_PRIMARIES_UV.T
    assert math.isclose(gamut_area_uv(u, v), 0.0649, abs_tol=1e-4)
    assert math.isclose(gamut_coverage_of_sRGB(u, v, space="uv"), 1.0)
    #The corner triangle cut off through two side midpoints is 1/4 of the area in xy (and shares two edges with sRGB,
    #which used to make the clipping divide by zero). u'v' weights areas differently, so there it is just between 0 and 1.
    sub = np.array([SRGB_PRIMARIES_XY[0], (SRGB_PRIMARIES_XY[0] + SRGB_PRIMARIES_XY[1])/2, (SRGB_PRIMARIES_XY[0] + SRGB_PRIMARIES_XY[2])/2])
    assert math.isclose(gamut_coverage_of_sRGB(*sub.T), 0.25)
    assert 0 < gamut_coverage_of_sRGB(*convertxyTouv(*sub.T), space="uv") < 1


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
