# Checks for the equivalent-retarder decomposition and the rotation transmission bounds.
# Run with pytest, or without it: python tests/test_retarder_analysis.py
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from retarder_analysis import *

DEG = math.pi/180
WAVELENGTHS = np.arange(400, 801, 10)/1e3
LH = np.array([1, 1, 0, 0], dtype=float)
ROTATIONS = np.arange(0, 180, 0.5)*DEG
THREE_TR_HWB = np.array([47.3*DEG, 76.4*DEG, 1.1, 0.0, 2.27, -76.4*DEG, 1.1])
RNG = np.random.default_rng(11)

def random_spec(layers=3):
    spec = [RNG.uniform(-math.pi/2, math.pi/2)]
    for _ in range(layers):
        spec += [RNG.uniform(-2*math.pi, 2*math.pi), RNG.uniform(0.1, 4)]
    return np.array(spec)

def simulated_transmission(spec, configuration="crossed"):
    #Rotate with the full Mueller model and read off crossed/parallel transmission: rotations x wavelengths
    S1 = np.stack([(full_matrix_specification(rotate_MTR_specification(spec, r), WAVELENGTHS) @ LH)[:, 1] for r in ROTATIONS])
    return (1 - S1)/2 if configuration == "crossed" else (1 + S1)/2

def test_closed_form_matches_simulation_at_every_rotation():
    for spec in [THREE_TR_HWB, random_spec(3), random_spec(2), random_spec(1)]:
        for configuration in ("crossed", "parallel"):
            assert np.allclose(predicted_rotation_transmission(spec, WAVELENGTHS, ROTATIONS, configuration),
                               simulated_transmission(spec, configuration), atol=1e-9)

def test_bounds_and_peak_match_simulated_rotation():
    for spec in [THREE_TR_HWB, random_spec(3), random_spec(2)]:
        for configuration in ("crossed", "parallel"):
            T = simulated_transmission(spec, configuration)
            lower, upper, peak = rotation_transmission_bounds(spec, WAVELENGTHS, configuration)
            assert np.allclose(T.max(0), upper, atol=1e-4) and np.allclose(T.min(0), lower, atol=1e-4)
            #transmission at the predicted peak rotation equals the upper bound
            at_peak = np.array([predicted_rotation_transmission(spec, [w], [p], configuration)[0, 0] for w, p in zip(WAVELENGTHS, peak)])
            assert np.allclose(at_peak, upper, atol=1e-9)
            assert np.all((0 <= peak) & (peak < math.pi/2))

def test_decomposition_identities():
    #cos(G/2) = cos(GC/2) cos(GL/2) and n3 sin(G/2) = sin(GC/2) cos(GL/2), so LB = sin^2(GC/2) cos^2(GL/2)
    p = equivalent_retarder_parameters(full_matrix_specification(random_spec(3), WAVELENGTHS))
    G, GC, GL = p["total_retardance"], p["circular_retardance"], p["linear_retardance"]
    assert np.allclose(np.cos(G/2), np.cos(GC/2)*np.cos(GL/2))
    assert np.allclose(np.sqrt(p["n3_squared"])*np.sin(G/2), np.sin(GC/2)*np.cos(GL/2))
    lower, _, _ = rotation_transmission_bounds(random_spec(3), WAVELENGTHS)
    assert np.all(lower >= -1e-12)

def test_known_retarders():
    #Pure linear half-wave plate at 0 degrees: Gamma = pi, no circular part, axis along S1
    hwp = np.array([0.0, 0.0, 0.55/(2*birefringence_formula(0.55))])
    p = equivalent_retarder_parameters(full_matrix_specification(hwp, np.array([0.55])))
    assert np.isclose(p["total_retardance"][0], math.pi) and np.isclose(p["n3_squared"][0], 0, atol=1e-12)
    assert np.isclose(p["circular_retardance"][0], 0) and np.isclose(p["linear_retardance"][0], math.pi)
    lower, upper, peak = rotation_transmission_bounds(hwp, np.array([0.55]))
    assert np.allclose([lower[0], upper[0], peak[0]], [0, 1, math.pi/4])   #half-wave plate: dark to full, peaks at 45
    #A pure rotator (Mueller matrix built by hand, turning the Stokes vector by 2a): all circular, circular
    #retardance 2a, no linear part, and the lower bound equals the upper bound
    a = 0.4
    rotator = np.array([[[1, 0, 0, 0], [0, math.cos(2*a), -math.sin(2*a), 0], [0, math.sin(2*a), math.cos(2*a), 0], [0, 0, 0, 1]]])
    p = equivalent_retarder_parameters(rotator)
    assert np.isclose(p["n3_squared"][0], 1) and np.isclose(p["circular_retardance"][0], 2*a) and np.isclose(p["linear_retardance"][0], 0)

def test_no_retardance_is_handled():
    p = equivalent_retarder_parameters(np.eye(4)[None])
    assert p["total_retardance"][0] == 0 and p["n3_squared"][0] == 0 and np.isfinite(p["axis_phi"][0])

def test_symmetric_3TR_has_no_circular_retardance():
    #+76.4 / 0 / -76.4 degree twists with equal outer layers: the circular parts cancel, so the lower bound is 0
    p = equivalent_retarder_parameters(full_matrix_specification(THREE_TR_HWB, WAVELENGTHS))
    assert np.all(p["n3_squared"] < 1e-12)
    assert np.all(np.degrees(p["circular_retardance"]) < 0.01)   #Gamma_C is ill-conditioned near half-wave; 0.01 deg is "zero"
    assert np.allclose(rotation_transmission_bounds(THREE_TR_HWB, WAVELENGTHS)[0], 0, atol=1e-12)

def sinc(x):
    return np.sinc(x/np.pi)   # sin(x)/x

def test_eq74_form_and_its_scale_factor_in_both_decompositions():
    spec = random_spec(2)
    p = equivalent_retarder_parameters(full_matrix_specification(spec, WAVELENGTHS))
    lower, _, _ = rotation_transmission_bounds(spec, WAVELENGTHS)
    #product split: missing factor cos^2(Gamma_L/2)
    assert np.allclose(lower, eq74_lower_bound(spec, WAVELENGTHS, "product")*np.cos(p["linear_retardance"]/2)**2)
    #pseudopolar (dissertation): missing factor [sinc(Gamma/2)/sinc(CB/2)]^2, never above 1
    factor = (sinc(p["total_retardance"]/2)/sinc(p["pseudopolar_CB"]/2))**2
    assert np.allclose(lower, eq74_lower_bound(spec, WAVELENGTHS)*factor)
    assert np.all(factor <= 1 + 1e-12)

def test_pseudopolar_parameters():
    p = equivalent_retarder_parameters(full_matrix_specification(random_spec(3), WAVELENGTHS))
    G, LB, LB45, CB = p["total_retardance"], p["pseudopolar_LB"], p["pseudopolar_LB45"], p["pseudopolar_CB"]
    assert np.allclose(G**2, LB**2 + LB45**2 + CB**2)
    assert np.allclose(p["pseudopolar_linear"], np.hypot(LB, LB45))
    #exact lower bound in pseudopolar terms: (CB/2)^2 sinc^2(Gamma/2)
    spec = random_spec(3)
    p = equivalent_retarder_parameters(full_matrix_specification(spec, WAVELENGTHS))
    assert np.allclose(rotation_transmission_bounds(spec, WAVELENGTHS)[0], (p["pseudopolar_CB"]/2)**2*sinc(p["total_retardance"]/2)**2)

def test_pseudopolar_known_elements_and_small_retardance():
    #half-wave plate at 0: all LB, no LB' or CB
    hwp = np.array([0.0, 0.0, 0.55/(2*birefringence_formula(0.55))])
    p = equivalent_retarder_parameters(full_matrix_specification(hwp, np.array([0.55])))
    assert np.allclose([abs(p["pseudopolar_LB"][0]), p["pseudopolar_LB45"][0], p["pseudopolar_CB"][0]], [math.pi, 0, 0], atol=1e-9)
    #pure rotator: |CB| equals the total retardance, and Eq. 7.4 is exact. Sign: for this hand-built matrix the
    #Lu-Chipman formula gives R_C = (m12 - m21)/(2 sin R) * R = -2a, so CB = R_C = -2a
    a = 0.4
    rotator = np.array([[[1, 0, 0, 0], [0, math.cos(2*a), -math.sin(2*a), 0], [0, math.sin(2*a), math.cos(2*a), 0], [0, 0, 0, 1]]])
    p = equivalent_retarder_parameters(rotator)
    assert np.isclose(p["pseudopolar_CB"][0], -2*a) and np.isclose(p["pseudopolar_linear"][0], 0)
    #thin twisted film (small retardance): Eq. 7.4 with pseudopolar CB is close to exact
    thin = np.array([0.0, 60*DEG, 0.3])
    lower, _, _ = rotation_transmission_bounds(thin, WAVELENGTHS)
    assert np.allclose(eq74_lower_bound(thin, WAVELENGTHS), lower, atol=1e-3)


def test_unwrapped_retardance_of_a_thick_plate_matches_the_analytic_value():
    #A 2 micron untwisted plate passes half-wave inside the band: the principal value folds back at pi, the
    #unwrapped value must follow 2 pi dn d / lambda exactly
    plate = np.array([0.0, 0.0, 2.0])
    analytic = 2*math.pi*birefringence_formula(WAVELENGTHS)*2.0/WAVELENGTHS
    assert analytic.max() > math.pi > analytic.min()          # the test really crosses the fold
    u = unwrapped_pseudopolar(full_matrix_specification(plate, WAVELENGTHS))
    assert np.allclose(u["total_retardance"], analytic, atol=1e-9)
    assert np.allclose(u["CB"], 0, atol=1e-9)

def test_unwrapping_keeps_the_exact_bound_and_smooths_eq74():
    spec = np.array([0.0, 70*DEG, 1.1, 0.0, 1.6])   # example 2TR whose total retardance passes pi near 583 nm
    wl = np.arange(400, 801, 1)/1e3
    muller = full_matrix_specification(spec, wl)
    u = unwrapped_pseudopolar(muller)
    lower, _, _ = rotation_transmission_bounds(spec, wl)
    #the exact bound is the same on every branch
    assert np.allclose(lower, (u["CB"]/2)**2*sinc(u["total_retardance"]/2)**2, atol=1e-9)
    #the principal retardance has a corner where it folds back at pi; the unwrapped retardance does not
    principal = equivalent_retarder_parameters(muller)["total_retardance"]
    corner = lambda x: np.abs(np.diff(x, 2)).max()
    assert principal.max() > math.pi - 0.01 and u["total_retardance"].max() > math.pi + 0.5
    assert corner(principal) > 5*corner(u["total_retardance"])
    assert np.abs(np.diff(u["total_retardance"])).max() < 0.02 and np.abs(np.diff(u["CB"])).max() < 0.1

def test_arteaga_signs():
    #LB = -R_H, LB' = -R_45, CB = R_C (Arteaga and Canillas 2009), with R from the Lu-Chipman formula
    p = equivalent_retarder_parameters(full_matrix_specification(random_spec(3), WAVELENGTHS))
    R = p["retardance_vector"]
    assert np.allclose(p["pseudopolar_LB"], -R[:, 0]) and np.allclose(p["pseudopolar_LB45"], -R[:, 1])
    assert np.allclose(p["pseudopolar_CB"], R[:, 2])
    #the sign really comes from the antisymmetric part: compare with the Lu-Chipman vector away from half-wave
    M = full_matrix_specification(random_spec(3), WAVELENGTHS)
    p = equivalent_retarder_parameters(M)
    r, G = M[:, 1:, 1:], p["total_retardance"]
    away = np.abs(np.sin(G)) > 0.1
    lu_chipman = np.stack([r[:, 1, 2] - r[:, 2, 1], r[:, 2, 0] - r[:, 0, 2], r[:, 0, 1] - r[:, 1, 0]], axis=1)*(G/(2*np.sin(G)))[:, None]
    assert np.allclose(p["retardance_vector"][away], lu_chipman[away], atol=1e-9)
    #in this code's model a quarter-wave plate at 0 degrees has LB = +pi/2, and at 45 degrees LB' = +pi/2
    qwp = 0.55/(4*birefringence_formula(0.55))
    for angle, key in [(0.0, "pseudopolar_LB"), (45*DEG, "pseudopolar_LB45")]:
        p = equivalent_retarder_parameters(full_matrix_specification(np.array([angle, 0.0, qwp]), np.array([0.55])))
        assert np.isclose(p[key][0], math.pi/2)

def test_unwrapped_and_principal_agree_where_no_unwrapping_is_needed():
    spec = random_spec(2)
    wl = np.arange(600, 801, 2)/1e3
    muller = full_matrix_specification(spec, wl)
    p, u = equivalent_retarder_parameters(muller), unwrapped_pseudopolar(muller)
    on_principal = u["total_retardance"] < math.pi - 0.05
    for a, b in [("pseudopolar_LB", "LB"), ("pseudopolar_LB45", "LB45"), ("pseudopolar_CB", "CB")]:
        assert np.allclose(p[a][on_principal], u[b][on_principal], atol=1e-9)


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
