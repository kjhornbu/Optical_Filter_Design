## Equivalent retarder of an MTR and the bounds on its transmission when rotated
# Any lossless, non-depolarizing element (every MTR design in this repo) is equivalent to an elliptical
# retarder: total retardance Gamma about a unit axis n = (n1, n2, n3) on the Poincare sphere, where n3 is the
# circular part. Two ways to split it into linear and circular parts are provided:
#
#  Pseudopolar (exponential) form, as in O. Arteaga and A. Canillas, "Pseudopolar decomposition of the Jones
#  and Mueller-Jones exponential polarization matrices," J. Opt. Soc. Am. A 26, 783-793 (2009): the element is
#  one exponential of effective properties acting together, J ~ exp[-(i/2)(LB s1 + LB' s2 + CB s3)], so
#      LB = Gamma n1,  LB' = Gamma n2,  CB = Gamma n3,  Gamma^2 = LB^2 + LB'^2 + CB^2
#  (lossless case only: LD, LD', CD are zero for these simulated films).
#
#  Product form: a rotator (Gamma_C) times a linear retarder (Gamma_L), in either order, with
#      cos(Gamma/2) = cos(Gamma_C/2) cos(Gamma_L/2),   n3 sin(Gamma/2) = sin(Gamma_C/2) cos(Gamma_L/2)
#
# Rotating the element by theta between ideal crossed linear polarizers then transmits exactly
#     T(theta) = LB_T + (UB_T - LB_T) sin^2(2 theta + phi)
#     UB_T = sin^2(Gamma/2) = 1 - cos^2(Gamma/2)                                  (dissertation Eq. 7.3)
#     LB_T = n3^2 sin^2(Gamma/2) = (CB/2)^2 sinc^2(Gamma/2) = sin^2(Gamma_C/2) cos^2(Gamma_L/2)   (exact)
# with sinc(x) = sin(x)/x and phi = atan2(n2, n1). In pseudopolar terms the dissertation's Eq. 7.4,
# LB_T ~ 1 - cos^2(CB/2) = sin^2(CB/2), is exact for small retardance and for a pure rotator, and in general
#     LB_T = sin^2(CB/2) [sinc(Gamma/2) / sinc(CB/2)]^2
# so its missing scale factor is [sinc(Gamma/2)/sinc(CB/2)]^2 <= 1, which shrinks as linear retardance adds to
# the total. Parallel polarizers transmit 1 - T, so their bounds swap.
#
# Conventions and caveats:
#  - Gamma is the principal value in [0, pi]; thick films whose true retardance passes pi fold back, so CB and
#    LB here are per-wavelength principal values (unwrap across wavelength if a continuous curve is needed).
#    The bounds depend only on sin^2(Gamma/2) and n3^2, so they are the same on every branch.
#  - Signs follow Arteaga and Canillas (2009): LB = -R_H, LB' = -R_45, CB = R_C, where (R_H, R_45, R_C) is the
#    retardance vector from the Lu-Chipman formula R_i = R/(2 sin R) * sum_jk eps_ijk m_jk over the 3x3 retarder
#    block, applied to this code's Mueller matrices. The axis direction is read from the symmetric part (accurate
#    near half-wave) and its sign from that antisymmetric formula; at exactly R = pi the sign is undefined (R and
#    -R are the same element there) and is picked arbitrarily. Whether this code's own Mueller conventions (e.g.
#    the sign of S3) match the paper's is a separate check.
#  - The product split (Gamma_C, Gamma_L) is ill-conditioned when Gamma ~ pi AND n3 ~ 0 (e.g. the symmetric
#    3TR); the pseudopolar values and the bounds are not affected.
#  - No depolarization/diattenuation handling: for measured elements, decompose first (Mueller-Jones part).
#
# Each wavelength peaks at theta_peak = (pi/4 - phi/2) mod pi/2: a wavelength-dependent axis (phi varying with
# wavelength) is what makes a rotated element change color rather than only brightness.
import numpy as np

from Multi_Twist_Main import *

def _signed_axis(R, cos_G):
    #Unit retardance axis of each 3x3 retarder block in an (..., 3, 3) stack, in the Lu-Chipman sign convention.
    #Direction from the symmetric part, cos(G) I + (1 - cos(G)) n n^T, which stays accurate near half-wave; sign
    #from the antisymmetric part via a_i ~ sum_jk eps_ijk m_jk = (m23 - m32, m31 - m13, m12 - m21) (1-based).
    #Returns zeros where there is no retardance.
    symmetric = (R + np.swapaxes(R, -1, -2))/2
    one_minus_cos = 1 - cos_G
    with np.errstate(invalid="ignore", divide="ignore"):
        nn = (symmetric - cos_G[..., None, None]*np.eye(3)) / one_minus_cos[..., None, None]
    column = np.argmax(np.nan_to_num(np.diagonal(nn, axis1=-2, axis2=-1)), axis=-1)
    picked = np.take_along_axis(nn, column[..., None, None], axis=-1)[..., 0]                      # nn[..., :, column]
    scale = np.sqrt(np.clip(np.take_along_axis(np.diagonal(nn, axis1=-2, axis2=-1), column[..., None], axis=-1), 1e-300, None))
    axis = picked/scale
    lu_chipman = np.stack([R[..., 1, 2] - R[..., 2, 1], R[..., 2, 0] - R[..., 0, 2], R[..., 0, 1] - R[..., 1, 0]], axis=-1)
    axis = np.where((np.sum(axis*lu_chipman, axis=-1) < 0)[..., None], -axis, axis)
    return np.where((one_minus_cos < 1e-12)[..., None], 0.0, np.nan_to_num(axis))

def equivalent_retarder_parameters(muller_matrices):
    #Equivalent elliptical retarder of each Mueller matrix in an (..., 4, 4) stack (e.g. one per wavelength).
    #Magnitudes and axis direction come from the symmetric part of the 3x3 retarder block, which stays accurate
    #near half-wave (Gamma = pi) where the antisymmetric part vanishes; the antisymmetric part only fixes the sign
    #of the axis (Lu-Chipman convention). Returns a dict of arrays (radians):
    #  total_retardance  Gamma in [0, pi]
    #  n3_squared        circular fraction of the axis, n3^2 in [0, 1]
    #  axis_phi          phi = atan2(n2, n1), in-plane direction of the axis on the Poincare sphere (mod pi)
    #  retardance_vector   (R_H, R_45, R_C) = Gamma * axis, Lu-Chipman sign convention, shape (..., 3)
    #  pseudopolar_LB, pseudopolar_LB45, pseudopolar_CB   -R_H, -R_45, R_C (Arteaga and Canillas 2009 signs)
    #  pseudopolar_linear  sqrt(LB^2 + LB'^2), the magnitude of the linear birefringence
    #  circular_retardance Gamma_C in [0, pi], linear_retardance Gamma_L in [0, pi]   (product split)
    M = np.asarray(muller_matrices, dtype=float)
    R = M[..., 1:, 1:]
    symmetric = (R + np.swapaxes(R, -1, -2))/2
    cos_G = np.clip((np.trace(R, axis1=-2, axis2=-1) - 1)/2, -1, 1)
    G = np.arccos(cos_G)
    one_minus_cos = 1 - cos_G

    #Symmetric part = cos(G) I + (1 - cos(G)) n n^T, so n n^T follows directly (undefined when G = 0: no retardance)
    with np.errstate(invalid="ignore", divide="ignore"):
        nn = (symmetric - cos_G[..., None, None]*np.eye(3)) / one_minus_cos[..., None, None]
    n3_squared = np.clip(nn[..., 2, 2], 0, 1)
    phi = 0.5*np.arctan2(2*nn[..., 0, 1], nn[..., 0, 0] - nn[..., 1, 1])   #direction of (n1, n2), defined mod pi

    sin_half, cos_half = np.sin(G/2), np.cos(G/2)
    circular = 2*np.arctan2(np.sqrt(n3_squared)*sin_half, cos_half)
    linear = 2*np.arccos(np.clip(np.sqrt(cos_half**2 + n3_squared*sin_half**2), 0, 1))

    no_retardance = one_minus_cos < 1e-12
    n3_squared = np.where(no_retardance, 0.0, n3_squared)
    phi = np.where(no_retardance, 0.0, phi)
    in_plane = np.sqrt(1 - n3_squared)
    R_vector = G[..., None]*_signed_axis(R, cos_G)
    return {"total_retardance": G, "n3_squared": n3_squared, "axis_phi": phi, "retardance_vector": R_vector,
            "pseudopolar_LB": -R_vector[..., 0], "pseudopolar_LB45": -R_vector[..., 1],
            "pseudopolar_CB": R_vector[..., 2], "pseudopolar_linear": G*in_plane,
            "circular_retardance": np.where(no_retardance, 0.0, circular),
            "linear_retardance": np.where(no_retardance, 0.0, linear)}

def rotation_transmission_bounds(MTR_specification, wavelengths, configuration="crossed"):
    #Lowest and highest transmission each wavelength can reach as the element is rotated between fixed linear
    #polarizers, plus the rotation where it peaks. Returns (lower, upper, peak_rotation) arrays, rotation in
    #radians within [0, pi/2). The spec's own initial orientation is included, so peak_rotation is the extra
    #rotation to apply to this spec.
    params = equivalent_retarder_parameters(full_matrix_specification(MTR_specification, np.asarray(wavelengths).reshape(-1)))
    upper = np.sin(params["total_retardance"]/2)**2
    lower = params["n3_squared"]*upper
    peak = (np.pi/4 - params["axis_phi"]/2) % (np.pi/2)
    if configuration == "crossed":
        return lower, upper, peak
    elif configuration == "parallel":
        return 1 - upper, 1 - lower, (peak + np.pi/4) % (np.pi/2)   #parallel peaks where crossed is lowest
    raise ValueError(f'configuration must be "crossed" or "parallel", got {configuration!r}')

def unwrapped_pseudopolar(muller_matrices):
    #Pseudopolar retardance vector (LB, LB', CB) continued smoothly across wavelength, for an N x 4 x 4 stack
    #ordered by wavelength. With the signed axis n, the matrix fixes the retardance vector only up to its
    #equivalent forms (Gamma + 2 pi m) n, so the principal values fold back each time the true retardance
    #passes a multiple of pi. Starting from the LAST entry (longest wavelength, smallest retardance, assumed to be on the
    #principal branch) and moving toward shorter wavelengths, each step picks the equivalent vector closest to
    #the previous one. Returns a dict of arrays: LB, LB45, CB in the Arteaga and Canillas signs (-R_H, -R_45, R_C),
    #retardance_vector (R_H, R_45, R_C) in the Lu-Chipman signs, and total_retardance = |vector|, which can exceed pi.
    M = np.asarray(muller_matrices, dtype=float)
    R = M[:, 1:, 1:]
    cos_G = np.clip((np.trace(R, axis1=-2, axis2=-1) - 1)/2, -1, 1)
    G = np.arccos(cos_G)
    #Signed axes: Gamma and 2 pi - Gamma share a symmetric part, so the sign from the antisymmetric part is what
    #lets the unwrapping tell the two apart
    axes = _signed_axis(R, cos_G)
    vectors = np.zeros((M.shape[0], 3))
    previous = None
    for k in range(M.shape[0] - 1, -1, -1):
        #every equivalent retardance vector of this element is (Gamma + 2 pi m) * axis
        if previous is None:
            vectors[k] = G[k]*axes[k]
        else:
            candidates = [(G[k] + 2*np.pi*m)*axes[k] for m in range(-4, 5)]
            vectors[k] = min(candidates, key=lambda c: np.linalg.norm(c - previous))
        previous = vectors[k]
    return {"LB": -vectors[:, 0], "LB45": -vectors[:, 1], "CB": vectors[:, 2], "retardance_vector": vectors,
            "total_retardance": np.linalg.norm(vectors, axis=1)}

def eq74_lower_bound(MTR_specification, wavelengths, decomposition="pseudopolar", unwrap=True):
    #The dissertation's Eq. 7.4 form, 1 - cos^2(C/2) = sin^2(C/2), without its scale factor. C is the circular
    #retardance from the chosen decomposition: "pseudopolar" (CB, as in the dissertation; missing factor
    #[sinc(Gamma/2)/sinc(CB/2)]^2) or "product" (Gamma_C of a rotator x linear retarder; missing factor cos^2(Gamma_L/2))
    #With unwrap=True (default) CB is continued across wavelength (wavelengths must be in order), so the curve
    #has no kinks where the total retardance passes pi; unwrap=False uses per-wavelength principal values.
    wavelengths = np.asarray(wavelengths).reshape(-1)
    muller = full_matrix_specification(MTR_specification, wavelengths)
    params = equivalent_retarder_parameters(muller)
    if decomposition == "pseudopolar":
        CB = unwrapped_pseudopolar(muller)["CB"] if unwrap else params["pseudopolar_CB"]
        return np.sin(CB/2)**2
    elif decomposition == "product":
        return np.sin(params["circular_retardance"]/2)**2
    raise ValueError(f'decomposition must be "pseudopolar" or "product", got {decomposition!r}')

def predicted_rotation_transmission(MTR_specification, wavelengths, rotations, configuration="crossed"):
    #Transmission (len(rotations) x N) from the closed form LB + (UB - LB) sin^2(2 theta + phi) alone,
    #with no further matrix multiplication: a check on the model and a fast way to sweep one element.
    params = equivalent_retarder_parameters(full_matrix_specification(MTR_specification, np.asarray(wavelengths).reshape(-1)))
    upper = np.sin(params["total_retardance"]/2)**2
    lower = params["n3_squared"]*upper
    theta = np.asarray(rotations, dtype=float).reshape(-1, 1)
    crossed = lower + (upper - lower)*np.sin(2*theta + params["axis_phi"])**2
    if configuration == "crossed":
        return crossed
    elif configuration == "parallel":
        return 1 - crossed
    raise ValueError(f'configuration must be "crossed" or "parallel", got {configuration!r}')
