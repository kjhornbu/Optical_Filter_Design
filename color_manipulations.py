## Cost Functions Utilized in This work
# Analysis Imports
import pandas as pd
import numpy as np
import math
import warnings
import scipy
import os

# importing all the functions defined in Multi_Twist_Main.py 
from Multi_Twist_Main import *

def CIELABColorFunction(t):
    delta=6/29
    
    if (t > (delta**3)):
        colorFunction = t**(1/3)
    else:
        colorFunction = ((1/3)*t*(delta**-2))+(4/29)
    return colorFunction

# CIE 1964 10 degree observer shipped with this repo; the CIE URL is used as a fallback
DEFAULT_OBSERVER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "CIE_xyz_1964_10deg.csv")
CIE_OBSERVER_URL = "https://files.cie.co.at/CIE_xyz_1964_10deg.csv"

# Visible range used for color integration, 1 nm spacing (in nm)
CMF_WAVELENGTHS = np.arange(380,780,1)

_observer_cache = {}

def _trapezoid(y, x):
    #np.trapz was renamed np.trapezoid in NumPy 2.0
    integrate = getattr(np, "trapezoid", None) or np.trapz
    return integrate(y, x)

def loadColorMatchingFunctions(observerPath=None):
    #Returns xBar, yBar, zBar sampled on CMF_WAVELENGTHS. Cached so the CSV is read only once per path.
    if observerPath is None:
        observerPath = DEFAULT_OBSERVER_PATH if os.path.exists(DEFAULT_OBSERVER_PATH) else CIE_OBSERVER_URL
    
    if observerPath not in _observer_cache:
        if os.path.exists(observerPath): 
            standardObservers_CIE1964_10deg = pd.read_csv(observerPath)
        else:
            #load standard observers: https://cie.co.at/datatable/cie-1964-colour-matching-functions-10-degree-observer
            colnames = ['wL', 'xbar', 'ybar','zbar']
            standardObservers_CIE1964_10deg = pd.read_csv(observerPath,names=colnames, header=None)
        
        cmfs = []
        for column in ['xbar','ybar','zbar']:
            bar = np.interp(CMF_WAVELENGTHS, standardObservers_CIE1964_10deg['wL'].values, standardObservers_CIE1964_10deg[column].values)
            bar[np.isnan(bar)] = 0
            cmfs.append(bar)
        _observer_cache[observerPath] = tuple(cmfs)
    
    return _observer_cache[observerPath]

_interpolation_cache = {}

def _interpolation_matrix(wavelengths):
    #Linear interpolation from the given wavelength grid (micron) onto CMF_WAVELENGTHS, written as a matrix W
    #so that T_on_cmf_grid = T @ W.T works for any number of leading dimensions. Same result as np.interp
    #(including holding the end values outside the grid). Cached per wavelength grid.
    wavelengths_nm = np.asarray(wavelengths, dtype=float).reshape(-1)*1e3 # Need to convert to both being in nm hence x1e3
    key = wavelengths_nm.tobytes()
    if key not in _interpolation_cache:
        identity = np.eye(wavelengths_nm.size)
        _interpolation_cache[key] = np.stack([np.interp(CMF_WAVELENGTHS, wavelengths_nm, column) for column in identity], axis=1)
    return _interpolation_cache[key]

def convertTransmissionToXYZ(wavelengths,transmission,observerPath=None):
    #wavelengths in micron, transmission 0-1 with wavelength as the LAST axis (shape (N,) or (..., N)).
    #Returns X,Y,Z normalized so a perfect transmitter has Y = 1; each has the shape of transmission minus its last axis.
    xBar, yBar, zBar = loadColorMatchingFunctions(observerPath)
    
    # Assuming illumination is flat at this point across color range (CIE illuminant E)
    illumnationSourceProfile = 1
    
    transmission = np.asarray(transmission, dtype=float)
    transmission_adjustedRange = transmission @ _interpolation_matrix(wavelengths).T
    transmission_adjustedRange[np.isnan(transmission_adjustedRange)] = 0
    
    # Math Following:  https://www.konicaminolta.com/instruments/knowledge/color/part4/02.html
    # Trapezoidal rule: integrate the spectrum (y) over wavelength (x)
    N = _trapezoid(yBar*illumnationSourceProfile, CMF_WAVELENGTHS)
    
    X = _trapezoid(xBar*illumnationSourceProfile*transmission_adjustedRange, CMF_WAVELENGTHS) / N
    Y = _trapezoid(yBar*illumnationSourceProfile*transmission_adjustedRange, CMF_WAVELENGTHS) / N
    Z = _trapezoid(zBar*illumnationSourceProfile*transmission_adjustedRange, CMF_WAVELENGTHS) / N
    
    return X,Y,Z

def convertMTRToXYZ(wavelengths,stokesData,observerPath=None,configuration="crossed"):
    #Color of the MTR output seen through a horizontal analyzer setup.
    #configuration "crossed": vertical analyzer, transmission (1 - S1)/2 (the original behavior)
    #configuration "parallel": horizontal analyzer, transmission (1 + S1)/2
    #stokesData may be (N, 4) or (..., N, 4); the result drops the last two axes.
    if configuration == "crossed":
        transmission = convertStokesToCrossedLinearHorzTransmission(stokesData)
    elif configuration == "parallel":
        transmission = convertStokesToParallelLinearHorzTransmission(stokesData)
    else:
        raise ValueError(f'configuration must be "crossed" or "parallel", got {configuration!r}')
    return convertTransmissionToXYZ(wavelengths,transmission,observerPath)

def whitePointXYZ(observerPath=None):
    #XYZ (scaled so Yn = 100) of a perfect transmitter under the same flat illuminant used in convertMTRToXYZ
    X, Y, Z = convertTransmissionToXYZ(CMF_WAVELENGTHS/1e3, np.ones(len(CMF_WAVELENGTHS)), observerPath)
    return 100*X, 100*Y, 100*Z

def convertStokesToCrossedCircularTransmission(stokesData):
    transmission = (1-stokesData[...,3])/2
    return transmission

def convertStokesToParallelCircularTransmission(stokesData):
    transmission = (1+stokesData[...,3])/2
    return transmission

def convertStokesToCrossedLinear45Transmission(stokesData):
    transmission = (1-stokesData[...,2])/2
    return transmission

def convertStokesToParallelLinear45Transmission(stokesData):
    transmission = (1+stokesData[...,2])/2
    return transmission

def convertStokesToCrossedLinearHorzTransmission(stokesData):
    transmission = (1-stokesData[...,1])/2
    return transmission

def convertStokesToParallelLinearHorzTransmission(stokesData):
    transmission = (1+stokesData[...,1])/2
    return transmission

def convertXYZToCIELAB(X,Y,Z,Xn=None,Yn=None,Zn=None):
    #X,Y,Z on a 0-1 scale (as returned by convertMTRToXYZ); white point on a 0-100 scale.
    #By default the white point matches the flat (illuminant E) source used in convertMTRToXYZ, so a perfect
    #transmitter maps to a* = b* = 0. Pass D65 (Xn=95.0489,Yn=100,Zn=108.8840) only if the spectra are D65 weighted.
    if Xn is None or Yn is None or Zn is None:
        Xn, Yn, Zn = whitePointXYZ()
    L = 116*CIELABColorFunction((100*Y)/Yn)-16
    a = 500*(CIELABColorFunction((100*X)/Xn) - CIELABColorFunction((100*Y)/Yn))
    b = 200*(CIELABColorFunction((100*Y)/Yn) - CIELABColorFunction((100*Z)/Zn))
     
    return L, a, b


## Chromaticity and gamut
# sRGB primaries in CIE xy. Note sRGB is defined with the 2 degree observer and D65, while this code uses
# the 1964 10 degree observer and a flat source, so sRGB coverage here is an approximate comparison.
SRGB_PRIMARIES_XY = np.array([[0.64, 0.33], [0.30, 0.60], [0.15, 0.06]])

def convertXYZToxy(X,Y,Z):
    #CIE xy chromaticity. Points with no light at all (X+Y+Z == 0) have no defined color and return nan.
    X, Y, Z = np.asarray(X, dtype=float), np.asarray(Y, dtype=float), np.asarray(Z, dtype=float)
    total = X + Y + Z
    with np.errstate(invalid="ignore", divide="ignore"):
        x = np.where(total > 0, X/total, np.nan)
        y = np.where(total > 0, Y/total, np.nan)
    return x, y

def convertXYZTouv(X,Y,Z):
    #CIE 1976 UCS chromaticity u', v'. More perceptually uniform than xy: equal distances are closer to
    #equal perceived color differences. Points with no light (X+15Y+3Z == 0) return nan.
    X, Y, Z = np.asarray(X, dtype=float), np.asarray(Y, dtype=float), np.asarray(Z, dtype=float)
    denominator = X + 15*Y + 3*Z
    with np.errstate(invalid="ignore", divide="ignore"):
        u = np.where(denominator > 0, 4*X/denominator, np.nan)
        v = np.where(denominator > 0, 9*Y/denominator, np.nan)
    return u, v

def convertxyTouv(x,y):
    #Same chromaticity, from xy to u'v' (lines stay lines, so polygons such as gamut triangles stay polygons)
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    denominator = -2*x + 12*y + 3
    return 4*x/denominator, 9*y/denominator

SRGB_PRIMARIES_UV = np.column_stack(convertxyTouv(*SRGB_PRIMARIES_XY.T))

def convertXYZToCIELUV(X,Y,Z,Xn=None,Yn=None,Zn=None):
    #CIE 1976 L*u*v*. X,Y,Z on a 0-1 scale (as returned by convertMTRToXYZ); white point on a 0-100 scale,
    #defaulting to the flat (illuminant E) source used in convertMTRToXYZ, like convertXYZToCIELAB.
    if Xn is None or Yn is None or Zn is None:
        Xn, Yn, Zn = whitePointXYZ()
    X, Y, Z = 100*np.asarray(X, dtype=float), 100*np.asarray(Y, dtype=float), 100*np.asarray(Z, dtype=float)
    un, vn = convertXYZTouv(Xn, Yn, Zn)
    u, v = convertXYZTouv(X, Y, Z)
    t = Y/Yn
    L = np.where(t > (6/29)**3, 116*np.cbrt(t) - 16, (29/3)**3*t)
    with np.errstate(invalid="ignore"):
        u_star = np.where(L > 0, 13*L*(u - un), 0.0)
        v_star = np.where(L > 0, 13*L*(v - vn), 0.0)
    return L, u_star, v_star

def convertXYZToDisplayRGB(X,Y,Z):
    #Approximate on-screen color for plotting: XYZ -> linear sRGB -> gamma-encoded, at TRUE relative brightness
    #(dim settings look dim). Scaled so the flat-illuminant white maps to RGB (1, 1, 1); colors outside sRGB are
    #clipped. Returns an array of shape (..., 3) with values 0-1, ready for matplotlib.
    XYZ_to_sRGB = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]])
    white_rgb = XYZ_to_sRGB @ (np.array(whitePointXYZ())/100)
    rgb = (np.stack([np.asarray(X, dtype=float), np.asarray(Y, dtype=float), np.asarray(Z, dtype=float)], axis=-1) @ XYZ_to_sRGB.T) / white_rgb
    rgb = np.clip(rgb, 0, 1)
    return np.where(rgb <= 0.0031308, 12.92*rgb, 1.055*rgb**(1/2.4) - 0.055)

def _polygon_area(points):
    #Shoelace formula for a polygon given as ordered (K, 2) vertices
    if len(points) < 3:
        return 0.0
    x, y = points[:, 0], points[:, 1]
    return 0.5*abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def _convex_hull_xy(x, y):
    #Ordered hull vertices (K, 2) of the finite (x, y) points, or an empty array if they don't span an area
    from scipy.spatial import ConvexHull, QhullError
    points = np.column_stack([np.ravel(x), np.ravel(y)])
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) < 3:
        return np.empty((0, 2))
    try:
        hull = ConvexHull(points)
    except QhullError:
        return np.empty((0, 2))   #all points on a line: no area
    return points[hull.vertices]

def _clip_polygon(subject, clip):
    #Sutherland-Hodgman: part of convex polygon `subject` inside convex polygon `clip` (both counterclockwise)
    output = list(subject)
    for i in range(len(clip)):
        a, b = clip[i], clip[(i+1) % len(clip)]
        #small tolerance so points lying on the clip edge (up to rounding) count as inside
        inside = lambda p: (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0]) >= -1e-12
        current, output = output, []
        for j in range(len(current)):
            p, q = current[j], current[(j+1) % len(current)]
            if inside(q):
                if not inside(p):
                    output.append(_line_intersection(p, q, a, b))
                output.append(q)
            elif inside(p):
                output.append(_line_intersection(p, q, a, b))
        if not output:
            break
    return np.array(output)

def _line_intersection(p, q, a, b):
    #Point where segment p-q crosses the line through a-b
    d1, d2 = np.subtract(q, p), np.subtract(b, a)
    denominator = d1[0]*d2[1] - d1[1]*d2[0]
    if abs(denominator) < 1e-15:
        return np.asarray(q, dtype=float)   #segment runs along the clip edge: its endpoint is the crossing
    t = ((a[0]-p[0])*d2[1] - (a[1]-p[1])*d2[0]) / denominator
    return np.add(p, t*d1)

def gamut_area_xy(x, y):
    #Area in CIE xy of the convex hull of all reachable colors.
    #The reachable region of a two-rotation sweep can be non-convex, so this is an upper bound on the true area.
    return _polygon_area(_convex_hull_xy(x, y))

def gamut_area_uv(u, v):
    #Same as gamut_area_xy but in CIE 1976 u'v', the usual space for comparing display gamuts
    return _polygon_area(_convex_hull_xy(u, v))

def gamut_coverage_of_sRGB(a, b, space="xy"):
    #Fraction (0-1) of the sRGB triangle covered by the convex hull of the reachable colors.
    #a, b are x, y (space="xy") or u', v' (space="uv"). The fraction differs between spaces since areas are weighted differently.
    if space == "xy":
        reference = SRGB_PRIMARIES_XY
    elif space == "uv":
        reference = SRGB_PRIMARIES_UV
    else:
        raise ValueError(f'space must be "xy" or "uv", got {space!r}')
    hull = _convex_hull_xy(a, b)
    if len(hull) < 3:
        return 0.0
    overlap = _clip_polygon(hull, reference)   #both are counterclockwise (scipy hulls and R->G->B are CCW in xy and u'v')
    return _polygon_area(overlap) / _polygon_area(reference)
