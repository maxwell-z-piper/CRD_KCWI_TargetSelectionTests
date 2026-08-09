#!/usr/bin/env python
# coding: utf-8

"""
Predict KCWI/KCRM adaptive-bin statistics for a configurable CRD target as a function of exposure time.

IMPORTANT CORRECTION
--------------------
There are TWO different S/N quantities:

1. sn_ref
   S/N associated with the exposure-time calibration.
   Example BL:
       S/N_ref = 40
       t_ref   = 4.10 hr
       mu_ref  = 20.70 mag/arcsec^2

2. TARGET_SN
   Final adaptive-binning target.
   Current choice:
       BL  = 30
       RH3 = 35

Changing TARGET_SN must NOT change sn_ref.

Procedure
---------
1. Use MaNGA DR17 flux distribution as a surface-brightness template.
2. Rotate the selected KCWI slicer field so the 20.4-arcsec dimension is at PA_kin.
3. Sample the field at the selected slicer width and detector spatial pixel scale.
4. Convert surface brightness into expected S/N using fixed exposure calibrations.
5. Adaptive-bin to BL S/N=30 and RH3 S/N=35.
6. Repeat for 2, 3, 4, 5, 6 hr.

To run
---------
- Need to be in the correct conda environment to get PowerBin working:
1. cd into correct directory
2. "conda activate powerbin_env"
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.ndimage import gaussian_filter

# =============================================================================
# USER CONFIGURATION
# =============================================================================

ROOT_CANDIDATES = [
    Path("/Users/maxpiper/Desktop/CRD_Thesis"),
    Path("/users/maxpiper/Desktop/CRD_Thesis"),
]

CRD_ROOT = next((p for p in ROOT_CANDIDATES if p.exists()), ROOT_CANDIDATES[0])
CRD_Decom_Root = Path("/Users/maxpiper/Desktop/CRD_Decomposition")

# =============================================================================
# SHARED TARGET CONFIGURATION
# =============================================================================
# All target/slicer quantities are read from target_config.txt in the same
# directory as this script. Edit that ONE file when switching targets/slicers.
CONFIG_PATH = Path(__file__).resolve().parent / "target_config.txt"


def load_target_config(path):
    """Read a simple KEY = VALUE configuration file."""
    if not path.exists():
        raise FileNotFoundError(
            "Target configuration file not found:\n{}".format(path)
        )

    config = {}
    with open(str(path), "r") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            if "=" not in line:
                raise ValueError(
                    "Invalid configuration line {}:\n{}\n"
                    "Expected format: KEY = VALUE".format(
                        line_number, raw_line.rstrip()
                    )
                )
            key, value = line.split("=", 1)
            config[key.strip()] = value.strip()
    return config


CONFIG = load_target_config(CONFIG_PATH)

REQUIRED_TARGET_KEYS = [
    "SLICER",
    "TARGET_NAME",
    "TARGET_REDSHIFT",
    "V_A",
    "V_B",
    "SIGMA_A",
    "SIGMA_B",
]
missing = [key for key in REQUIRED_TARGET_KEYS if key not in CONFIG]
if missing:
    raise KeyError(
        "Missing required entries in target_config.txt:\n{}".format(
            "\n".join(missing)
        )
    )

SLICER = CONFIG["SLICER"]
TARGET_NAME = CONFIG["TARGET_NAME"]
TARGET_REDSHIFT = float(CONFIG["TARGET_REDSHIFT"])
V_A = float(CONFIG["V_A"])
V_B = float(CONFIG["V_B"])
SIGMA_A = float(CONFIG["SIGMA_A"])
SIGMA_B = float(CONFIG["SIGMA_B"])
TARGET_FRAC_A = float(CONFIG.get("TARGET_FRAC_A", 0.50))

# Backward-compatible aliases used by the existing analysis code.
TARGET_V_A = V_A
TARGET_V_B = V_B
TARGET_SIGMA_A = SIGMA_A
TARGET_SIGMA_B = SIGMA_B
Z_TARGET = TARGET_REDSHIFT
DELTA_V = abs(TARGET_V_A - TARGET_V_B)

if TARGET_REDSHIFT < 0:
    raise ValueError("TARGET_REDSHIFT must be >= 0.")
if SIGMA_A <= 0 or SIGMA_B <= 0:
    raise ValueError("SIGMA_A and SIGMA_B must both be > 0 km/s.")
if not (0.0 < TARGET_FRAC_A < 1.0):
    raise ValueError("TARGET_FRAC_A must lie between 0 and 1.")


# Additional target quantities used only by the spatial/PowerBin planning code.
POWERBIN_REQUIRED_KEYS = ["PLATEIFU", "PA_KIN_DEG"]
missing_powerbin = [key for key in POWERBIN_REQUIRED_KEYS if key not in CONFIG]
if missing_powerbin:
    raise KeyError(
        "Missing PowerBin entries in target_config.txt:\n{}".format(
            "\n".join(missing_powerbin)
        )
    )

PLATEIFU = CONFIG["PLATEIFU"]
Z_GAL = TARGET_REDSHIFT
PA_KIN_DEG = float(CONFIG["PA_KIN_DEG"])
TWO_SIGMA_RADIUS_ARCSEC = float(CONFIG.get("TWO_SIGMA_RADIUS_ARCSEC", 8.0))
TWO_SIGMA_RADIAL_HALF_WIDTH_ARCSEC = float(
    CONFIG.get("TWO_SIGMA_RADIAL_HALF_WIDTH_ARCSEC", 1.5)
)
TWO_SIGMA_MINOR_HALF_WIDTH_ARCSEC = float(
    CONFIG.get("TWO_SIGMA_MINOR_HALF_WIDTH_ARCSEC", 1.0)
)
IFU_CENTER_X_ARCSEC = float(CONFIG.get("IFU_CENTER_X_ARCSEC", 0.0))
IFU_CENTER_Y_ARCSEC = float(CONFIG.get("IFU_CENTER_Y_ARCSEC", 0.0))

# Standard MaNGA DAP MAPS filename derived from PLATEIFU.
MAPS_PATH = (
    CRD_ROOT
    / "CRD_MAPS"
    / ("manga-{}-MAPS-HYB10-MILESHC-MASTARSSP.fits.gz".format(PLATEIFU))
)

# SLICER is read from target_config.txt.
# If True, orient whichever IFU dimension is LONGEST along PA_KIN_DEG. This
# reproduces the original Small/Medium orientation (20.4" along the major axis)
# and automatically rotates Large so its 33" dimension spans the major axis.
ALIGN_LONGEST_IFU_AXIS_WITH_KINEMATIC_PA = True

# Geometry uses current KCWI slicer dimensions. Along-slice sampling is doubled
# for the normal 2x2 detector binning used with Medium/Large. The Small values
# reproduce the original calculation exactly.
SLICER_CONFIGS = {
    "Small": {
        "width_arcsec": 8.4,
        "length_arcsec": 20.4,
        "slice_width_arcsec": 0.35,
        "along_slice_pixel_arcsec": 0.147,
        "detector_binning": 1,
    },
    "Medium": {
        "width_arcsec": 16.5,
        "length_arcsec": 20.4,
        "slice_width_arcsec": 0.69,
        "along_slice_pixel_arcsec": 0.294,
        "detector_binning": 2,
    },
    "Large": {
        "width_arcsec": 33.0,
        "length_arcsec": 20.4,
        "slice_width_arcsec": 1.35,
        "along_slice_pixel_arcsec": 0.294,
        "detector_binning": 2,
    },
}

_slicer_lookup = {key.lower(): key for key in SLICER_CONFIGS}
_slicer_key = _slicer_lookup.get(str(SLICER).strip().lower())
if _slicer_key is None:
    raise ValueError(
        "Unknown SLICER={!r}. Use 'Small', 'Medium', or 'Large'.".format(SLICER)
    )
SLICER = _slicer_key
_SLICER_CFG = SLICER_CONFIGS[SLICER]
KCWI_WIDTH_ARCSEC = float(_SLICER_CFG["width_arcsec"])
KCWI_LENGTH_ARCSEC = float(_SLICER_CFG["length_arcsec"])
SLICE_WIDTH_ARCSEC = float(_SLICER_CFG["slice_width_arcsec"])
ALONG_SLICE_PIXEL_ARCSEC = float(_SLICER_CFG["along_slice_pixel_arcsec"])

# The physical 20.4" direction is sampled along slices; the variable-width
# direction is sampled by the slice width. Choose which one becomes u (the
# kinematic-major-axis coordinate) according to the orientation setting above.
if (
    ALIGN_LONGEST_IFU_AXIS_WITH_KINEMATIC_PA
    and KCWI_WIDTH_ARCSEC > KCWI_LENGTH_ARCSEC
):
    IFU_U_EXTENT_ARCSEC = KCWI_WIDTH_ARCSEC
    IFU_V_EXTENT_ARCSEC = KCWI_LENGTH_ARCSEC
    IFU_U_PIXEL_ARCSEC = SLICE_WIDTH_ARCSEC
    IFU_V_PIXEL_ARCSEC = ALONG_SLICE_PIXEL_ARCSEC
    IFU_AXIS_ALONG_PA = "variable-width slicer axis"
else:
    IFU_U_EXTENT_ARCSEC = KCWI_LENGTH_ARCSEC
    IFU_V_EXTENT_ARCSEC = KCWI_WIDTH_ARCSEC
    IFU_U_PIXEL_ARCSEC = ALONG_SLICE_PIXEL_ARCSEC
    IFU_V_PIXEL_ARCSEC = SLICE_WIDTH_ARCSEC
    IFU_AXIS_ALONG_PA = "20.4-arcsec along-slice axis"

TARGET_TAG = "".join(c if (c.isalnum() or c in "-_") else "_" for c in TARGET_NAME)

# =============================================================================
# SCIENCE WAVELENGTH WINDOWS
# =============================================================================

BLUE_REST_WINDOW = (4800.0, 5500.0)
RH3_REST_WINDOW = (8470.0, 8900.0)

# =============================================================================
# EXPOSURE GRID
# =============================================================================

EXPOSURE_HOURS = np.array([2, 3, 4, 5, 6], dtype=float)

# =============================================================================
# DESIRED FINAL BIN S/N
# =============================================================================

TARGET_SN = {
    "BL": 30.0,
    "RH3": 30.0,
}

# =============================================================================
# EXPOSURE-TIME CALIBRATION
# =============================================================================
#
# sn_ref is the S/N associated with t_ref_hr at mu_ref.
# DO NOT change sn_ref when changing TARGET_SN.
#
# IMPORTANT WHEN CHANGING TARGET OR SLICER:
# These exposure-calibration anchors came from the previous planning setup.
# Re-run the BL/RH3 ETC for the new target/slicer and update mu_ref, t_ref_hr,
# and sn_ref here before treating the predicted bin counts quantitatively.
#

ARM_CALIBRATION = {
    "BL": {
        "mu_ref": 20.70,
        "t_ref_hr": 4.10,
        "sn_ref": 40.0,
    },
    "RH3": {
        "mu_ref": 20.00,
        "t_ref_hr": 2.98,
        "sn_ref": 35.0,
    },
}

# =============================================================================
# FALLBACK COLOR
# =============================================================================

FALLBACK_G_MINUS_RH3_MAG = 1.50

# =============================================================================
# FAINT-END MASK
# =============================================================================

MAX_USEFUL_MU = {
    "BL": 25.0,
    "RH3": 24.5,
}

# =============================================================================
# MaNGA MAP SMOOTHING
# =============================================================================

MANGA_SMOOTH_SIGMA_SPAX = 0.75

# =============================================================================
# OUTPUT DIRECTORY
# =============================================================================

FOLDER_NAME = (
    "KCWI_Voronoi_Planning_{}_{}_BlueSN{}_RedSN{}".format(
        TARGET_TAG,
        SLICER,
        int(TARGET_SN["BL"]),
        int(TARGET_SN["RH3"]),
    )
)
OUTPUT_DIR = CRD_Decom_Root / FOLDER_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# POWERBIN / VORBIN IMPORT
# =============================================================================

BINNING_BACKEND = None

try:
    from powerbin import PowerBin
    BINNING_BACKEND = "powerbin"
except Exception:
    PowerBin = None

if BINNING_BACKEND is None:
    try:
        from vorbin.voronoi_2d_binning import voronoi_2d_binning
        BINNING_BACKEND = "vorbin"
        warnings.warn(
            "\nPowerBin is not installed.\n"
            "Falling back to legacy VorBin.\n\n"
            "To install PowerBin:\n"
            "    pip install powerbin\n"
        )
    except Exception as exc:
        raise ImportError(
            "\nNeither PowerBin nor VorBin is installed.\n\n"
            "Install PowerBin with:\n"
            "    pip install powerbin\n\n"
            "or legacy VorBin with:\n"
            "    pip install vorbin\n"
        ) from exc

# =============================================================================
# CONSTANTS
# =============================================================================

C_A_PER_S = 2.99792458e18
MANGA_SPAXEL_AREA_ARCSEC2 = 0.5 * 0.5

# =============================================================================
# NAN-SAFE SMOOTHING
# =============================================================================

def robust_nan_gaussian(image, sigma):
    image = np.asarray(image, dtype=float)
    good = np.isfinite(image)

    if sigma is None or sigma <= 0:
        return image.copy()

    val = np.where(good, image, 0.0)
    wt = good.astype(float)

    val_s = gaussian_filter(val, sigma=sigma, mode="nearest")
    wt_s = gaussian_filter(wt, sigma=sigma, mode="nearest")

    out = np.full_like(image, np.nan, dtype=float)
    ok = wt_s > 1e-8
    out[ok] = val_s[ok] / wt_s[ok]

    return out

# =============================================================================
# FLAMBDA -> AB SURFACE BRIGHTNESS
# =============================================================================

def flam_to_mu_ab(flam_per_arcsec2, wavelength_A):
    flam = np.asarray(flam_per_arcsec2, dtype=float)
    fnu = flam * float(wavelength_A) ** 2 / C_A_PER_S

    mu = np.full_like(fnu, np.nan, dtype=float)
    good = np.isfinite(fnu) & (fnu > 0)

    mu[good] = -2.5 * np.log10(fnu[good]) - 48.60
    return mu

# =============================================================================
# SURFACE BRIGHTNESS -> RELATIVE FLUX
# =============================================================================

def mu_to_relative_flux(mu, mu_ref):
    return 10.0 ** (-0.4 * (np.asarray(mu) - float(mu_ref)))

# =============================================================================
# FIND MaNGA DRP CUBE
# =============================================================================

def locate_drp_cube(root):
    patterns = [
        f"manga-{PLATEIFU}-LOGCUBE.fits.gz",
        f"manga-{PLATEIFU}-LINCUBE.fits.gz",
        f"manga-{PLATEIFU}-LOGCUBE.fits",
        f"manga-{PLATEIFU}-LINCUBE.fits",
    ]

    for pat in patterns:
        matches = list(root.rglob(pat))
        if matches:
            return matches[0]

    return None

# =============================================================================
# LOAD DAP MAPS FILE
# =============================================================================

def load_manga_maps(maps_path):
    if not maps_path.exists():
        raise FileNotFoundError(f"MAPS file not found:\n{maps_path}")

    with fits.open(maps_path) as hdul:
        x = np.asarray(hdul["SPX_SKYCOO"].data[0], dtype=float)
        y = np.asarray(hdul["SPX_SKYCOO"].data[1], dtype=float)
        mflux = np.asarray(hdul["SPX_MFLUX"].data, dtype=float)

        if "SPX_MFLUX_IVAR" in hdul:
            ivar = np.asarray(hdul["SPX_MFLUX_IVAR"].data, dtype=float)
            mflux[(~np.isfinite(ivar)) | (ivar <= 0)] = np.nan

    flam_per_arcsec2 = (
        mflux
        * 1e-17
        / MANGA_SPAXEL_AREA_ARCSEC2
    )

    flam_per_arcsec2 = robust_nan_gaussian(
        flam_per_arcsec2,
        MANGA_SMOOTH_SIGMA_SPAX,
    )

    return x, y, flam_per_arcsec2

# =============================================================================
# SYNTHETIC CONTINUUM IMAGE FROM MaNGA DRP CUBE
# =============================================================================

def synthesize_window_map_from_drp_cube(
        cube_path,
        x_manga,
        y_manga,
        rest_window):

    obs_lo = rest_window[0] * (1.0 + Z_GAL)
    obs_hi = rest_window[1] * (1.0 + Z_GAL)

    with fits.open(cube_path, memmap=True) as hdul:
        wave = np.asarray(hdul["WAVE"].data, dtype=float)
        flux = np.asarray(hdul["FLUX"].data, dtype=float)
        ivar = np.asarray(hdul["IVAR"].data, dtype=float)

        if "MASK" in hdul:
            mask = np.asarray(hdul["MASK"].data)
        else:
            mask = np.zeros_like(flux, dtype=np.int64)

        use_wave = (wave >= obs_lo) & (wave <= obs_hi)

        if np.sum(use_wave) < 10:
            raise RuntimeError(
                f"Too few wavelength channels inside "
                f"{rest_window[0]}-{rest_window[1]} A rest-frame window."
            )

        f = flux[use_wave].copy()
        iv = ivar[use_wave]
        mk = mask[use_wave]

        bad = (
            (~np.isfinite(f))
            | (~np.isfinite(iv))
            | (iv <= 0)
            | (mk != 0)
        )

        f[bad] = np.nan

        band = np.nanmedian(f, axis=0)
        lambda_eff_obs = float(np.nanmedian(wave[use_wave]))

    if band.shape != x_manga.shape:
        raise RuntimeError(
            f"DRP cube spatial shape {band.shape} does not match "
            f"MAPS shape {x_manga.shape}."
        )

    flam_per_arcsec2 = (
        band
        * 1e-17
        / MANGA_SPAXEL_AREA_ARCSEC2
    )

    flam_per_arcsec2 = robust_nan_gaussian(
        flam_per_arcsec2,
        MANGA_SMOOTH_SIGMA_SPAX,
    )

    return flam_per_arcsec2, lambda_eff_obs

# =============================================================================
# BUILD ROTATED KCWI GRID
# =============================================================================

def build_kcwi_grid(pa_deg=PA_KIN_DEG):
    u = np.arange(
        -IFU_U_EXTENT_ARCSEC / 2 + IFU_U_PIXEL_ARCSEC / 2,
        +IFU_U_EXTENT_ARCSEC / 2,
        IFU_U_PIXEL_ARCSEC,
    )

    v = np.arange(
        -IFU_V_EXTENT_ARCSEC / 2 + IFU_V_PIXEL_ARCSEC / 2,
        +IFU_V_EXTENT_ARCSEC / 2,
        IFU_V_PIXEL_ARCSEC,
    )

    uu, vv = np.meshgrid(u, v)
    pa = np.deg2rad(pa_deg)

    # PA measured east of north.
    x = (
        IFU_CENTER_X_ARCSEC
        + uu * np.sin(pa)
        + vv * np.cos(pa)
    )

    y = (
        IFU_CENTER_Y_ARCSEC
        + uu * np.cos(pa)
        - vv * np.sin(pa)
    )

    pixel_area = IFU_U_PIXEL_ARCSEC * IFU_V_PIXEL_ARCSEC

    return uu, vv, x, y, pixel_area

# =============================================================================
# INTERPOLATE MaNGA IMAGE ONTO KCWI FIELD
# =============================================================================

def interpolate_manga_to_kcwi(
        x_manga,
        y_manga,
        image_manga,
        x_kcwi,
        y_kcwi):

    good = (
        np.isfinite(x_manga)
        & np.isfinite(y_manga)
        & np.isfinite(image_manga)
        & (image_manga > 0)
    )

    pts = np.column_stack([
        x_manga[good],
        y_manga[good],
    ])

    vals = image_manga[good]

    lin = LinearNDInterpolator(
        pts,
        vals,
        fill_value=np.nan,
    )

    near = NearestNDInterpolator(
        pts,
        vals,
    )

    out = lin(x_kcwi, y_kcwi)
    missing = ~np.isfinite(out)

    if np.any(missing):
        out[missing] = near(
            x_kcwi[missing],
            y_kcwi[missing],
        )

    return np.asarray(out, dtype=float)

# =============================================================================
# PREDICT SPATIAL SIGNAL + NOISE
# =============================================================================

def make_sn_arrays(
        mu_map,
        arm,
        exposure_hr,
        pixel_area_arcsec2):

    """
    Construct abstract spatial signal and noise arrays.

    The absolute scale is set by:
        sn_ref
        t_ref_hr
        mu_ref

    NOT by TARGET_SN.

    Example BL:
        mu = 20.70
        t = 4.10 hr
        S/N = 40 per resolution element

    TARGET_SN["BL"] can then independently be set to 30.
    """

    cfg = ARM_CALIBRATION[arm]

    mu_ref = cfg["mu_ref"]
    t_ref = cfg["t_ref_hr"]
    sn_ref = cfg["sn_ref"]

    flux_ratio = mu_to_relative_flux(
        mu_map,
        mu_ref,
    )

    time_factor = np.sqrt(
        float(exposure_hr)
        / float(t_ref)
    )

    signal = (
        sn_ref
        * time_factor
        * flux_ratio
        * pixel_area_arcsec2
    )

    noise = np.full_like(
        signal,
        np.sqrt(pixel_area_arcsec2),
        dtype=float,
    )

    return signal, noise

# =============================================================================
# CALCULATE S/N OF A BIN
# =============================================================================

def bin_sn(indices, signal, noise):
    indices = np.asarray(indices, dtype=int)

    s = np.sum(signal[indices])
    n2 = np.sum(noise[indices] ** 2)

    if n2 <= 0:
        return 0.0

    return s / np.sqrt(n2)

# =============================================================================
# ADAPTIVE BINNING
# =============================================================================

def run_adaptive_binning(
        x,
        y,
        signal,
        noise,
        target_sn):

    xy = np.column_stack([x, y])

    if BINNING_BACKEND == "powerbin":

        def capacity_spec(index):
            return bin_sn(
                index,
                signal,
                noise,
            ) ** 2

        pb = PowerBin(
            xy,
            capacity_spec,
            target_capacity=float(target_sn) ** 2,
            verbose=0,
        )

        return np.asarray(pb.bin_num, dtype=int)

    result = voronoi_2d_binning(
        x,
        y,
        signal,
        noise,
        float(target_sn),
        cvt=True,
        wvt=True,
        plot=False,
        quiet=True,
    )

    return np.asarray(result[0], dtype=int)

# =============================================================================
# ANALYZE ONE ARM / ONE EXPOSURE
# =============================================================================

def analyze_one_case(
        arm,
        exposure_hr,
        uu,
        vv,
        x_sky,
        y_sky,
        mu_map,
        pixel_area):

    target_sn = TARGET_SN[arm]
    max_mu = MAX_USEFUL_MU[arm]

    signal_map, noise_map = make_sn_arrays(
        mu_map,
        arm,
        exposure_hr,
        pixel_area,
    )

    valid = (
        np.isfinite(mu_map)
        & np.isfinite(signal_map)
        & np.isfinite(noise_map)
        & (signal_map > 0)
        & (noise_map > 0)
        & (mu_map <= max_mu)
    )

    if np.sum(valid) < 10:
        raise RuntimeError(
            f"{arm} {exposure_hr} hr: too few valid KCWI pixels."
        )

    xv = x_sky[valid].ravel()
    yv = y_sky[valid].ravel()
    sv = signal_map[valid].ravel()
    nv = noise_map[valid].ravel()
    uv = uu[valid].ravel()
    vv_valid = vv[valid].ravel()

    total_sn = bin_sn(
        np.arange(sv.size),
        sv,
        nv,
    )

    if total_sn < target_sn:
        warnings.warn(
            f"{arm} {exposure_hr:.1f} hr: entire valid field has only "
            f"S/N={total_sn:.1f}, below target {target_sn:.1f}."
        )

    bin_num_valid = run_adaptive_binning(
        xv,
        yv,
        sv,
        nv,
        target_sn,
    )

    bin_map = np.full(
        mu_map.shape,
        -1,
        dtype=int,
    )

    bin_map[valid] = bin_num_valid

    rows = []
    unique_bins = np.unique(bin_num_valid)

    for b in unique_bins:
        idx = np.where(bin_num_valid == b)[0]

        if idx.size == 0:
            continue

        weights = sv[idx]

        if np.sum(weights) <= 0:
            weights = np.ones_like(weights)

        xbar = np.average(
            xv[idx],
            weights=weights,
        )

        ybar = np.average(
            yv[idx],
            weights=weights,
        )

        ubar = np.average(
            uv[idx],
            weights=weights,
        )

        vbar = np.average(
            vv_valid[idx],
            weights=weights,
        )

        area = idx.size * pixel_area
        sn = bin_sn(idx, sv, nv)

        radius = np.hypot(
            xbar - IFU_CENTER_X_ARCSEC,
            ybar - IFU_CENTER_Y_ARCSEC,
        )

        equiv_diameter = 2.0 * np.sqrt(
            area / np.pi
        )

        rows.append({
            "arm": arm,
            "exposure_hr": float(exposure_hr),
            "bin_id": int(b),
            "n_spatial_pixels": int(idx.size),
            "area_arcsec2": float(area),
            "equiv_diameter_arcsec": float(equiv_diameter),
            "sn_pred": float(sn),
            "x_arcsec": float(xbar),
            "y_arcsec": float(ybar),
            "u_major_arcsec": float(ubar),
            "v_minor_arcsec": float(vbar),
            "r_arcsec": float(radius),
        })

    bins = pd.DataFrame(rows)

    if len(bins) > 0:
        frac_at_target = np.mean(
            bins["sn_pred"]
            >= 0.95 * target_sn
        )
    else:
        frac_at_target = np.nan

    # Approximate two-sigma regions around +/-TWO_SIGMA_RADIUS_ARCSEC.
    two_sigma_lo = max(
        0.0,
        TWO_SIGMA_RADIUS_ARCSEC - TWO_SIGMA_RADIAL_HALF_WIDTH_ARCSEC,
    )
    two_sigma_hi = (
        TWO_SIGMA_RADIUS_ARCSEC + TWO_SIGMA_RADIAL_HALF_WIDTH_ARCSEC
    )
    two_sigma_region = (
        (np.abs(bins["u_major_arcsec"]) >= two_sigma_lo)
        & (np.abs(bins["u_major_arcsec"]) <= two_sigma_hi)
        & (np.abs(bins["v_minor_arcsec"]) <= TWO_SIGMA_MINOR_HALF_WIDTH_ARCSEC)
    )

    summary = {
        "target": TARGET_NAME,
        "plateifu": PLATEIFU,
        "slicer": SLICER,
        "ifu_u_extent_arcsec": float(IFU_U_EXTENT_ARCSEC),
        "ifu_v_extent_arcsec": float(IFU_V_EXTENT_ARCSEC),
        "arm": arm,
        "exposure_hr": float(exposure_hr),
        "sn_ref": float(ARM_CALIBRATION[arm]["sn_ref"]),
        "t_ref_hr": float(ARM_CALIBRATION[arm]["t_ref_hr"]),
        "mu_ref": float(ARM_CALIBRATION[arm]["mu_ref"]),
        "target_sn": float(target_sn),
        "n_bins": int(len(bins)),
        "total_valid_area_arcsec2": float(np.sum(valid) * pixel_area),
        "median_bin_area_arcsec2": float(
            np.nanmedian(bins["area_arcsec2"])
        ),
        "p16_bin_area_arcsec2": float(
            np.nanpercentile(
                bins["area_arcsec2"],
                16,
            )
        ),
        "p84_bin_area_arcsec2": float(
            np.nanpercentile(
                bins["area_arcsec2"],
                84,
            )
        ),
        "median_equiv_diameter_arcsec": float(
            np.nanmedian(
                bins["equiv_diameter_arcsec"]
            )
        ),
        "median_bin_sn": float(
            np.nanmedian(
                bins["sn_pred"]
            )
        ),
        "fraction_bins_within_5pct_target": float(frac_at_target),
        "n_bins_r_lt_4": int(
            np.sum(
                bins["r_arcsec"] < 4.0
            )
        ),
        "n_bins_r_4_6": int(
            np.sum(
                (bins["r_arcsec"] >= 4.0)
                & (bins["r_arcsec"] < 6.0)
            )
        ),
        "n_bins_r_6_8": int(
            np.sum(
                (bins["r_arcsec"] >= 6.0)
                & (bins["r_arcsec"] < 8.0)
            )
        ),
        "n_bins_r_ge_8": int(
            np.sum(
                bins["r_arcsec"] >= 8.0
            )
        ),
        "n_bins_two_sigma_region": int(
            np.sum(
                two_sigma_region
            )
        ),
    }

    return (
        summary,
        bins,
        bin_map,
        valid,
        signal_map,
        noise_map,
    )

# =============================================================================
# PLOT SURFACE-BRIGHTNESS MAP
# =============================================================================

def plot_surface_brightness(
        uu,
        vv,
        mu_map,
        arm,
        source_label):

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    im = ax.pcolormesh(
        uu,
        vv,
        mu_map,
        shading="auto",
        cmap="magma_r",
    )

    cb = fig.colorbar(
        im,
        ax=ax,
    )

    cb.set_label(
        r"AB surface brightness [mag arcsec$^{-2}$]"
    )

    ax.set_xlabel(
        "Along PA={:.1f} deg major axis, u [arcsec]".format(PA_KIN_DEG)
    )

    ax.set_ylabel(
        "Across major axis, v [arcsec]"
    )

    ax.set_title(
        f"{arm}: MaNGA-based surface-brightness template\n"
        f"{source_label}"
    )

    ax.set_aspect("equal")

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / f"{arm}_surface_brightness_template.png",
        dpi=180,
    )

    plt.close(fig)

# =============================================================================
# PLOT BIN MAP
# =============================================================================

def plot_bin_map(
        uu,
        vv,
        mu_map,
        bin_map,
        bins,
        arm,
        exposure_hr):

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.pcolormesh(
        uu,
        vv,
        mu_map,
        shading="auto",
        cmap="Greys_r",
        alpha=0.40,
    )

    masked_bins = np.ma.masked_where(
        bin_map < 0,
        bin_map,
    )

    ax.pcolormesh(
        uu,
        vv,
        masked_bins,
        shading="auto",
        cmap="nipy_spectral",
        alpha=0.85,
    )

    ax.scatter(
        bins["u_major_arcsec"],
        bins["v_minor_arcsec"],
        s=5,
        c="black",
        alpha=0.6,
    )

    # Approximate +/- two-sigma locations.
    ax.scatter(
        [-TWO_SIGMA_RADIUS_ARCSEC, +TWO_SIGMA_RADIUS_ARCSEC],
        [0.0, 0.0],
        marker="*",
        s=180,
        facecolors="none",
        edgecolors="black",
        linewidths=1.5,
        label=r"Approx. $2\sigma$ locations",
    )

    # Outline approximate two-sigma diagnostic regions.
    radial_lo = max(
        0.0,
        TWO_SIGMA_RADIUS_ARCSEC - TWO_SIGMA_RADIAL_HALF_WIDTH_ARCSEC,
    )
    radial_hi = TWO_SIGMA_RADIUS_ARCSEC + TWO_SIGMA_RADIAL_HALF_WIDTH_ARCSEC
    minor = TWO_SIGMA_MINOR_HALF_WIDTH_ARCSEC

    for sign in [-1, +1]:
        if sign < 0:
            x1 = -radial_hi
            x2 = -radial_lo
        else:
            x1 = +radial_lo
            x2 = +radial_hi

        ax.plot(
            [x1, x2, x2, x1, x1],
            [-minor, -minor, +minor, +minor, -minor],
            ls="--",
            lw=1.0,
            c="black",
            alpha=0.7,
        )

    ax.axvline(
        0,
        lw=0.8,
        alpha=0.5,
    )

    ax.axhline(
        0,
        lw=0.8,
        alpha=0.5,
    )

    ax.set_xlabel(
        "Along kinematic major axis, u [arcsec]"
    )

    ax.set_ylabel(
        "Across kinematic major axis, v [arcsec]"
    )

    ax.set_title(
        f"{arm}: {exposure_hr:.0f} hr, "
        f"target S/N={TARGET_SN[arm]:.0f}, "
        f"N_bins={len(bins)}"
    )

    ax.set_aspect("equal")

    ax.legend(
        loc="upper right",
        fontsize=8,
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / f"{arm}_bins_{int(exposure_hr)}hr.png",
        dpi=180,
    )

    plt.close(fig)

# =============================================================================
# SUMMARY PLOTS
# =============================================================================

def plot_summary(summary_df):

    # Number of bins versus exposure time.
    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    for arm, sub in summary_df.groupby("arm"):
        sub = sub.sort_values("exposure_hr")

        ax.plot(
            sub["exposure_hr"],
            sub["n_bins"],
            marker="o",
            label=arm,
        )

    ax.set_xlabel("Exposure time [hr]")
    ax.set_ylabel("Predicted number of useful bins")
    ax.set_title("KCWI/KCRM predicted adaptive-bin count")
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / "bin_count_vs_exposure.png",
        dpi=180,
    )

    plt.close(fig)

    # Median bin area versus exposure.
    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    for arm, sub in summary_df.groupby("arm"):
        sub = sub.sort_values("exposure_hr")

        ax.plot(
            sub["exposure_hr"],
            sub["median_bin_area_arcsec2"],
            marker="o",
            label=arm,
        )

    ax.set_xlabel("Exposure time [hr]")
    ax.set_ylabel(
        r"Median bin area [arcsec$^2$]"
    )
    ax.set_title(
        "Spatial sampling gained with exposure time"
    )
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / "median_bin_area_vs_exposure.png",
        dpi=180,
    )

    plt.close(fig)

    # Outer-disk counts.
    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    width = 0.35

    for arm_index, (arm, sub) in enumerate(
            summary_df.groupby("arm")):

        sub = sub.sort_values("exposure_hr")

        x = (
            sub["exposure_hr"].to_numpy()
            + (arm_index - 0.5) * width
        )

        ax.bar(
            x,
            sub["n_bins_r_6_8"],
            width=width,
            alpha=0.75,
            label=f"{arm}: 6-8 arcsec",
        )

    ax.set_xlabel("Exposure time [hr]")
    ax.set_ylabel(
        "Number of bin centroids at 6 <= R < 8 arcsec"
    )
    ax.set_title("Outer-disk spatial sampling")
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / "outer_bin_count_vs_exposure.png",
        dpi=180,
    )

    plt.close(fig)

    # Two-sigma-region counts.
    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    for arm, sub in summary_df.groupby("arm"):
        sub = sub.sort_values("exposure_hr")

        ax.plot(
            sub["exposure_hr"],
            sub["n_bins_two_sigma_region"],
            marker="o",
            label=arm,
        )

    ax.set_xlabel("Exposure time [hr]")
    ax.set_ylabel(
        r"Bins near expected $2\sigma$ regions"
    )
    ax.set_title(
        "Sampling near |u| ~ {:.1f} arcsec, |v| <= {:.1f} arcsec".format(
            TWO_SIGMA_RADIUS_ARCSEC,
            TWO_SIGMA_MINOR_HALF_WIDTH_ARCSEC,
        )
    )
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / "two_sigma_region_bins_vs_exposure.png",
        dpi=180,
    )

    plt.close(fig)

# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 78)
    print("{} KCWI/KCRM ADAPTIVE-BIN PLANNING".format(TARGET_NAME))
    print("=" * 78)

    print("\nRoot directory:")
    print(CRD_ROOT)

    print("\nMAPS file:")
    print(MAPS_PATH)

    print("\nBinning backend:")
    print(BINNING_BACKEND)

    print(
        f"\nPA_kin = {PA_KIN_DEG:.1f} deg east of north"
    )

    print(
        "\n{} slicer FOV = {:.1f} x {:.1f} arcsec".format(
            SLICER, KCWI_WIDTH_ARCSEC, KCWI_LENGTH_ARCSEC
        )
    )
    print(
        "Native slicer sampling = {:.3f} x {:.3f} arcsec".format(
            SLICE_WIDTH_ARCSEC, ALONG_SLICE_PIXEL_ARCSEC
        )
    )
    print(
        "Oriented u x v footprint = {:.1f} x {:.1f} arcsec".format(
            IFU_U_EXTENT_ARCSEC, IFU_V_EXTENT_ARCSEC
        )
    )
    print("Axis aligned with PA_kin: {}".format(IFU_AXIS_ALONG_PA))
    print(
        "Approx. two-sigma radius = {:.2f} arcsec".format(
            TWO_SIGMA_RADIUS_ARCSEC
        )
    )

    print("\n" + "=" * 78)
    print("S/N CALIBRATION VS BINNING TARGET")
    print("=" * 78)

    for arm in ["BL", "RH3"]:
        cfg = ARM_CALIBRATION[arm]

        print(f"\n{arm}:")
        print(
            "  Exposure calibration: "
            f"S/N={cfg['sn_ref']:.1f} "
            f"at mu={cfg['mu_ref']:.2f} "
            f"in {cfg['t_ref_hr']:.2f} hr"
        )
        print(
            "  Voronoi target:       "
            f"S/N={TARGET_SN[arm]:.1f}"
        )

    print()

    # Load MaNGA spatial maps.
    x_manga, y_manga, mflux_flam = load_manga_maps(
        MAPS_PATH
    )

    # Find DRP cube.
    drp_cube = locate_drp_cube(
        CRD_ROOT
    )

    if drp_cube is not None:
        print("Found DRP cube:")
        print(drp_cube)

        blue_flam, blue_lambda = synthesize_window_map_from_drp_cube(
            drp_cube,
            x_manga,
            y_manga,
            BLUE_REST_WINDOW,
        )

        red_flam, red_lambda = synthesize_window_map_from_drp_cube(
            drp_cube,
            x_manga,
            y_manga,
            RH3_REST_WINDOW,
        )

        source_label_blue = (
            "DRP cube median continuum, "
            f"rest {BLUE_REST_WINDOW[0]:.0f}-"
            f"{BLUE_REST_WINDOW[1]:.0f} A"
        )

        source_label_red = (
            "DRP cube median continuum, "
            f"rest {RH3_REST_WINDOW[0]:.0f}-"
            f"{RH3_REST_WINDOW[1]:.0f} A"
        )

    else:
        print("\nWARNING:")
        print("No DRP LOGCUBE/LINCUBE found.")
        print("Using SPX_MFLUX fallback.")

        blue_flam = mflux_flam.copy()
        blue_lambda = 4770.0

        red_flam = None
        red_lambda = 8700.0

        source_label_blue = "DAP SPX_MFLUX fallback"
        source_label_red = (
            "DAP SPX_MFLUX + global color fallback"
        )

    # Build rotated KCWI grid.
    uu, vv, x_kcwi, y_kcwi, pixel_area = build_kcwi_grid()

    print(
        "\nKCWI grid shape:",
        uu.shape,
    )

    print(
        f"KCWI model pixel area = "
        f"{pixel_area:.5f} arcsec^2"
    )

    # BL surface brightness.
    blue_interp = interpolate_manga_to_kcwi(
        x_manga,
        y_manga,
        blue_flam,
        x_kcwi,
        y_kcwi,
    )

    blue_mu = flam_to_mu_ab(
        blue_interp,
        blue_lambda,
    )

    # RH3 surface brightness.
    if red_flam is not None:
        red_interp = interpolate_manga_to_kcwi(
            x_manga,
            y_manga,
            red_flam,
            x_kcwi,
            y_kcwi,
        )

        red_mu = flam_to_mu_ab(
            red_interp,
            red_lambda,
        )

    else:
        red_mu = (
            blue_mu
            - FALLBACK_G_MINUS_RH3_MAG
        )

    # Save surface-brightness plots.
    plot_surface_brightness(
        uu,
        vv,
        blue_mu,
        "BL",
        source_label_blue,
    )

    plot_surface_brightness(
        uu,
        vv,
        red_mu,
        "RH3",
        source_label_red,
    )

    np.savez_compressed(
        OUTPUT_DIR
        / "kcwi_surface_brightness_templates.npz",
        u_arcsec=uu,
        v_arcsec=vv,
        x_sky_arcsec=x_kcwi,
        y_sky_arcsec=y_kcwi,
        mu_BL=blue_mu,
        mu_RH3=red_mu,
        pixel_area_arcsec2=pixel_area,
        slicer=np.array(SLICER),
        target_name=np.array(TARGET_NAME),
        plateifu=np.array(PLATEIFU),
    )

    all_summaries = []
    all_bin_tables = []

    for arm, mu_map in [
        ("BL", blue_mu),
        ("RH3", red_mu),
    ]:

        print("\n" + "-" * 78)
        print("ARM:", arm)

        print(
            "Calibration: "
            f"S/N={ARM_CALIBRATION[arm]['sn_ref']:.1f} "
            f"at {ARM_CALIBRATION[arm]['t_ref_hr']:.2f} hr"
        )

        print(
            "Binning target: "
            f"S/N={TARGET_SN[arm]:.1f}"
        )

        for t_hr in EXPOSURE_HOURS:

            (
                summary,
                bins,
                bin_map,
                valid,
                signal_map,
                noise_map,
            ) = analyze_one_case(
                arm,
                t_hr,
                uu,
                vv,
                x_kcwi,
                y_kcwi,
                mu_map,
                pixel_area,
            )

            all_summaries.append(summary)
            all_bin_tables.append(bins)

            plot_bin_map(
                uu,
                vv,
                mu_map,
                bin_map,
                bins,
                arm,
                t_hr,
            )

            np.savez_compressed(
                OUTPUT_DIR
                / f"{arm}_{int(t_hr)}hr_binning.npz",
                bin_map=bin_map,
                valid=valid,
                u_arcsec=uu,
                v_arcsec=vv,
                x_sky_arcsec=x_kcwi,
                y_sky_arcsec=y_kcwi,
                mu_map=mu_map,
                signal_map=signal_map,
                noise_map=noise_map,
            )

            print(
                f"  {t_hr:.0f} hr: "
                f"Nbin={summary['n_bins']:4d}, "
                f"median area="
                f"{summary['median_bin_area_arcsec2']:.3f} arcsec^2, "
                f"R=6-8\"={summary['n_bins_r_6_8']:3d}, "
                f"R>=8\"={summary['n_bins_r_ge_8']:3d}, "
                f"2sigma region="
                f"{summary['n_bins_two_sigma_region']:3d}"
            )

    # Save CSV tables.
    summary_df = pd.DataFrame(
        all_summaries
    )

    bins_df = pd.concat(
        all_bin_tables,
        ignore_index=True,
    )

    summary_df.to_csv(
        OUTPUT_DIR
        / "voronoi_planning_summary.csv",
        index=False,
    )

    bins_df.to_csv(
        OUTPUT_DIR
        / "voronoi_planning_all_bins.csv",
        index=False,
    )

    # Summary figures.
    plot_summary(
        summary_df
    )

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)

    useful_cols = [
        "arm",
        "exposure_hr",
        "sn_ref",
        "target_sn",
        "n_bins",
        "median_bin_area_arcsec2",
        "median_equiv_diameter_arcsec",
        "n_bins_r_4_6",
        "n_bins_r_6_8",
        "n_bins_r_ge_8",
        "n_bins_two_sigma_region",
    ]

    print(
        summary_df[
            useful_cols
        ].to_string(
            index=False
        )
    )

    print("\nOutputs written to:")
    print(OUTPUT_DIR)

    print("\nKey files:")
    print("  voronoi_planning_summary.csv")
    print("  voronoi_planning_all_bins.csv")
    print("  bin_count_vs_exposure.png")
    print("  median_bin_area_vs_exposure.png")
    print("  outer_bin_count_vs_exposure.png")
    print("  two_sigma_region_bins_vs_exposure.png")
    print("  BL_bins_2hr.png ... BL_bins_6hr.png")
    print("  RH3_bins_2hr.png ... RH3_bins_6hr.png")


if __name__ == "__main__":
    main()
