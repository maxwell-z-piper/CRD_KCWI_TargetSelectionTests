#!/usr/bin/env python
# coding: utf-8

"""
BLUE POPULATION-RECOVERY SIMULATION FOR A CONFIGURABLE CRD TARGET

Question
--------
If RH3 has already determined the two stellar LOSVDs, how much BL S/N is
required to recover:

    1. the BLUE light fraction f_A,
    2. the stellar age of Disk A,
    3. the stellar age of Disk B,
    4. the metallicity of Disk A,
    5. the metallicity of Disk B,
    6. which component is younger / more metal rich?

The kinematics are FIXED to the known RH3 solution during the BL pPXF fit.

This is intentionally different from the earlier injection/recovery test:
we are no longer asking BL to discover the two LOSVDs.

Outputs
-------
population_recovery_trials.csv
population_recovery_summary.csv

and several diagnostic PNG figures.
"""


# =============================================================================
# IMPORTS
# =============================================================================

from pathlib import Path
import inspect
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter1d, shift

from ppxf.ppxf import ppxf
from ppxf import sps_util


# =============================================================================
# USER SETTINGS
# =============================================================================

# -----------------------------------------------------------------------------
# XSL SPS file
# -----------------------------------------------------------------------------
#
# Set this to the SAME XSL npz file you used in your earlier KCWI simulations.
#
# If spectra_xsl_9.0.npz is somewhere below CRD_Thesis, the script will try
# to locate it automatically.
# -----------------------------------------------------------------------------

CRD_ROOT = Path(
    "/Users/maxpiper/Desktop/CRD_Thesis"
)

SPS_FILE = None


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


# This population-recovery experiment deliberately retains its existing
# F_A_GRID below. TARGET_FRAC_A is loaded for shared metadata/consistency but
# does not replace that grid.

# =============================================================================
# BL INSTRUMENT MODEL
# =============================================================================
# Preserve the ORIGINAL Small-slicer model exactly. Medium/Large scale the
# spectral FWHM by the KCWI slit-width resolution ratio and adopt the normal
# 2x2 detector binning. This changes only the instrumental resolution/sampling;
# the population-recovery calculations below are otherwise unchanged.
BL_SLICER_CONFIGS = {
    "Small": {
        "R_nominal": 3600.0,
        "FWHM_GAL_A": 1.25,
        "VELSCALE": 36.0,
        "NPIX_PER_RESEL": 2.0,
        "detector_binning": 1,
    },
    "Medium": {
        "R_nominal": 1800.0,
        "FWHM_GAL_A": 2.50,
        "VELSCALE": 72.0,
        "NPIX_PER_RESEL": 2.0,
        "detector_binning": 2,
    },
    "Large": {
        "R_nominal": 900.0,
        "FWHM_GAL_A": 5.00,
        "VELSCALE": 72.0,
        "NPIX_PER_RESEL": 4.0,
        "detector_binning": 2,
    },
}

_slicer_lookup = {key.lower(): key for key in BL_SLICER_CONFIGS}
_slicer_key = _slicer_lookup.get(str(SLICER).strip().lower())
if _slicer_key is None:
    raise ValueError(
        "Unknown SLICER={!r}. Use 'Small', 'Medium', or 'Large'.".format(SLICER)
    )
SLICER = _slicer_key
_BL_CFG = BL_SLICER_CONFIGS[SLICER]
FWHM_GAL_A = float(_BL_CFG["FWHM_GAL_A"])
VELSCALE = float(_BL_CFG["VELSCALE"])
NPIX_PER_RESEL = float(_BL_CFG["NPIX_PER_RESEL"])


# =============================================================================
# WAVELENGTH REGIONS
# =============================================================================

# Load slightly wider than the fit range to avoid edge effects from LOSVD shifts.
LOAD_RANGE_A = (3800.0, 5600.0)

# Population-analysis range.
FIT_RANGE_A = (3900.0, 5500.0)

# Define f_A as the component-A fraction of the stellar light in this band.
# This is deliberately similar to the normalization region used before.
LIGHT_FRACTION_BAND_A = (5000.0, 5350.0)

# S/N quoted per resolution element will be normalized using this relatively
# red BL region.
SNR_REFERENCE_BAND_A = (4800.0, 5500.0)


# =============================================================================
# RH3-INFORMED KINEMATICS
# =============================================================================
# V_A, V_B, SIGMA_A, and SIGMA_B are read from target_config.txt above
# and are treated as KNOWN in this first population-recovery experiment.


# =============================================================================
# S/N GRID
# =============================================================================

SNR_RESEL_GRID = np.array(
    [15, 20, 25, 30, 35, 40],
    dtype=float,
)


# =============================================================================
# BLUE LIGHT-FRACTION GRID
# =============================================================================

# f_B = 1 - f_A automatically.

F_A_GRID = np.array(
    [0.20, 0.30, 0.50, 0.70, 0.80],
    dtype=float,
)


# =============================================================================
# MONTE CARLO SETTINGS
# =============================================================================

# 100 gives useful recovery fractions.
# For an initial debugging run, change this to 20.
N_MC = 100

RANDOM_SEED = 8675309


# =============================================================================
# POPULATION CASES
# =============================================================================
#
# pPXF will automatically select the nearest actual XSL SSP to these
# requested values.
#
# I recommend testing more than one population contrast because the ability
# to separate populations depends strongly on how different they actually are.
#
# Ages are in Gyr.
# Metallicities are [M/H] in dex.
# =============================================================================

POPULATION_CASES = {

    # Strong age difference, similar metallicity.
    "strong_age_contrast": {
        "age_A": 10.0,
        "metal_A": -0.20,

        "age_B": 3.0,
        "metal_B": -0.20,
    },

    # Age + metallicity difference.
    "age_metal_contrast": {
        "age_A": 10.0,
        "metal_A": 0.00,

        "age_B": 3.0,
        "metal_B": -0.50,
    },

    # More difficult / subtle case.
    "subtle_contrast": {
        "age_A": 8.0,
        "metal_A": -0.20,

        "age_B": 5.0,
        "metal_B": 0.00,
    },

}


# =============================================================================
# SUCCESS CRITERIA
# =============================================================================
#
# These are NOT fundamental thresholds.
#
# They simply allow us to make "recovery fraction" plots.
#
# The CSV also saves the continuous errors so we are not relying only on
# binary pass/fail definitions.
# =============================================================================

FRACTION_TOL = 0.10

LOGAGE_TOL_DEX = 0.15

METAL_TOL_DEX = 0.15


# =============================================================================
# pPXF POPULATION FIT SETTINGS
# =============================================================================
#
# For this first information-content experiment:
#
#   - no additive polynomial
#   - no multiplicative polynomial
#   - no emission lines
#   - exact instrumental LSF
#   - white Gaussian noise
#
# That isolates the population information contained in the spectrum.
#
# Later we should make this more realistic by adding:
#
#   - multiplicative continuum freedom
#   - wavelength-dependent S/N
#   - gas emission
#   - template mismatch
#   - RH3 kinematic uncertainties
# =============================================================================

DEGREE = -1
MDEGREE = 0


# =============================================================================
# OUTPUT DIRECTORY
# =============================================================================

TARGET_TAG = "".join(c if (c.isalnum() or c in "-_") else "_" for c in TARGET_NAME)
OUTPUT_DIR = (
    CRD_ROOT
    / ("KCWI_Blue_Population_Recovery_{}_{}".format(TARGET_TAG, SLICER))
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# HELPER: LOCATE XSL FILE
# =============================================================================

def locate_xsl_file():

    if SPS_FILE is not None:

        path = Path(SPS_FILE)

        if not path.exists():
            raise FileNotFoundError(path)

        return path


    matches = list(
        CRD_ROOT.rglob(
            "spectra_xsl_9.0.npz"
        )
    )


    if len(matches) == 0:

        raise FileNotFoundError(
            "\nCould not locate spectra_xsl_9.0.npz below:\n"
            f"{CRD_ROOT}\n\n"
            "Set SPS_FILE manually at the top of this script."
        )


    if len(matches) > 1:

        print(
            "Multiple XSL files found. Using:\n",
            matches[0],
        )


    return matches[0]


# =============================================================================
# HELPER: LOAD XSL SPS LIBRARY
# =============================================================================

def load_sps_library():

    filename = locate_xsl_file()

    print(
        "Using XSL SPS file:\n",
        filename,
    )


    # pPXF renamed wave_range -> lam_range in newer versions.
    # Handle both automatically.

    signature = inspect.signature(
        sps_util.sps_lib
    )


    kwargs = {}


    if "lam_range" in signature.parameters:

        kwargs["lam_range"] = LOAD_RANGE_A


    elif "wave_range" in signature.parameters:

        kwargs["wave_range"] = LOAD_RANGE_A


    sps = sps_util.sps_lib(

        str(filename),

        VELSCALE,

        FWHM_GAL_A,

        **kwargs,
    )


    templates_nd = np.asarray(
        sps.templates,
        dtype=float,
    )


    # Pixel dimension is first.
    pop_shape = templates_nd.shape[1:]


    templates = templates_nd.reshape(
        templates_nd.shape[0],
        -1,
    )


    lam = np.exp(
        np.asarray(
            sps.ln_lam_temp
        )
    )


    age_grid = np.asarray(
        sps.age_grid,
        dtype=float,
    )


    metal_grid = np.asarray(
        sps.metal_grid,
        dtype=float,
    )


    # -------------------------------------------------------------------------
    # Flatten age/metal grids so one age and one metallicity correspond to
    # every template column.
    # -------------------------------------------------------------------------

    if (
        age_grid.shape == pop_shape
        and
        metal_grid.shape == pop_shape
    ):

        ages = age_grid.ravel()

        metals = metal_grid.ravel()


    elif (
        age_grid.ndim == 1
        and
        metal_grid.ndim == 1
        and
        len(pop_shape) >= 2
        and
        age_grid.size == pop_shape[0]
        and
        metal_grid.size == pop_shape[1]
    ):

        aa, zz = np.meshgrid(
            age_grid,
            metal_grid,
            indexing="ij",
        )

        ages = aa.ravel()

        metals = zz.ravel()


    else:

        try:

            ages = np.broadcast_to(
                age_grid,
                pop_shape,
            ).ravel()

            metals = np.broadcast_to(
                metal_grid,
                pop_shape,
            ).ravel()


        except Exception as exc:

            raise RuntimeError(
                "Could not map XSL age_grid / metal_grid onto "
                f"template shape {pop_shape}.\n"
                f"age_grid shape = {age_grid.shape}\n"
                f"metal_grid shape = {metal_grid.shape}"
            ) from exc


    if (
        templates.shape[1]
        !=
        ages.size
    ):

        raise RuntimeError(
            "Template metadata mismatch:\n"
            f"N templates = {templates.shape[1]}\n"
            f"N ages      = {ages.size}"
        )


    # -------------------------------------------------------------------------
    # IMPORTANT:
    #
    # For population fitting, do NOT independently normalize every SSP.
    #
    # Scale the entire library by ONE scalar so the relative spectral
    # normalization of different populations is preserved.
    # -------------------------------------------------------------------------

    fit_mask = (
        (lam >= FIT_RANGE_A[0])
        &
        (lam <= FIT_RANGE_A[1])
    )


    scalar = np.nanmedian(
        templates[
            fit_mask,
            :
        ]
    )


    templates = templates / scalar


    return (
        templates,
        lam,
        ages,
        metals,
    )


# =============================================================================
# HELPER: FIND NEAREST SSP
# =============================================================================

def nearest_ssp_index(
        ages,
        metals,
        age_target,
        metal_target):

    """
    Select nearest SSP in log(age) and metallicity.
    """

    distance = (

        (
            np.log10(ages)
            -
            np.log10(age_target)
        ) ** 2

        +

        (
            metals
            -
            metal_target
        ) ** 2

    )


    return int(
        np.nanargmin(
            distance
        )
    )


# =============================================================================
# HELPER: APPLY LOSVD TO A LOG-LAMBDA TEMPLATE
# =============================================================================

def apply_losvd(
        spectrum,
        velocity,
        sigma):

    """
    Approximate a Gaussian LOSVD on the logarithmic wavelength grid.

    velocity and sigma are km/s.
    """

    spec = np.asarray(
        spectrum,
        dtype=float,
    )


    sigma_pix = (
        sigma
        /
        VELSCALE
    )


    velocity_pix = (
        velocity
        /
        VELSCALE
    )


    broadened = gaussian_filter1d(

        spec,

        sigma=sigma_pix,

        mode="nearest",

    )


    shifted = shift(

        broadened,

        shift=velocity_pix,

        order=3,

        mode="nearest",

        prefilter=True,

    )


    return shifted


# =============================================================================
# HELPER: APPLY LOSVD TO ALL TEMPLATES
# =============================================================================

def transform_template_library(
        templates,
        velocity,
        sigma):

    """
    Precompute transformed template library for reconstructing each component.
    """

    sigma_pix = (
        sigma
        /
        VELSCALE
    )


    velocity_pix = (
        velocity
        /
        VELSCALE
    )


    out = gaussian_filter1d(

        templates,

        sigma=sigma_pix,

        axis=0,

        mode="nearest",

    )


    out = shift(

        out,

        shift=(velocity_pix, 0),

        order=3,

        mode="nearest",

        prefilter=True,

    )


    return out


# =============================================================================
# HELPER: NORMALIZE COMPONENT IN BLUE LIGHT-FRACTION BAND
# =============================================================================

def normalize_component(
        spectrum,
        lam):

    mask = (

        (lam >= LIGHT_FRACTION_BAND_A[0])

        &

        (lam <= LIGHT_FRACTION_BAND_A[1])

    )


    norm = np.nanmean(
        spectrum[
            mask
        ]
    )


    return (
        spectrum
        /
        norm
    )


# =============================================================================
# CREATE NOISELESS TWO-COMPONENT GALAXY
# =============================================================================

def make_truth_spectrum(
        templates,
        lam,
        index_A,
        index_B,
        frac_A):


    A = apply_losvd(

        templates[
            :,
            index_A
        ],

        V_A,

        SIGMA_A,

    )


    B = apply_losvd(

        templates[
            :,
            index_B
        ],

        V_B,

        SIGMA_B,

    )


    # -------------------------------------------------------------------------
    # Normalize the TWO injected component spectra in the SAME blue band.
    #
    # This guarantees that frac_A really means:
    #
    #        blue light from A
    #     ------------------------
    #     total blue stellar light
    #
    # in LIGHT_FRACTION_BAND_A.
    # -------------------------------------------------------------------------

    A = normalize_component(
        A,
        lam,
    )


    B = normalize_component(
        B,
        lam,
    )


    galaxy = (

        frac_A
        *
        A

        +

        (
            1.0
            -
            frac_A
        )
        *
        B

    )


    return galaxy


# =============================================================================
# ADD NOISE
# =============================================================================

def add_noise(
        noiseless,
        lam,
        snr_resel,
        rng):

    """
    Define the requested S/N per BL RESOLUTION ELEMENT using the
    4800-5500 A reference region.

    Approx:
        S/N_pixel = S/N_resel / sqrt(2)
    """

    snr_pixel = (

        float(
            snr_resel
        )

        /

        np.sqrt(
            NPIX_PER_RESEL
        )

    )


    ref = (

        (lam >= SNR_REFERENCE_BAND_A[0])

        &

        (lam <= SNR_REFERENCE_BAND_A[1])

    )


    reference_flux = np.nanmedian(

        noiseless[
            ref
        ]

    )


    noise_sigma = (

        reference_flux

        /

        snr_pixel

    )


    noise = np.full_like(

        noiseless,

        noise_sigma,

        dtype=float,

    )


    galaxy = (

        noiseless

        +

        rng.normal(

            loc=0.0,

            scale=noise_sigma,

            size=noiseless.size,

        )

    )


    return (
        galaxy,
        noise,
        snr_pixel,
    )


# =============================================================================
# RECOVER f_A + POPULATIONS FROM pPXF WEIGHTS
# =============================================================================

def recover_component_properties(
        pp,
        transformed_A,
        transformed_B,
        lam,
        ages,
        metals):


    n_templates = len(
        ages
    )


    weights = np.asarray(
        pp.weights,
        dtype=float,
    )


    weights_A = weights[
        :n_templates
    ].copy()


    weights_B = weights[
        n_templates:
        2 * n_templates
    ].copy()


    # Numerical tiny negative weights, if present, are not physically useful
    # for population summaries.

    weights_A = np.clip(
        weights_A,
        0,
        None,
    )

    weights_B = np.clip(
        weights_B,
        0,
        None,
    )


    # -------------------------------------------------------------------------
    # Reconstruct the model contribution from EACH stellar disk.
    # -------------------------------------------------------------------------

    model_A = (

        transformed_A

        @

        weights_A

    )


    model_B = (

        transformed_B

        @

        weights_B

    )


    band = (

        (lam >= LIGHT_FRACTION_BAND_A[0])

        &

        (lam <= LIGHT_FRACTION_BAND_A[1])

    )


    flux_A = np.trapz(

        model_A[
            band
        ],

        lam[
            band
        ],

    )


    flux_B = np.trapz(

        model_B[
            band
        ],

        lam[
            band
        ],

    )


    total_flux = (
        flux_A
        +
        flux_B
    )


    if (
        total_flux
        <=
        0
    ):

        frac_A_rec = np.nan

    else:

        frac_A_rec = (
            flux_A
            /
            total_flux
        )


    # -------------------------------------------------------------------------
    # Light-weight individual SSP contributions in the SAME reference band.
    #
    # This is preferable to simply averaging age by raw pPXF weight.
    # -------------------------------------------------------------------------

    template_flux_A = np.trapz(

        transformed_A[
            band,
            :
        ],

        lam[
            band
        ],

        axis=0,

    )


    template_flux_B = np.trapz(

        transformed_B[
            band,
            :
        ],

        lam[
            band
        ],

        axis=0,

    )


    light_weights_A = (

        weights_A

        *

        template_flux_A

    )


    light_weights_B = (

        weights_B

        *

        template_flux_B

    )


    def population_summary(
            light_weights):

        light_weights = np.clip(
            light_weights,
            0,
            None,
        )


        total = np.sum(
            light_weights
        )


        if (
            total
            <=
            0
        ):

            return (
                np.nan,
                np.nan,
            )


        w = (
            light_weights
            /
            total
        )


        # Logarithmic light-weighted age is more useful than linear age
        # when populations span many Gyr.

        log_age = np.sum(

            w

            *

            np.log10(
                ages
            )

        )


        metallicity = np.sum(

            w

            *

            metals

        )


        return (
            log_age,
            metallicity,
        )


    log_age_A, metal_A = population_summary(
        light_weights_A
    )


    log_age_B, metal_B = population_summary(
        light_weights_B
    )


    return {

        "f_A_rec":
            frac_A_rec,

        "logage_A_rec":
            log_age_A,

        "age_A_rec_Gyr":
            10 ** log_age_A
            if np.isfinite(log_age_A)
            else np.nan,

        "metal_A_rec":
            metal_A,

        "logage_B_rec":
            log_age_B,

        "age_B_rec_Gyr":
            10 ** log_age_B
            if np.isfinite(log_age_B)
            else np.nan,

        "metal_B_rec":
            metal_B,

    }


# =============================================================================
# FIT ONE NOISY REALIZATION
# =============================================================================

def fit_one_realization(
        galaxy,
        noise,
        templates,
        lam,
        ages,
        metals,
        transformed_A,
        transformed_B):


    n_templates = templates.shape[1]


    # Duplicate the full SSP library:
    #
    # first copy  -> Disk A
    # second copy -> Disk B

    fit_templates = np.column_stack(

        [
            templates,
            templates,
        ]

    )


    component = np.concatenate(

        [
            np.zeros(
                n_templates,
                dtype=int,
            ),

            np.ones(
                n_templates,
                dtype=int,
            ),
        ]

    )


    goodpixels = np.where(

        (lam >= FIT_RANGE_A[0])

        &

        (lam <= FIT_RANGE_A[1])

    )[0]


    # -------------------------------------------------------------------------
    # KEY PART OF THIS EXPERIMENT:
    #
    # moments = [-2, -2]
    #
    # tells pPXF to KEEP BOTH stellar LOSVDs FIXED.
    #
    # So the fit does NOT search for V or sigma.
    # It only determines the stellar-template contributions.
    # -------------------------------------------------------------------------

    pp = ppxf(

        fit_templates,

        galaxy,

        noise,

        VELSCALE,

        start=[
            [V_A, SIGMA_A],
            [V_B, SIGMA_B],
        ],

        moments=[
            -2,
            -2,
        ],

        component=component,

        degree=DEGREE,

        mdegree=MDEGREE,

        goodpixels=goodpixels,

        lam=lam,

        lam_temp=lam,

        quiet=True,

        linear_method="lsq_box",

    )


    result = recover_component_properties(

        pp,

        transformed_A,

        transformed_B,

        lam,

        ages,

        metals,

    )


    result["chi2"] = float(
        pp.chi2
    )


    return result


# =============================================================================
# RUN FULL MONTE CARLO GRID
# =============================================================================

def run_simulation():


    rng = np.random.default_rng(
        RANDOM_SEED
    )


    (
        templates,
        lam,
        ages,
        metals,

    ) = load_sps_library()


    print(
        "\nXSL library:"
    )

    print(
        "  wavelength range = "
        f"{lam.min():.1f}-{lam.max():.1f} A"
    )

    print(
        "  number templates = ",
        templates.shape[1],
    )

    print(
        "  age range = "
        f"{ages.min():.3f}-{ages.max():.3f} Gyr"
    )

    print(
        "  metallicity range = "
        f"{metals.min():.2f}-{metals.max():.2f}"
    )


    # -------------------------------------------------------------------------
    # Precompute all template spectra with the two KNOWN LOSVDs.
    #
    # This is used only for reconstructing component spectra/properties.
    # -------------------------------------------------------------------------

    transformed_A = transform_template_library(

        templates,

        V_A,

        SIGMA_A,

    )


    transformed_B = transform_template_library(

        templates,

        V_B,

        SIGMA_B,

    )


    rows = []


    for case_name, case in POPULATION_CASES.items():


        idx_A = nearest_ssp_index(

            ages,

            metals,

            case["age_A"],

            case["metal_A"],

        )


        idx_B = nearest_ssp_index(

            ages,

            metals,

            case["age_B"],

            case["metal_B"],

        )


        age_A_true = float(
            ages[
                idx_A
            ]
        )


        metal_A_true = float(
            metals[
                idx_A
            ]
        )


        age_B_true = float(
            ages[
                idx_B
            ]
        )


        metal_B_true = float(
            metals[
                idx_B
            ]
        )


        print(
            "\n"
            +
            "=" * 72
        )

        print(
            case_name
        )

        print(
            "=" * 72
        )

        print(
            "Requested A: "
            f"age={case['age_A']} Gyr, "
            f"[M/H]={case['metal_A']:+.2f}"
        )

        print(
            "Actual XSL A: "
            f"age={age_A_true:.3f} Gyr, "
            f"[M/H]={metal_A_true:+.2f}"
        )

        print(
            "Requested B: "
            f"age={case['age_B']} Gyr, "
            f"[M/H]={case['metal_B']:+.2f}"
        )

        print(
            "Actual XSL B: "
            f"age={age_B_true:.3f} Gyr, "
            f"[M/H]={metal_B_true:+.2f}"
        )


        logage_A_true = np.log10(
            age_A_true
        )


        logage_B_true = np.log10(
            age_B_true
        )


        true_delta_logage = (
            logage_A_true
            -
            logage_B_true
        )


        true_delta_metal = (
            metal_A_true
            -
            metal_B_true
        )


        for frac_A_true in F_A_GRID:


            noiseless = make_truth_spectrum(

                templates,

                lam,

                idx_A,

                idx_B,

                frac_A_true,

            )


            for snr_resel in SNR_RESEL_GRID:


                print(
                    f"  f_A={frac_A_true:.2f}, "
                    f"S/N={snr_resel:.0f}"
                )


                for mc in range(
                    N_MC
                ):


                    (
                        galaxy,
                        noise,
                        snr_pixel,

                    ) = add_noise(

                        noiseless,

                        lam,

                        snr_resel,

                        rng,

                    )


                    result = fit_one_realization(

                        galaxy,

                        noise,

                        templates,

                        lam,

                        ages,

                        metals,

                        transformed_A,

                        transformed_B,

                    )


                    # =========================================================
                    # ERRORS
                    # =========================================================

                    d_frac = (

                        result[
                            "f_A_rec"
                        ]

                        -
                        frac_A_true

                    )


                    d_logage_A = (

                        result[
                            "logage_A_rec"
                        ]

                        -
                        logage_A_true

                    )


                    d_logage_B = (

                        result[
                            "logage_B_rec"
                        ]

                        -
                        logage_B_true

                    )


                    d_metal_A = (

                        result[
                            "metal_A_rec"
                        ]

                        -
                        metal_A_true

                    )


                    d_metal_B = (

                        result[
                            "metal_B_rec"
                        ]

                        -
                        metal_B_true

                    )


                    delta_logage_rec = (

                        result[
                            "logage_A_rec"
                        ]

                        -
                        result[
                            "logage_B_rec"
                        ]

                    )


                    delta_metal_rec = (

                        result[
                            "metal_A_rec"
                        ]

                        -
                        result[
                            "metal_B_rec"
                        ]

                    )


                    # =========================================================
                    # PASS / FAIL FLAGS
                    # =========================================================

                    fraction_pass = (

                        abs(
                            d_frac
                        )

                        <=

                        FRACTION_TOL

                    )


                    age_A_pass = (

                        abs(
                            d_logage_A
                        )

                        <=

                        LOGAGE_TOL_DEX

                    )


                    age_B_pass = (

                        abs(
                            d_logage_B
                        )

                        <=

                        LOGAGE_TOL_DEX

                    )


                    metal_A_pass = (

                        abs(
                            d_metal_A
                        )

                        <=

                        METAL_TOL_DEX

                    )


                    metal_B_pass = (

                        abs(
                            d_metal_B
                        )

                        <=

                        METAL_TOL_DEX

                    )


                    # Was the relative ordering recovered?
                    #
                    # e.g. did the code correctly infer that A is older than B?

                    age_order_correct = (

                        np.sign(
                            delta_logage_rec
                        )

                        ==

                        np.sign(
                            true_delta_logage
                        )

                    )


                    # Only meaningful if the true metallicities differ.

                    if (
                        abs(
                            true_delta_metal
                        )
                        >
                        1e-6
                    ):

                        metal_order_correct = (

                            np.sign(
                                delta_metal_rec
                            )

                            ==

                            np.sign(
                                true_delta_metal
                            )

                        )

                    else:

                        metal_order_correct = np.nan


                    joint_population_pass = (

                        fraction_pass

                        and

                        age_A_pass

                        and

                        age_B_pass

                        and

                        metal_A_pass

                        and

                        metal_B_pass

                    )


                    rows.append({

                        "population_case":
                            case_name,

                        "mc":
                            mc,

                        "snr_resel":
                            float(
                                snr_resel
                            ),

                        "snr_pixel":
                            float(
                                snr_pixel
                            ),

                        "f_A_true":
                            float(
                                frac_A_true
                            ),

                        "f_A_rec":
                            result[
                                "f_A_rec"
                            ],

                        "d_f_A":
                            d_frac,

                        "abs_d_f_A":
                            abs(
                                d_frac
                            ),


                        "age_A_true_Gyr":
                            age_A_true,

                        "age_A_rec_Gyr":
                            result[
                                "age_A_rec_Gyr"
                            ],

                        "d_logage_A":
                            d_logage_A,

                        "abs_d_logage_A":
                            abs(
                                d_logage_A
                            ),


                        "age_B_true_Gyr":
                            age_B_true,

                        "age_B_rec_Gyr":
                            result[
                                "age_B_rec_Gyr"
                            ],

                        "d_logage_B":
                            d_logage_B,

                        "abs_d_logage_B":
                            abs(
                                d_logage_B
                            ),


                        "metal_A_true":
                            metal_A_true,

                        "metal_A_rec":
                            result[
                                "metal_A_rec"
                            ],

                        "d_metal_A":
                            d_metal_A,

                        "abs_d_metal_A":
                            abs(
                                d_metal_A
                            ),


                        "metal_B_true":
                            metal_B_true,

                        "metal_B_rec":
                            result[
                                "metal_B_rec"
                            ],

                        "d_metal_B":
                            d_metal_B,

                        "abs_d_metal_B":
                            abs(
                                d_metal_B
                            ),


                        "delta_logage_true":
                            true_delta_logage,

                        "delta_logage_rec":
                            delta_logage_rec,

                        "delta_metal_true":
                            true_delta_metal,

                        "delta_metal_rec":
                            delta_metal_rec,


                        "fraction_pass":
                            fraction_pass,

                        "age_A_pass":
                            age_A_pass,

                        "age_B_pass":
                            age_B_pass,

                        "metal_A_pass":
                            metal_A_pass,

                        "metal_B_pass":
                            metal_B_pass,

                        "age_order_correct":
                            age_order_correct,

                        "metal_order_correct":
                            metal_order_correct,

                        "joint_population_pass":
                            joint_population_pass,

                        "chi2":
                            result[
                                "chi2"
                            ],

                    })


    return pd.DataFrame(
        rows
    )


# =============================================================================
# SUMMARIZE MONTE CARLO RESULTS
# =============================================================================

def summarize_results(
        trials):


    group_cols = [

        "population_case",

        "snr_resel",

        "f_A_true",

    ]


    summary_rows = []


    for key, sub in trials.groupby(
        group_cols
    ):


        (
            population_case,
            snr_resel,
            f_A_true,

        ) = key


        row = {

            "population_case":
                population_case,

            "snr_resel":
                snr_resel,

            "f_A_true":
                f_A_true,

            "n":
                len(
                    sub
                ),


            # -------------------------------------------------------------
            # f_A
            # -------------------------------------------------------------

            "median_f_A_rec":
                np.nanmedian(
                    sub["f_A_rec"]
                ),

            "median_abs_df_A":
                np.nanmedian(
                    sub["abs_d_f_A"]
                ),

            "p84_abs_df_A":
                np.nanpercentile(
                    sub["abs_d_f_A"],
                    84,
                ),

            "fraction_recovery_fraction":
                np.nanmean(
                    sub["fraction_pass"]
                ),


            # -------------------------------------------------------------
            # AGE
            # -------------------------------------------------------------

            "median_abs_dlogage_A":
                np.nanmedian(
                    sub["abs_d_logage_A"]
                ),

            "median_abs_dlogage_B":
                np.nanmedian(
                    sub["abs_d_logage_B"]
                ),

            "p84_abs_dlogage_A":
                np.nanpercentile(
                    sub["abs_d_logage_A"],
                    84,
                ),

            "p84_abs_dlogage_B":
                np.nanpercentile(
                    sub["abs_d_logage_B"],
                    84,
                ),

            "age_A_recovery_fraction":
                np.nanmean(
                    sub["age_A_pass"]
                ),

            "age_B_recovery_fraction":
                np.nanmean(
                    sub["age_B_pass"]
                ),

            "age_order_fraction":
                np.nanmean(
                    sub["age_order_correct"]
                ),


            # -------------------------------------------------------------
            # METALLICITY
            # -------------------------------------------------------------

            "median_abs_dmetal_A":
                np.nanmedian(
                    sub["abs_d_metal_A"]
                ),

            "median_abs_dmetal_B":
                np.nanmedian(
                    sub["abs_d_metal_B"]
                ),

            "p84_abs_dmetal_A":
                np.nanpercentile(
                    sub["abs_d_metal_A"],
                    84,
                ),

            "p84_abs_dmetal_B":
                np.nanpercentile(
                    sub["abs_d_metal_B"],
                    84,
                ),

            "metal_A_recovery_fraction":
                np.nanmean(
                    sub["metal_A_pass"]
                ),

            "metal_B_recovery_fraction":
                np.nanmean(
                    sub["metal_B_pass"]
                ),

            "metal_order_fraction":
                np.nanmean(
                    sub["metal_order_correct"]
                ),


            # -------------------------------------------------------------
            # STRICT JOINT SUCCESS
            # -------------------------------------------------------------

            "joint_population_recovery_fraction":
                np.nanmean(
                    sub["joint_population_pass"]
                ),

        }


        summary_rows.append(
            row
        )


    return pd.DataFrame(
        summary_rows
    )


# =============================================================================
# PLOTS
# =============================================================================

def make_plots(
        summary):


    for case in summary[
        "population_case"
    ].unique():


        ss = summary[
            summary["population_case"]
            ==
            case
        ]


        # =====================================================================
        # JOINT SUCCESS
        # =====================================================================

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )


        for f_true in F_A_GRID:

            s = ss[
                ss["f_A_true"]
                ==
                f_true
            ].sort_values(
                "snr_resel"
            )


            ax.plot(

                s["snr_resel"],

                s[
                    "joint_population_recovery_fraction"
                ],

                marker="o",

                label=f"f_A={f_true:.1f}",

            )


        ax.axhline(
            0.90,
            ls="--",
        )


        ax.set_ylim(
            0,
            1.03,
        )


        ax.set_xlabel(
            "S/N per BL resolution element"
        )


        ax.set_ylabel(
            "Joint population-recovery fraction"
        )


        ax.set_title(
            case
        )


        ax.legend()


        fig.tight_layout()


        fig.savefig(

            OUTPUT_DIR
            /
            f"{case}_joint_recovery.png",

            dpi=180,

        )


        plt.close(
            fig
        )


        # =====================================================================
        # f_A ERROR
        # =====================================================================

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )


        for f_true in F_A_GRID:

            s = ss[
                ss["f_A_true"]
                ==
                f_true
            ].sort_values(
                "snr_resel"
            )


            ax.plot(

                s["snr_resel"],

                s[
                    "median_abs_df_A"
                ],

                marker="o",

                label=f"f_A={f_true:.1f}",

            )


        ax.axhline(
            FRACTION_TOL,
            ls="--",
        )


        ax.set_xlabel(
            "S/N per BL resolution element"
        )


        ax.set_ylabel(
            r"Median $|f_{A,rec}-f_{A,true}|$"
        )


        ax.set_title(
            case + ": blue light-fraction recovery"
        )


        ax.legend()


        fig.tight_layout()


        fig.savefig(

            OUTPUT_DIR
            /
            f"{case}_fraction_error.png",

            dpi=180,

        )


        plt.close(
            fig
        )


        # =====================================================================
        # AGE ERROR
        # =====================================================================

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )


        # Average A/B median error only for visualization.

        for f_true in F_A_GRID:

            s = ss[
                ss["f_A_true"]
                ==
                f_true
            ].sort_values(
                "snr_resel"
            )


            mean_age_error = (

                s[
                    "median_abs_dlogage_A"
                ].to_numpy()

                +

                s[
                    "median_abs_dlogage_B"
                ].to_numpy()

            ) / 2.0


            ax.plot(

                s["snr_resel"],

                mean_age_error,

                marker="o",

                label=f"f_A={f_true:.1f}",

            )


        ax.axhline(
            LOGAGE_TOL_DEX,
            ls="--",
        )


        ax.set_xlabel(
            "S/N per BL resolution element"
        )


        ax.set_ylabel(
            "Mean median |Δ log age| [dex]"
        )


        ax.set_title(
            case + ": stellar-age recovery"
        )


        ax.legend()


        fig.tight_layout()


        fig.savefig(

            OUTPUT_DIR
            /
            f"{case}_age_error.png",

            dpi=180,

        )


        plt.close(
            fig
        )


        # =====================================================================
        # METALLICITY ERROR
        # =====================================================================

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )


        for f_true in F_A_GRID:

            s = ss[
                ss["f_A_true"]
                ==
                f_true
            ].sort_values(
                "snr_resel"
            )


            mean_metal_error = (

                s[
                    "median_abs_dmetal_A"
                ].to_numpy()

                +

                s[
                    "median_abs_dmetal_B"
                ].to_numpy()

            ) / 2.0


            ax.plot(

                s["snr_resel"],

                mean_metal_error,

                marker="o",

                label=f"f_A={f_true:.1f}",

            )


        ax.axhline(
            METAL_TOL_DEX,
            ls="--",)

        ax.set_xlabel(
            "S/N per BL resolution element")

        ax.set_ylabel(
            "Mean median |Δ[M/H]| [dex]")

        ax.set_title(
            case + ": metallicity recovery")


        ax.legend()
        
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR/f"{case}_metallicity_error.png",dpi=180,)
        plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main():


    print("=" * 78)
    print("{} BLUE POPULATION-RECOVERY SIMULATION".format(TARGET_NAME))
    print("=" * 78)

    print("\nInstrument / target configuration:")
    print("  Slicer: {}".format(SLICER))
    print("  Nominal BL R: {:.0f}".format(_BL_CFG["R_nominal"]))
    print("  BL FWHM used: {:.3f} A".format(FWHM_GAL_A))
    print("  Velscale: {:.1f} km/s/pixel".format(VELSCALE))
    print("  Pixels/resel: {:.2f}".format(NPIX_PER_RESEL))
    print("  Target redshift (metadata): {:.6f}".format(TARGET_REDSHIFT))
    print("\nRH3 kinematics assumed known:")
    print(f"  Disk A: V={V_A:+.1f}, sigma={SIGMA_A:.1f} km/s")
    print(f"  Disk B: V={V_B:+.1f}, sigma={SIGMA_B:.1f} km/s")
    print("  DeltaV: {:.1f} km/s".format(abs(V_A - V_B)))
    print("\nS/N grid:",SNR_RESEL_GRID,)
    print("f_A grid:",F_A_GRID,)
    print("N_MC:",N_MC,)

    trials = run_simulation()
    trials.to_csv(OUTPUT_DIR/"population_recovery_trials.csv", index=False,)
    
    summary = summarize_results(trials)
    summary.to_csv(OUTPUT_DIR/"population_recovery_summary.csv",index=False,)

    make_plots(summary)

    print("\n"+"=" * 78)
    print("SUMMARY")
    print("=" * 78)


    cols = [
        "population_case",
        "snr_resel",
        "f_A_true",
        "median_abs_df_A",
        "median_abs_dlogage_A",
        "median_abs_dlogage_B",
        "median_abs_dmetal_A",
        "median_abs_dmetal_B",
        "age_order_fraction",
        "joint_population_recovery_fraction",
    ]

    print(summary[cols].to_string(index=False))
    print("\nOutputs written to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
