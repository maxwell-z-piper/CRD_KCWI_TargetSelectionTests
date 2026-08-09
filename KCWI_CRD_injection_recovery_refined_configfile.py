#!/usr/bin/env python
# coding: utf-8

"""
KCWI BL TWO-COMPONENT CRD INJECTION / RECOVERY TEST
==========================================================

PURPOSE
-------
This script determines the signal-to-noise ratio (S/N) required to recover
both stellar components of a counter-rotating disk (CRD) with KCWI+BL.
It is designed specifically around the analysis strategy discussed for IC 25
and J164055:

1. Build synthetic spectra containing two stellar populations with known
   velocities, intrinsic dispersions, and light fractions.
2. Match the stellar templates to the selected KCWI-slicer + BL instrumental resolution.
3. Add Gaussian noise over a grid of requested S/N values.
4. For EVERY noisy realization, perform a Mitzkus-style brute-force search in
   (V_A, V_B): each grid cell is fitted independently, with pPXF constrained
   to remain inside that velocity cell. This avoids dependence on a single
   initial velocity guess.
5. Fit a one-component model to the same realization.
6. Run an independent set of true one-component simulations to calibrate how
   often noise/template freedom produces an apparently significant
   two-component solution (the false-positive/null experiment).
7. Run TWO template cases on the same noise realizations:
       - a sparse deliberately mismatched basis,
       - a matched control containing the exact injected SSP.
8. For the CRD injections, quantify THREE success levels:
       - secure two-LOSVD detection,
       - accurate relative separation DeltaV,
       - accurate absolute V_A and V_B.
9. Save CSV tables and plots showing recovery fraction versus S/N and diagnose
   common-mode velocity offsets separately from separation errors.

WHY THE NULL EXPERIMENT MATTERS
-------------------------------
A two-component model has more freedom than a one-component model. A raw
Delta-chi^2 improvement alone is therefore not enough to claim that two
components are detected. The code first simulates spectra that truly contain
ONLY ONE stellar LOSVD. It then runs the exact same two-component grid search.
For each S/N and wavelength window, the 95th percentile of the null
Delta-chi^2 distribution is used as an empirical detection threshold.

A CRD realization is counted as a successful recovery only when:
    (a) both recovered velocities are close to the injected velocities,
    (b) the recovered separation is sensible,
    (c) both components contribute non-negligible light, and
    (d) its Delta-chi^2 exceeds the empirical one-component null threshold.

S/N CONVENTION
--------------
The S/N values in SNR_RESEL_VALUES are defined PER KCWI RESOLUTION ELEMENT.
PIXELS_PER_RESEL is computed from the selected slicer and detector binning.
For uncorrelated pixels:

    S/N_per_pixel = S/N_per_resolution_element / sqrt(N_pix_per_resel)

The actual reduced KCWI cube will contain resampling covariance, so the final
observed S/N should be checked empirically from fit residuals. This simulation
is intended to establish the information-content threshold for the spectra;
it does not replace the KCWI ETC or a realistic wavelength-dependent sky model.

IMPORTANT LIMITATION
--------------------
The default simulation uses Gaussian, wavelength-independent noise within each
fit window. This is deliberate for the first test: it isolates the S/N needed
for the decomposition itself. A 99% Moon will make the real noise strongly
wavelength dependent. Once the required S/N is established, compare that S/N
against the ETC / observed noise spectrum at 4800-5500 A and 4000-4600 A.

The code is intentionally conservative in one useful way: the default
"identical_population" injection gives both disks the same stellar population.
That prevents population differences from artificially helping the kinematic
separation. A second, different-population scenario can be enabled below.
"""

# =============================================================================
# IMPORTS
# =============================================================================

# Prevent BLAS from starting many threads inside each multiprocessing worker.
# These MUST be set before importing NumPy on many systems.
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import warnings
from pathlib import Path
from multiprocessing import cpu_count, get_context
import sys
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

from ppxf.ppxf import ppxf
import ppxf.sps_util as lib
import ppxf.ppxf_util as util


# =============================================================================
# USER-CONFIGURABLE CONSTANTS
# =============================================================================

C_KMS = 299792.458
RANDOM_SEED = 12345

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

# -----------------------------------------------------------------------------
# KCWI slicer + BL instrument model
# -----------------------------------------------------------------------------
# SLICER is read from target_config.txt. The Small-slicer numbers reproduce the
# original simulation exactly; Medium/Large retain the same existing scaling.

BL_SLICER_CONFIGS = {
    "Small": {
        "R_nominal": 3600.0,
        "fwhm_gal_A": 1.25,
        "linear_pixel_A": 0.625,
        "detector_binning": 1,
    },
    "Medium": {
        "R_nominal": 1800.0,
        "fwhm_gal_A": 2.50,
        "linear_pixel_A": 1.250,
        "detector_binning": 2,
    },
    "Large": {
        "R_nominal": 900.0,
        "fwhm_gal_A": 5.00,
        "linear_pixel_A": 1.250,
        "detector_binning": 2,
    },
}

def configure_bl_instrument(slicer):
    """Configure the BL LSF/sampling for Small, Medium, or Large slicer."""
    global SLICER, FWHM_GAL_A, KCWI_LINEAR_PIXEL_A, PIXELS_PER_RESEL
    global INSTRUMENT_LABEL, INSTRUMENT_R_NOMINAL

    slicer_lookup = {key.lower(): key for key in BL_SLICER_CONFIGS}
    key = slicer_lookup.get(str(slicer).strip().lower())
    if key is None:
        raise ValueError(
            "Unknown SLICER={!r}. Use 'Small', 'Medium', or 'Large'.".format(slicer)
        )

    SLICER = key
    cfg = BL_SLICER_CONFIGS[key]
    FWHM_GAL_A = float(cfg["fwhm_gal_A"])
    KCWI_LINEAR_PIXEL_A = float(cfg["linear_pixel_A"])
    PIXELS_PER_RESEL = float(FWHM_GAL_A / KCWI_LINEAR_PIXEL_A)
    INSTRUMENT_LABEL = "BL"
    INSTRUMENT_R_NOMINAL = float(cfg["R_nominal"])

    # Wrappers can call this after import. Clear prepared templates if needed so
    # the next base.main() rebuilds them at the selected instrumental resolution.
    if "G" in globals() and isinstance(globals().get("G"), dict):
        globals()["G"].clear()

    return dict(cfg)

BL_INSTRUMENT = configure_bl_instrument(SLICER)

# Full synthetic wavelength range. The templates are padded beyond the science
# windows so that Doppler shifts do not run into template edges.
WAVE_GAL_MIN = 3900.0
WAVE_GAL_MAX = 5500.0
WAVE_TEMPLATE_PAD = 150.0

# Wavelength windows to test independently.
# "red_kinematics" is expected to be the most important decomposition region.
# "blue_population" asks how useful the blue Balmer / 4000-A information is.
# "full" uses the complete Small+BL science interval simultaneously.
# FAST PILOT: test the primary kinematic window first.
# After locating the S/N transition, add the blue/full windows back in.
FIT_WINDOWS = {
    "red_kinematics": (4800.0, 5500.0),
}

# Regions removed from the stellar fit. We keep Balmer absorption available.
# In the real analysis, ionized-gas templates should be fit simultaneously;
# these masks simply prevent simulated stellar kinematics from being driven by
# wavelengths where strong nebular lines may complicate a first-pass test.
MASK_REGIONS_A = [
    (4953.0, 4965.0),   # [O III] 4959
    (4999.0, 5014.0),   # [O III] 5007
    (5193.0, 5205.0),   # [N I] 5198/5200
]

# -----------------------------------------------------------------------------
# S/N grid
# -----------------------------------------------------------------------------
# Defined PER RESOLUTION ELEMENT. The script also reports S/N per spectral pixel.
# Start broad: the literature spans roughly ~20-40 in useful two-component
# decompositions, while some analyses use much higher S/N.
SNR_RESEL_VALUES = np.array([30, 35, 40, 45, 50], dtype=float)

# Refined Monte Carlo experiment focused on the transition found by the pilot.
# The null sample calibrates the empirical 95th-percentile false-positive
# threshold, while the larger CRD sample measures the recovery probability.
N_MC_NULL = 50
N_MC_CRD = 100

# -----------------------------------------------------------------------------
# TARGET SETTINGS FROM target_config.txt
# -----------------------------------------------------------------------------
# V_A and V_B are rest-frame residual velocities relative to systemic.
# DeltaV is derived automatically as abs(V_A - V_B).
TARGETS = {
    TARGET_NAME: {
        "v_A": float(TARGET_V_A),
        "v_B": float(TARGET_V_B),
        "sigma_A": float(TARGET_SIGMA_A),
        "sigma_B": float(TARGET_SIGMA_B),
        "frac_A": float(TARGET_FRAC_A),
    }
}
TARGET_NAMES_TO_RUN = [TARGET_NAME]

# -----------------------------------------------------------------------------
# Stellar population injections
# -----------------------------------------------------------------------------
# Ages are Gyr; metallicity is [M/H]. The exact nearest XSL SSP grid point is
# selected automatically.
#
# identical_population is the conservative kinematic test: the ONLY way pPXF
# can distinguish the components is their LOSVDs.
#
# old_plus_young is more formation-history-like and can be enabled after the
# conservative threshold is known.
POPULATION_SCENARIOS = {
    "identical_population": {
        "age_A": 8.0, "metal_A": -0.2,
        "age_B": 8.0, "metal_B": -0.2,
    },
    "old_plus_young": {
        "age_A": 8.0, "metal_A": -0.2,
        "age_B": 2.0, "metal_B": -0.4,
    },
}
POPULATION_NAMES_TO_RUN = ["identical_population"]

# -----------------------------------------------------------------------------
# Template-mismatch experiment
# -----------------------------------------------------------------------------
# Run every noise realization twice:
#
#   mismatched_basis
#       Uses the original sparse 8-template kinematic basis. The exact injected
#       SSP is NOT explicitly included unless it happens to coincide with one
#       of those grid points.
#
#   matched_control
#       Uses the same sparse basis PLUS the exact XSL SSP(s) used to construct
#       the injected spectrum (and the null spectrum). This isolates how much
#       the recovery threshold is being driven by photon noise / KCWI resolution
#       versus template mismatch.
#
# The same random-noise seeds are reused between the two template cases so that
# their comparison is paired realization-by-realization.
TEMPLATE_CASES_TO_RUN = ["mismatched_basis", "matched_control"]

# -----------------------------------------------------------------------------
# Optional stress tests
# -----------------------------------------------------------------------------
# The baseline first run uses the target values above. Once the approximate S/N
# transition is known, enable these to test unequal light ratios and different
# intrinsic dispersions around that S/N transition.
RUN_STRESS_TEST = False
STRESS_LIGHT_FRACTIONS_A = [0.50, 0.35, 0.20]
STRESS_SIGMA_PAIRS = [(40.0, 40.0), (50.0, 50.0), (40.0, 70.0)]

# -----------------------------------------------------------------------------
# Mitzkus-style global velocity search
# -----------------------------------------------------------------------------
# 17 values -> 289 pPXF fits per noisy spectrum, the same number of grid cells
# as the Mitzkus et al. implementation discussed in the analysis.
VEL_GRID_MIN = -160.0
VEL_GRID_MAX = +160.0
VEL_GRID_STEP = 20.0

SIGMA_START = 60.0
SIGMA_MIN = 5.0
SIGMA_MAX = 180.0

# Fit flexibility. Keep modest for kinematics.
ADEGREE = 4
MDEGREE = 0

# -----------------------------------------------------------------------------
# Recovery / detection criteria
# -----------------------------------------------------------------------------
# We now keep THREE deliberately distinct success definitions.
#
# 1) detection_success:
#       "Did the spectrum securely require two meaningful LOSVDs?"
#       Requires empirical significance, non-negligible light in both
#       components, a non-zero separation, and DeltaV recovered within 25 km/s.
#
# 2) relative_kinematic_success:
#       "Did we recover the LOSVD separation accurately?"
#       Same physical/significance requirements, but DeltaV must be within
#       15 km/s of the injected separation.
#
# 3) absolute_kinematic_success:
#       "Did we recover both individual velocities precisely?"
#       Requires detection_success AND both V_A and V_B within 20 km/s.
#
# This distinction is important because the pilot showed cases where both
# velocities shifted together by a common zero-point offset while DeltaV was
# nevertheless recovered very accurately.
ABSOLUTE_VELOCITY_TOLERANCE = 20.0    # km/s, each component vs truth
DETECTION_SEPARATION_TOLERANCE = 25.0 # km/s, broad two-LOSVD detection
RELATIVE_SEPARATION_TOLERANCE = 15.0  # km/s, precise DeltaV recovery
MIN_COMPONENT_LIGHT = 0.15            # both components must contribute >=15%
MIN_RECOVERED_SEPARATION = 50.0       # reject near-single-component ridge
NULL_PERCENTILE = 95.0                # empirical 5% false-positive Delta-chi2 threshold

# -----------------------------------------------------------------------------
# One-component null spectrum
# -----------------------------------------------------------------------------
# A broad single LOSVD is a useful false-positive stress test because it can
# superficially resemble two blended components.
NULL_V = 0.0
NULL_SIGMA = 90.0
NULL_AGE = 8.0
NULL_METAL = -0.2

# -----------------------------------------------------------------------------
# Kinematic fitting template basis
# -----------------------------------------------------------------------------
# Like Mitzkus et al., do NOT use hundreds of SSPs in every velocity-grid cell.
# Use a compact basis spanning plausible stellar populations. Both kinematic
# components receive an identical copy of this basis.
KINEMATIC_BASIS_AGE_METAL = [
    (1.5, -0.5),
    (1.5,  0.0),
    (3.0, -0.5),
    (3.0,  0.0),
    (6.0, -0.5),
    (6.0,  0.0),
    (10.0, -0.5),
    (10.0, 0.0),
]

# Multiprocessing. Set to 1 when running interactively/Jupyter if multiprocessing
# causes spawn/pickling issues. For a standalone .py script, multiple processes
# can save substantial time.
N_PROCESSES = 3

# On macOS, Python normally uses the "spawn" start method. That would re-import
# this script in every worker and the large XSL template state in G would not be
# inherited automatically. For a standalone script on macOS/Linux, "fork" is
# much more convenient here because the already-loaded template arrays are
# inherited copy-on-write. On Windows the code falls back to serial execution.
MP_START_METHOD = "fork" if sys.platform != "win32" else None

# Output directory
TARGET_TAG = "".join(c if (c.isalnum() or c in "-_") else "_" for c in TARGET_NAME)
OUTPUT_DIR = Path(
    "KCWI_CRD_injection_recovery_refined_results_{}_{}".format(
        TARGET_TAG, SLICER
    )
)

# Extra full chi^2 maps are useful later but add 289 fits per S/N/window.
# Keep False for the fast pilot run.
RUN_EXAMPLE_CHI2_MAPS = False

# Print progress after every completed Monte Carlo realization.
# Each realization itself contains the complete 17x17 = 289-cell pPXF grid.
PROGRESS_EVERY = 1

# The brute-force grid intentionally visits degenerate cells, which can make
# capfit emit harmless divide-by-zero/invalid-value RuntimeWarnings. Hide only
# these numerical warnings during the pilot to keep the progress output readable.
SUPPRESS_CAPFIT_NUMERIC_WARNINGS = True

if SUPPRESS_CAPFIT_NUMERIC_WARNINGS:
    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
        module=r"capfit\.capfit",
    )


# =============================================================================
# GLOBALS FILLED BY prepare_templates()
# =============================================================================

G = {}


# =============================================================================
# TEMPLATE / WAVELENGTH PREPARATION
# =============================================================================

def nearest_ssp_indices(sps, age_gyr, metallicity):
    """Return nearest (age_index, metallicity_index) in a pPXF SPS grid."""
    age_grid = np.asarray(sps.age_grid)
    metal_grid = np.asarray(sps.metal_grid)

    # pPXF SPS grids are commonly 2D with shape (n_age, n_metal).
    if age_grid.ndim == 2:
        dist = ((np.log10(age_grid) - np.log10(age_gyr)) / 0.25)**2 \
             + ((metal_grid - metallicity) / 0.25)**2
        i, j = np.unravel_index(np.nanargmin(dist), dist.shape)
        return int(i), int(j)

    # Fallback if a version exposes 1D age and metallicity axes.
    i = int(np.nanargmin(np.abs(np.log10(age_grid) - np.log10(age_gyr))))
    j = int(np.nanargmin(np.abs(metal_grid - metallicity)))
    return i, j


def get_ssp_template(sps, age_gyr, metallicity):
    """Return one XSL SSP template and the actual grid parameters selected."""
    i, j = nearest_ssp_indices(sps, age_gyr, metallicity)
    template = np.asarray(sps.templates[:, i, j], dtype=float).copy()

    age_grid = np.asarray(sps.age_grid)
    metal_grid = np.asarray(sps.metal_grid)
    if age_grid.ndim == 2:
        age_actual = float(age_grid[i, j])
        metal_actual = float(metal_grid[i, j])
    else:
        age_actual = float(age_grid[i])
        metal_actual = float(metal_grid[j])

    return template, age_actual, metal_actual


def normalize_template(template, lam, norm_range=(5000.0, 5350.0)):
    """Normalize a template to median unity in a continuum-rich optical region."""
    use = (lam >= norm_range[0]) & (lam <= norm_range[1]) & np.isfinite(template)
    if np.sum(use) < 10:
        use = np.isfinite(template)
    med = np.nanmedian(template[use])
    if not np.isfinite(med) or med == 0:
        raise ValueError("Template normalization failed.")
    return template / med


def prepare_templates():
    """
    Load XSL at the KCWI LSF, construct the galaxy log-lambda grid, and build
    BOTH kinematic template cases used in the refined experiment.

    The two fit cases are:

        mismatched_basis
            Original sparse 8-template basis.

        matched_control
            The same sparse basis plus the exact XSL grid SSP(s) used to
            generate the injected CRD and null spectra.

    Both stellar components always receive identical copies of a given basis,
    so no population identity is imposed on Disk A versus Disk B during the
    kinematic decomposition.
    """
    # Estimate the native KCWI velocity sampling by log-rebinning a linear
    # 0.625-A/pixel wavelength grid.
    wave_linear = np.arange(
        WAVE_GAL_MIN,
        WAVE_GAL_MAX + KCWI_LINEAR_PIXEL_A,
        KCWI_LINEAR_PIXEL_A
    )
    dummy = np.ones_like(wave_linear)
    _, loglam_gal, velscale = util.log_rebin(
        [wave_linear[0], wave_linear[-1]], dummy
    )
    lam_gal = np.exp(loglam_gal)

    # Load XSL SSPs broadened to the assumed KCWI Small+BL instrumental LSF.
    ppxf_dir = Path(lib.__file__).parent
    filename = ppxf_dir / "sps_models/spectra_xsl_9.0.npz"
    if not filename.is_file():
        raise FileNotFoundError(
            "XSL SPS file not found: {}\n"
            "Download spectra_xsl_9.0.npz using the standard pPXF SPS-data "
            "instructions before running this simulation.".format(filename)
        )

    sps = lib.sps_lib(filename, velscale, FWHM_GAL_A)
    lam_temp = np.asarray(sps.lam_temp, dtype=float)
    loglam_temp = np.log(lam_temp)

    # Keep enough template padding to permit Doppler shifts without edge loss.
    keep_temp = (
        (lam_temp >= WAVE_GAL_MIN - WAVE_TEMPLATE_PAD) &
        (lam_temp <= WAVE_GAL_MAX + WAVE_TEMPLATE_PAD)
    )
    lam_temp_pad = lam_temp[keep_temp]
    loglam_temp_pad = loglam_temp[keep_temp]

    def add_template_if_new(basis_list, info_list, seen, age, metal, source):
        """
        Select the nearest actual XSL SSP, normalize it, and append it only if
        that exact (age, metallicity) grid point is not already present.
        """
        temp_all, age_actual, metal_actual = get_ssp_template(sps, age, metal)
        key = (age_actual, metal_actual)

        if key in seen:
            return

        temp = normalize_template(temp_all[keep_temp], lam_temp_pad)
        basis_list.append(temp)
        info_list.append((age_actual, metal_actual, source))
        seen.add(key)

    # -------------------------------------------------------------------------
    # CASE 1: original sparse / deliberately mismatched basis
    # -------------------------------------------------------------------------
    sparse_list = []
    sparse_info = []
    sparse_seen = set()

    for age, metal in KINEMATIC_BASIS_AGE_METAL:
        add_template_if_new(
            sparse_list, sparse_info, sparse_seen,
            age, metal, "sparse_basis"
        )

    sparse_basis = np.column_stack(sparse_list)

    # -------------------------------------------------------------------------
    # CASE 2: matched control
    # -------------------------------------------------------------------------
    # Begin with exactly the same sparse basis, then explicitly add every SSP
    # used to generate the requested CRD injections plus the null spectrum.
    matched_list = [sparse_basis[:, k].copy() for k in range(sparse_basis.shape[1])]
    matched_info = list(sparse_info)
    matched_seen = set((age, metal) for age, metal, _ in sparse_info)

    for pop_name in POPULATION_NAMES_TO_RUN:
        scenario = POPULATION_SCENARIOS[pop_name]

        add_template_if_new(
            matched_list, matched_info, matched_seen,
            scenario["age_A"], scenario["metal_A"],
            "{}_A_exact".format(pop_name)
        )
        add_template_if_new(
            matched_list, matched_info, matched_seen,
            scenario["age_B"], scenario["metal_B"],
            "{}_B_exact".format(pop_name)
        )

    # Also make the null experiment exactly matched in the matched_control case,
    # because the empirical significance threshold must be calibrated using the
    # SAME model flexibility as the corresponding CRD fits.
    add_template_if_new(
        matched_list, matched_info, matched_seen,
        NULL_AGE, NULL_METAL, "null_exact"
    )

    matched_basis = np.column_stack(matched_list)

    def package_fit_case(basis, info):
        n_basis = basis.shape[1]
        return {
            "basis": basis,
            "templates_fit": np.column_stack([basis, basis]),
            "templates_one": basis.copy(),
            "component_fit": np.array(
                [0] * n_basis + [1] * n_basis,
                dtype=int
            ),
            "n_basis": n_basis,
            "basis_info": list(info),
        }

    fit_cases = {
        "mismatched_basis": package_fit_case(sparse_basis, sparse_info),
        "matched_control": package_fit_case(matched_basis, matched_info),
    }

    # Verify the velocity scale represented by the SPS wavelength grid.
    local_velscale = C_KMS * np.nanmedian(np.diff(np.log(lam_temp_pad)))

    print("\nKCWI / XSL template preparation")
    print("--------------------------------")
    print("Galaxy log-grid velocity scale: {:.2f} km/s/pixel".format(velscale))
    print("XSL template velocity scale:    {:.2f} km/s/pixel".format(local_velscale))
    print("KCWI FWHM used:                 {:.3f} A".format(FWHM_GAL_A))

    for case_name in TEMPLATE_CASES_TO_RUN:
        case = fit_cases[case_name]
        print("\nTemplate case: {}".format(case_name))
        print("  Kinematic basis templates: {}".format(case["n_basis"]))
        for k, (age, metal, source) in enumerate(case["basis_info"]):
            print(
                "  basis {:02d}: age={:.3f} Gyr, [M/H]={:+.2f} ({})".format(
                    k, age, metal, source
                )
            )

    G.update({
        "sps": sps,
        "velscale": float(velscale),
        "lam_gal": lam_gal,
        "loglam_gal": loglam_gal,
        "lam_temp": lam_temp_pad,
        "loglam_temp": loglam_temp_pad,
        "keep_temp": keep_temp,
        "fit_cases": fit_cases,
    })


# =============================================================================
# SYNTHETIC SPECTRUM GENERATION
# =============================================================================

def apply_gaussian_losvd(template, loglam, velocity, sigma, velscale):
    """
    Apply a Gaussian LOSVD to a template already matched to the instrumental LSF.

    The instrumental LSF is already contained in the XSL template through
    sps_lib(..., FWHM_GAL_A). Therefore sigma here is the *intrinsic stellar*
    velocity dispersion, exactly as it will be in the pPXF fit.
    """
    # Intrinsic Gaussian broadening on the logarithmic wavelength grid.
    sigma_pix = max(float(sigma) / float(velscale), 1e-4)
    broadened = gaussian_filter1d(template, sigma_pix, mode="nearest")

    # Relativistically adequate logarithmic Doppler translation for these low V.
    dloglam = np.log1p(float(velocity) / C_KMS)

    # At output coordinate x, a redshifted spectrum samples the rest spectrum at
    # x - dloglam. Interpolation is done on the padded template grid.
    shifted = np.interp(
        loglam - dloglam,
        loglam,
        broadened,
        left=np.nan,
        right=np.nan
    )
    return shifted


def interpolate_to_galaxy_grid(template_on_temp_grid):
    """Interpolate a padded-template-grid model onto the galaxy wavelength grid."""
    return np.interp(
        G["loglam_gal"],
        G["loglam_temp"],
        template_on_temp_grid,
        left=np.nan,
        right=np.nan
    )


def make_two_component_noiseless(pop_name, v_A, v_B, sigma_A, sigma_B, frac_A):
    """Construct a noiseless two-component KCWI spectrum."""
    scenario = POPULATION_SCENARIOS[pop_name]
    sps = G["sps"]
    keep = G["keep_temp"]

    temp_A_all, age_A_actual, metal_A_actual = get_ssp_template(
        sps, scenario["age_A"], scenario["metal_A"]
    )
    temp_B_all, age_B_actual, metal_B_actual = get_ssp_template(
        sps, scenario["age_B"], scenario["metal_B"]
    )

    temp_A = normalize_template(temp_A_all[keep], G["lam_temp"])
    temp_B = normalize_template(temp_B_all[keep], G["lam_temp"])

    A_pad = apply_gaussian_losvd(
        temp_A, G["loglam_temp"], v_A, sigma_A, G["velscale"]
    )
    B_pad = apply_gaussian_losvd(
        temp_B, G["loglam_temp"], v_B, sigma_B, G["velscale"]
    )

    A = interpolate_to_galaxy_grid(A_pad)
    B = interpolate_to_galaxy_grid(B_pad)

    model = frac_A * A + (1.0 - frac_A) * B

    # Normalize combined spectrum to median unity in the main kinematic region.
    norm = (G["lam_gal"] >= 5000.0) & (G["lam_gal"] <= 5350.0)
    scale = np.nanmedian(model[norm])
    model /= scale
    A /= scale
    B /= scale

    info = {
        "age_A_actual": age_A_actual,
        "metal_A_actual": metal_A_actual,
        "age_B_actual": age_B_actual,
        "metal_B_actual": metal_B_actual,
    }
    return model, A, B, info


def make_one_component_noiseless():
    """Construct a broad true single-component spectrum for the null experiment."""
    sps = G["sps"]
    keep = G["keep_temp"]
    temp_all, age_actual, metal_actual = get_ssp_template(sps, NULL_AGE, NULL_METAL)
    temp = normalize_template(temp_all[keep], G["lam_temp"])

    model_pad = apply_gaussian_losvd(
        temp, G["loglam_temp"], NULL_V, NULL_SIGMA, G["velscale"]
    )
    model = interpolate_to_galaxy_grid(model_pad)

    norm = (G["lam_gal"] >= 5000.0) & (G["lam_gal"] <= 5350.0)
    model /= np.nanmedian(model[norm])
    return model, age_actual, metal_actual


def snr_resel_to_pixel(snr_resel):
    """Convert S/N per resolution element to approximate S/N per detector pixel."""
    return float(snr_resel) / np.sqrt(PIXELS_PER_RESEL)


def add_noise(noiseless, snr_resel, rng):
    """
    Add constant Gaussian noise corresponding to the requested S/N per KCWI resel.

    With continuum normalized near unity, sigma_noise = 1 / (S/N per pixel).
    """
    snr_pix = snr_resel_to_pixel(snr_resel)
    noise_sigma = 1.0 / snr_pix
    noise = np.full(noiseless.size, noise_sigma, dtype=float)
    galaxy = noiseless + rng.normal(0.0, noise_sigma, noiseless.size)
    return galaxy, noise


# =============================================================================
# GOOD-PIXEL MASKS
# =============================================================================

def goodpixels_for_window(window_name):
    """Return galaxy-pixel indices used by pPXF for a requested wavelength window."""
    lo, hi = FIT_WINDOWS[window_name]
    lam = G["lam_gal"]
    good = (lam >= lo) & (lam <= hi)

    for mlo, mhi in MASK_REGIONS_A:
        good &= ~((lam >= mlo) & (lam <= mhi))

    good &= np.isfinite(lam)
    return np.where(good)[0]


# =============================================================================
# pPXF FITS
# =============================================================================

def fit_one_component(galaxy, noise, goodpixels, template_case):
    """Fit one stellar LOSVD using the requested XSL kinematic template basis."""
    fit_case = G["fit_cases"][template_case]

    scale = np.nanmedian(galaxy[goodpixels])
    gal = galaxy / scale
    err = noise / scale

    start = [0.0, 80.0]
    bounds = [[-220.0, 220.0], [SIGMA_MIN, SIGMA_MAX]]

    pp = ppxf(
        fit_case["templates_one"],
        gal,
        err,
        G["velscale"],
        start,
        moments=2,
        bounds=bounds,
        goodpixels=goodpixels,
        degree=ADEGREE,
        mdegree=MDEGREE,
        lam=G["lam_gal"],
        lam_temp=G["lam_temp"],
        quiet=True,
    )

    chi2_total = np.sum(
        ((gal[goodpixels] - pp.bestfit[goodpixels]) / err[goodpixels])**2
    )
    return pp, float(chi2_total)


def two_component_cell_fit(
        galaxy, noise, goodpixels, v1_start, v2_start, template_case):
    """
    Fit one Mitzkus-style velocity cell using the requested template basis.

    pPXF may optimize each velocity only within +/- VEL_GRID_STEP/2 around that
    cell center. Sigma and linear template weights remain free.
    """
    fit_case = G["fit_cases"][template_case]

    scale = np.nanmedian(galaxy[goodpixels])
    gal = galaxy / scale
    err = noise / scale

    half = 0.5 * VEL_GRID_STEP
    start = [[v1_start, SIGMA_START], [v2_start, SIGMA_START]]
    bounds = [
        [[v1_start - half, v1_start + half], [SIGMA_MIN, SIGMA_MAX]],
        [[v2_start - half, v2_start + half], [SIGMA_MIN, SIGMA_MAX]],
    ]

    pp = ppxf(
        fit_case["templates_fit"],
        gal,
        err,
        G["velscale"],
        start,
        moments=[2, 2],
        component=fit_case["component_fit"],
        bounds=bounds,
        goodpixels=goodpixels,
        degree=ADEGREE,
        mdegree=MDEGREE,
        lam=G["lam_gal"],
        lam_temp=G["lam_temp"],
        quiet=True,
    )

    chi2_total = np.sum(
        ((gal[goodpixels] - pp.bestfit[goodpixels]) / err[goodpixels])**2
    )

    # Approximate light fractions for the kinematic quality-control test.
    w = np.asarray(pp.weights, dtype=float)
    n = fit_case["n_basis"]

    w1 = np.sum(np.clip(w[:n], 0.0, None))
    w2 = np.sum(np.clip(w[n:2*n], 0.0, None))
    wt = w1 + w2

    if wt > 0:
        frac1 = w1 / wt
        frac2 = w2 / wt
    else:
        frac1 = np.nan
        frac2 = np.nan

    return {
        "pp": pp,
        "chi2_total": float(chi2_total),
        "v1": float(pp.sol[0][0]),
        "sig1": float(pp.sol[0][1]),
        "v2": float(pp.sol[1][0]),
        "sig2": float(pp.sol[1][1]),
        "frac1": float(frac1),
        "frac2": float(frac2),
    }


def brute_force_velocity_grid(
        galaxy, noise, goodpixels, template_case, return_map=False):
    """
    Run the independent two-component velocity grid and return its global minimum.

    The same full square grid is retained for both template cases so the
    experiment remains directly comparable to the Mitzkus-style search.
    """
    v_grid = np.arange(
        VEL_GRID_MIN,
        VEL_GRID_MAX + 0.5*VEL_GRID_STEP,
        VEL_GRID_STEP
    )
    n = len(v_grid)
    chi2_map = np.full((n, n), np.nan)

    best = None
    best_chi2 = np.inf

    for i, v1 in enumerate(v_grid):
        for j, v2 in enumerate(v_grid):
            try:
                result = two_component_cell_fit(
                    galaxy, noise, goodpixels,
                    float(v1), float(v2),
                    template_case
                )
            except Exception as exc:
                warnings.warn(
                    "pPXF cell failed at ({}, {}) [{}]: {}".format(
                        v1, v2, template_case, exc
                    )
                )
                continue

            chi2_map[j, i] = result["chi2_total"]

            if result["chi2_total"] < best_chi2:
                best_chi2 = result["chi2_total"]
                best = result
                best["grid_v1_start"] = float(v1)
                best["grid_v2_start"] = float(v2)

    if best is None:
        raise RuntimeError(
            "All velocity-grid pPXF cells failed for template case {}.".format(
                template_case
            )
        )

    if return_map:
        return best, v_grid, chi2_map

    return best


# =============================================================================
# RECOVERY METRICS
# =============================================================================

def match_labels_to_truth(v1, v2, sig1, sig2, frac1, frac2, truth):
    """Resolve the mathematical A/B label symmetry by choosing the closer assignment."""
    direct = (v1 - truth["v_A"])**2 + (v2 - truth["v_B"])**2
    swapped = (v2 - truth["v_A"])**2 + (v1 - truth["v_B"])**2

    if direct <= swapped:
        return {
            "v_A_rec": v1, "v_B_rec": v2,
            "sigma_A_rec": sig1, "sigma_B_rec": sig2,
            "frac_A_rec": frac1, "frac_B_rec": frac2,
            "swapped": False,
        }
    return {
        "v_A_rec": v2, "v_B_rec": v1,
        "sigma_A_rec": sig2, "sigma_B_rec": sig1,
        "frac_A_rec": frac2, "frac_B_rec": frac1,
        "swapped": True,
    }


def evaluate_recovery(best_two, chi2_one, truth, null_delta_chi2_threshold):
    """
    Evaluate three distinct levels of recovery.

    detection_success
        Significant two-component model, both components carry useful light,
        recovered separation is non-trivial, and DeltaV is within the broad
        detection tolerance.

    relative_kinematic_success
        Same requirements, but DeltaV is recovered to the tighter relative
        kinematic tolerance.

    absolute_kinematic_success
        detection_success plus BOTH individual velocities recovered within the
        absolute velocity tolerance.

    The function also stores the common-mode velocity offset:
        dV_common = midpoint_rec - midpoint_true
    which diagnoses the failure mode seen in the pilot where both LOSVDs shift
    together while their separation remains correct.
    """
    matched = match_labels_to_truth(
        best_two["v1"], best_two["v2"],
        best_two["sig1"], best_two["sig2"],
        best_two["frac1"], best_two["frac2"],
        truth,
    )

    vA = matched["v_A_rec"]
    vB = matched["v_B_rec"]

    true_sep = abs(truth["v_A"] - truth["v_B"])
    rec_sep = abs(vA - vB)
    dsep = rec_sep - true_sep

    true_mid = 0.5 * (truth["v_A"] + truth["v_B"])
    rec_mid = 0.5 * (vA + vB)
    dV_common = rec_mid - true_mid

    delta_chi2 = float(chi2_one - best_two["chi2_total"])

    absolute_velocity_ok = (
        abs(vA - truth["v_A"]) <= ABSOLUTE_VELOCITY_TOLERANCE and
        abs(vB - truth["v_B"]) <= ABSOLUTE_VELOCITY_TOLERANCE
    )

    minimum_separation_ok = rec_sep >= MIN_RECOVERED_SEPARATION

    detection_separation_ok = (
        abs(dsep) <= DETECTION_SEPARATION_TOLERANCE and
        minimum_separation_ok
    )

    relative_separation_ok = (
        abs(dsep) <= RELATIVE_SEPARATION_TOLERANCE and
        minimum_separation_ok
    )

    light_ok = (
        matched["frac_A_rec"] >= MIN_COMPONENT_LIGHT and
        matched["frac_B_rec"] >= MIN_COMPONENT_LIGHT
    )

    significance_ok = delta_chi2 > null_delta_chi2_threshold

    detection_success = (
        significance_ok and
        light_ok and
        detection_separation_ok
    )

    relative_kinematic_success = (
        significance_ok and
        light_ok and
        relative_separation_ok
    )

    absolute_kinematic_success = (
        detection_success and
        absolute_velocity_ok
    )

    out = dict(matched)
    out.update({
        "delta_chi2": delta_chi2,
        "null_delta_chi2_threshold": float(null_delta_chi2_threshold),

        "true_sep": true_sep,
        "rec_sep": rec_sep,
        "dsep": dsep,
        "abs_dsep": abs(dsep),

        "true_midpoint": true_mid,
        "rec_midpoint": rec_mid,
        "dV_common": dV_common,
        "abs_dV_common": abs(dV_common),

        "absolute_velocity_ok": absolute_velocity_ok,
        "minimum_separation_ok": minimum_separation_ok,
        "detection_separation_ok": detection_separation_ok,
        "relative_separation_ok": relative_separation_ok,
        "light_ok": light_ok,
        "significance_ok": significance_ok,

        "detection_success": detection_success,
        "relative_kinematic_success": relative_kinematic_success,
        "absolute_kinematic_success": absolute_kinematic_success,

        # Keep "success" as an alias for the strictest criterion so older
        # downstream code will not silently change meaning.
        "success": absolute_kinematic_success,

        "dv_A": vA - truth["v_A"],
        "dv_B": vB - truth["v_B"],
    })
    return out


# =============================================================================
# ONE MONTE CARLO REALIZATION
# =============================================================================

def run_null_realization(args):
    """
    Worker: one true one-component realization fitted by one and two components.

    The template case is explicit because each fitting basis needs its own
    empirical false-positive Delta-chi2 distribution.
    """
    template_case, snr_resel, window_name, seed = args

    rng = np.random.RandomState(seed)
    noiseless, _, _ = make_one_component_noiseless()
    galaxy, noise = add_noise(noiseless, snr_resel, rng)
    goodpixels = goodpixels_for_window(window_name)

    _, chi2_one = fit_one_component(
        galaxy, noise, goodpixels, template_case
    )
    best_two = brute_force_velocity_grid(
        galaxy, noise, goodpixels,
        template_case=template_case,
        return_map=False
    )

    return {
        "template_case": template_case,
        "snr_resel": snr_resel,
        "snr_pixel": snr_resel_to_pixel(snr_resel),
        "window": window_name,
        "chi2_one": chi2_one,
        "chi2_two": best_two["chi2_total"],
        "delta_chi2": chi2_one - best_two["chi2_total"],
        "v1": best_two["v1"],
        "v2": best_two["v2"],
        "sep": abs(best_two["v1"] - best_two["v2"]),
        "frac1": best_two["frac1"],
        "frac2": best_two["frac2"],
    }


def run_crd_realization(args):
    """Worker: one true two-component realization."""
    (
        target_name, pop_name, template_case,
        snr_resel, window_name,
        sigma_A, sigma_B, frac_A,
        null_threshold, seed
    ) = args

    rng = np.random.RandomState(seed)
    target = TARGETS[target_name]

    truth = {
        "v_A": float(target["v_A"]),
        "v_B": float(target["v_B"]),
        "sigma_A": float(sigma_A),
        "sigma_B": float(sigma_B),
        "frac_A": float(frac_A),
        "frac_B": float(1.0 - frac_A),
    }

    noiseless, _, _, pop_info = make_two_component_noiseless(
        pop_name,
        truth["v_A"], truth["v_B"],
        truth["sigma_A"], truth["sigma_B"],
        truth["frac_A"],
    )

    galaxy, noise = add_noise(noiseless, snr_resel, rng)
    goodpixels = goodpixels_for_window(window_name)

    _, chi2_one = fit_one_component(
        galaxy, noise, goodpixels, template_case
    )
    best_two = brute_force_velocity_grid(
        galaxy, noise, goodpixels,
        template_case=template_case,
        return_map=False
    )

    metrics = evaluate_recovery(
        best_two, chi2_one, truth, null_threshold
    )

    row = {
        "target": target_name,
        "population": pop_name,
        "template_case": template_case,
        "window": window_name,
        "snr_resel": snr_resel,
        "snr_pixel": snr_resel_to_pixel(snr_resel),
        "v_A_true": truth["v_A"],
        "v_B_true": truth["v_B"],
        "sigma_A_true": truth["sigma_A"],
        "sigma_B_true": truth["sigma_B"],
        "frac_A_true": truth["frac_A"],
        "frac_B_true": truth["frac_B"],
        "chi2_one": chi2_one,
        "chi2_two": best_two["chi2_total"],
    }

    row.update(pop_info)
    row.update(metrics)
    return row


# =============================================================================
# MONTE CARLO DRIVERS
# =============================================================================

def _format_seconds(seconds):
    """Human-readable elapsed/ETA string."""
    if not np.isfinite(seconds) or seconds < 0:
        return "--"
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, sec = divmod(rem, 60)
    if h > 0:
        return "{}h {:02d}m {:02d}s".format(h, m, sec)
    if m > 0:
        return "{}m {:02d}s".format(m, sec)
    return "{}s".format(sec)


def pool_map(func, jobs, stage_name="jobs"):
    """
    Run workers serially or with multiprocessing and display live progress + ETA.

    Progress is counted per Monte Carlo realization, not per individual pPXF
    velocity cell. In this pilot each completed job therefore represents 289
    two-component pPXF grid fits plus one one-component fit.
    """
    total = len(jobs)
    if total == 0:
        return []

    t0 = time.time()
    results = []

    def report(done):
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else np.nan
        remaining = (total - done) / rate if np.isfinite(rate) and rate > 0 else np.nan
        pct = 100.0 * done / total
        print(
            "[{stage}] {done:4d}/{total:4d} ({pct:5.1f}%) | "
            "elapsed {elapsed} | ETA {eta}".format(
                stage=stage_name, done=done, total=total, pct=pct,
                elapsed=_format_seconds(elapsed), eta=_format_seconds(remaining)
            ),
            flush=True,
        )

    if N_PROCESSES <= 1 or MP_START_METHOD is None:
        for done, job in enumerate(jobs, start=1):
            results.append(func(job))
            if done == 1 or done % PROGRESS_EVERY == 0 or done == total:
                report(done)
        return results

    # chunksize=1 is intentional: it provides responsive progress updates and
    # distributes expensive realizations evenly among workers.
    ctx = get_context(MP_START_METHOD)
    with ctx.Pool(processes=N_PROCESSES) as pool:
        iterator = pool.imap_unordered(func, jobs, chunksize=1)
        for done, result in enumerate(iterator, start=1):
            results.append(result)
            if done == 1 or done % PROGRESS_EVERY == 0 or done == total:
                report(done)

    return results


def run_null_experiment():
    """
    Calibrate empirical Delta-chi2 thresholds separately for each template case.

    The SAME noise seeds are reused for mismatched_basis and matched_control so
    differences between their null distributions are not caused by different
    random realizations.
    """
    print("\n" + "="*78)
    print("NULL EXPERIMENT: true one-component spectra")
    print("="*78)

    rng = np.random.RandomState(RANDOM_SEED)
    jobs = []

    for window_name in FIT_WINDOWS:
        for snr in SNR_RESEL_VALUES:
            # Generate one seed set and reuse it for BOTH template cases.
            seeds = rng.randint(1, 2**31 - 1, size=N_MC_NULL)

            for template_case in TEMPLATE_CASES_TO_RUN:
                for seed in seeds:
                    jobs.append((
                        template_case,
                        float(snr),
                        window_name,
                        int(seed)
                    ))

    rows = pool_map(
        run_null_realization,
        jobs,
        stage_name="NULL"
    )
    null_df = pd.DataFrame(rows)

    thresholds = (
        null_df
        .groupby(["template_case", "window", "snr_resel"])["delta_chi2"]
        .quantile(NULL_PERCENTILE / 100.0)
        .rename("null_delta_chi2_threshold")
        .reset_index()
    )

    null_df.to_csv(
        OUTPUT_DIR / "null_realizations.csv",
        index=False
    )
    thresholds.to_csv(
        OUTPUT_DIR / "null_delta_chi2_thresholds.csv",
        index=False
    )

    print(
        "\nEmpirical {}th-percentile null Delta-chi2 thresholds:".format(
            NULL_PERCENTILE
        )
    )
    print(thresholds.to_string(index=False))

    return null_df, thresholds


def build_crd_scenarios():
    """Return requested (sigma_A, sigma_B, frac_A) scenarios."""
    scenarios = []
    if not RUN_STRESS_TEST:
        for target_name in TARGET_NAMES_TO_RUN:
            target = TARGETS[target_name]
            scenarios.append((
                target_name,
                float(target["sigma_A"]),
                float(target["sigma_B"]),
                float(target["frac_A"]),
            ))
        return scenarios

    for target_name in TARGET_NAMES_TO_RUN:
        for sigma_A, sigma_B in STRESS_SIGMA_PAIRS:
            for frac_A in STRESS_LIGHT_FRACTIONS_A:
                scenarios.append((target_name, sigma_A, sigma_B, frac_A))
    return scenarios


def run_crd_experiment(thresholds):
    """
    Run the two-component injection/recovery experiment for both template cases.

    As in the null experiment, the same noise seeds are reused between template
    cases. This makes matched-versus-mismatched comparisons paired.
    """
    print("\n" + "="*78)
    print("CRD INJECTION / RECOVERY EXPERIMENT")
    print("="*78)

    threshold_lookup = {
        (
            row.template_case,
            row.window,
            float(row.snr_resel)
        ): float(row.null_delta_chi2_threshold)
        for row in thresholds.itertuples(index=False)
    }

    rng = np.random.RandomState(RANDOM_SEED + 1000)
    jobs = []

    for target_name, sigma_A, sigma_B, frac_A in build_crd_scenarios():
        for pop_name in POPULATION_NAMES_TO_RUN:
            for window_name in FIT_WINDOWS:
                for snr in SNR_RESEL_VALUES:

                    # Generate one seed set and reuse it for BOTH template cases.
                    seeds = rng.randint(
                        1, 2**31 - 1,
                        size=N_MC_CRD
                    )

                    for template_case in TEMPLATE_CASES_TO_RUN:
                        null_threshold = threshold_lookup[
                            (
                                template_case,
                                window_name,
                                float(snr)
                            )
                        ]

                        for seed in seeds:
                            jobs.append((
                                target_name,
                                pop_name,
                                template_case,
                                float(snr),
                                window_name,
                                float(sigma_A),
                                float(sigma_B),
                                float(frac_A),
                                float(null_threshold),
                                int(seed)
                            ))

    rows = pool_map(
        run_crd_realization,
        jobs,
        stage_name="CRD"
    )

    df = pd.DataFrame(rows)
    df.to_csv(
        OUTPUT_DIR / "crd_recovery_realizations.csv",
        index=False
    )
    return df


# =============================================================================
# SUMMARY / RECOMMENDED S/N
# =============================================================================

def summarize_recovery(df):
    """
    Summarize all three recovery definitions plus the common-mode and DeltaV errors.
    """
    group_cols = [
        "target",
        "population",
        "template_case",
        "window",
        "snr_resel",
        "sigma_A_true",
        "sigma_B_true",
        "frac_A_true",
    ]

    summary = df.groupby(group_cols).agg(
        n=("detection_success", "size"),

        detection_fraction=("detection_success", "mean"),
        relative_kinematic_fraction=("relative_kinematic_success", "mean"),
        absolute_kinematic_fraction=("absolute_kinematic_success", "mean"),

        significance_pass_fraction=("significance_ok", "mean"),
        light_pass_fraction=("light_ok", "mean"),
        detection_sep_pass_fraction=("detection_separation_ok", "mean"),
        relative_sep_pass_fraction=("relative_separation_ok", "mean"),
        absolute_velocity_pass_fraction=("absolute_velocity_ok", "mean"),

        median_abs_dv_A=("dv_A", lambda x: np.nanmedian(np.abs(x))),
        median_abs_dv_B=("dv_B", lambda x: np.nanmedian(np.abs(x))),
        p84_abs_dv_A=("dv_A", lambda x: np.nanpercentile(np.abs(x), 84)),
        p84_abs_dv_B=("dv_B", lambda x: np.nanpercentile(np.abs(x), 84)),

        median_abs_dsep=("dsep", lambda x: np.nanmedian(np.abs(x))),
        p84_abs_dsep=("dsep", lambda x: np.nanpercentile(np.abs(x), 84)),

        median_abs_common_offset=(
            "dV_common",
            lambda x: np.nanmedian(np.abs(x))
        ),
        p84_abs_common_offset=(
            "dV_common",
            lambda x: np.nanpercentile(np.abs(x), 84)
        ),

        median_rec_sep=("rec_sep", "median"),
        median_frac_A_rec=("frac_A_rec", "median"),
        median_delta_chi2=("delta_chi2", "median"),
    ).reset_index()

    summary["snr_pixel"] = summary["snr_resel"].apply(
        snr_resel_to_pixel
    )

    summary.to_csv(
        OUTPUT_DIR / "recovery_summary.csv",
        index=False
    )

    print("\nRecovery summary:")
    print(summary.to_string(index=False))

    # -------------------------------------------------------------------------
    # Recommended S/N for each success definition
    # -------------------------------------------------------------------------
    scenario_cols = [
        "target",
        "population",
        "template_case",
        "window",
        "sigma_A_true",
        "sigma_B_true",
        "frac_A_true",
    ]

    rec_rows = []

    for key, sub in summary.groupby(scenario_cols):
        sub = sub.sort_values("snr_resel")

        out = dict(
            zip(
                scenario_cols,
                key if isinstance(key, tuple) else (key,)
            )
        )

        metric_map = {
            "detection": "detection_fraction",
            "relative_kinematic": "relative_kinematic_fraction",
            "absolute_kinematic": "absolute_kinematic_fraction",
        }

        for tag, col in metric_map.items():
            good = sub[sub[col] >= 0.90]

            if len(good):
                row = good.iloc[0]
                recommended = float(row["snr_resel"])
                rec_pix = float(row["snr_pixel"])
            else:
                recommended = np.nan
                rec_pix = np.nan

            out["snr_resel_90pct_{}".format(tag)] = recommended
            out["snr_pixel_90pct_{}".format(tag)] = rec_pix

        rec_rows.append(out)

    rec_df = pd.DataFrame(rec_rows)

    rec_df.to_csv(
        OUTPUT_DIR / "recommended_snr.csv",
        index=False
    )

    print("\nFirst tested S/N reaching >=90% for each recovery definition:")
    print(rec_df.to_string(index=False))

    return summary, rec_df


# =============================================================================
# PLOTS
# =============================================================================

def plot_recovery_curves(summary):
    """
    Plot the three recovery fractions versus S/N separately for each template case.

    This is the main diagnostic for deciding:
        - S/N needed simply to detect two LOSVDs,
        - S/N needed to recover DeltaV accurately,
        - S/N needed to recover both absolute velocities accurately.
    """
    scenario_cols = [
        "target",
        "population",
        "template_case",
        "sigma_A_true",
        "sigma_B_true",
        "frac_A_true",
    ]

    for key, sub0 in summary.groupby(scenario_cols):
        target, population, template_case, sigA, sigB, fracA = key

        for window_name in FIT_WINDOWS:
            sub = sub0[
                sub0["window"] == window_name
            ].sort_values("snr_resel")

            if len(sub) == 0:
                continue

            plt.figure(figsize=(8.5, 5.5))

            plt.plot(
                sub["snr_resel"],
                sub["detection_fraction"],
                marker="o",
                label="two-LOSVD detection",
            )
            plt.plot(
                sub["snr_resel"],
                sub["relative_kinematic_fraction"],
                marker="o",
                label="relative kinematics (DeltaV)",
            )
            plt.plot(
                sub["snr_resel"],
                sub["absolute_kinematic_fraction"],
                marker="o",
                label="absolute velocities",
            )

            plt.axhline(
                0.90,
                linestyle="--",
                linewidth=1,
                label="90% recovery"
            )

            plt.ylim(-0.02, 1.02)
            plt.xlabel("S/N per KCWI resolution element")
            plt.ylabel("Recovery fraction")
            plt.title(
                "{} | {} | {} | {}\n"
                "sigma=({:.0f},{:.0f}) km/s | f_A={:.2f}".format(
                    target,
                    population,
                    template_case,
                    window_name,
                    sigA,
                    sigB,
                    fracA,
                )
            )
            plt.legend()
            plt.tight_layout()

            frac_tag = "{:.2f}".format(fracA).replace(".", "p")
            name = (
                "recovery_{}_{}_{}_{}_sig{}-{}_f{}.png"
                .format(
                    target,
                    population,
                    template_case,
                    window_name,
                    int(sigA),
                    int(sigB),
                    frac_tag,
                )
            )
            plt.savefig(
                OUTPUT_DIR / name,
                dpi=180
            )
            plt.close()


def plot_velocity_errors(summary):
    """
    Plot absolute-velocity, separation, and common-mode errors versus S/N.

    The second plot is especially useful for diagnosing whether apparent
    absolute-velocity failures are really common zero-point shifts.
    """
    scenario_cols = [
        "target",
        "population",
        "template_case",
        "sigma_A_true",
        "sigma_B_true",
        "frac_A_true",
    ]

    for key, sub0 in summary.groupby(scenario_cols):
        target, population, template_case, sigA, sigB, fracA = key

        for window_name in FIT_WINDOWS:
            sub = sub0[
                sub0["window"] == window_name
            ].sort_values("snr_resel")

            if len(sub) == 0:
                continue

            # -------------------------------------------------------------
            # Individual absolute velocity errors
            # -------------------------------------------------------------
            plt.figure(figsize=(8.5, 5.5))
            plt.plot(
                sub["snr_resel"],
                sub["median_abs_dv_A"],
                marker="o",
                label="Disk A"
            )
            plt.plot(
                sub["snr_resel"],
                sub["median_abs_dv_B"],
                marker="o",
                label="Disk B"
            )
            plt.axhline(
                ABSOLUTE_VELOCITY_TOLERANCE,
                linestyle="--",
                linewidth=1,
                label="absolute-V tolerance"
            )

            plt.xlabel("S/N per KCWI resolution element")
            plt.ylabel("Median |V_rec - V_true| (km/s)")
            plt.title(
                "{} | {} | {} | {}".format(
                    target,
                    population,
                    template_case,
                    window_name
                )
            )
            plt.legend()
            plt.tight_layout()

            plt.savefig(
                OUTPUT_DIR /
                "velocity_error_{}_{}_{}_{}.png".format(
                    target,
                    population,
                    template_case,
                    window_name
                ),
                dpi=180,
            )
            plt.close()

            # -------------------------------------------------------------
            # DeltaV error versus common-mode velocity offset
            # -------------------------------------------------------------
            plt.figure(figsize=(8.5, 5.5))

            plt.plot(
                sub["snr_resel"],
                sub["median_abs_dsep"],
                marker="o",
                label="median |DeltaV_rec - DeltaV_true|"
            )
            plt.plot(
                sub["snr_resel"],
                sub["median_abs_common_offset"],
                marker="o",
                label="median common-mode velocity offset"
            )

            plt.axhline(
                RELATIVE_SEPARATION_TOLERANCE,
                linestyle="--",
                linewidth=1,
                label="relative-DeltaV tolerance"
            )

            plt.xlabel("S/N per KCWI resolution element")
            plt.ylabel("Median velocity error (km/s)")
            plt.title(
                "{} | {} | {} | {}\n"
                "separation error vs common zero-point offset".format(
                    target,
                    population,
                    template_case,
                    window_name
                )
            )
            plt.legend()
            plt.tight_layout()

            plt.savefig(
                OUTPUT_DIR /
                "separation_vs_common_offset_{}_{}_{}_{}.png".format(
                    target,
                    population,
                    template_case,
                    window_name
                ),
                dpi=180,
            )
            plt.close()


def save_example_chi2_maps(
        target_name=None,
        pop_name="identical_population",
        template_case="matched_control"):
    """
    Save one example Delta-chi2 velocity surface at each S/N/window for one
    chosen template case.
    """
    if target_name is None:
        target_name = TARGET_NAMES_TO_RUN[0]
    target = TARGETS[target_name]

    noiseless, _, _, _ = make_two_component_noiseless(
        pop_name,
        target["v_A"],
        target["v_B"],
        target["sigma_A"],
        target["sigma_B"],
        target["frac_A"],
    )

    rng = np.random.RandomState(RANDOM_SEED + 9999)

    for window_name in FIT_WINDOWS:
        goodpixels = goodpixels_for_window(window_name)

        for snr in SNR_RESEL_VALUES:
            galaxy, noise = add_noise(noiseless, snr, rng)

            best, v_grid, chi2_map = brute_force_velocity_grid(
                galaxy,
                noise,
                goodpixels,
                template_case=template_case,
                return_map=True
            )

            dchi2 = chi2_map - np.nanmin(chi2_map)

            plt.figure(figsize=(6.5, 5.5))
            extent = [
                v_grid[0] - VEL_GRID_STEP/2,
                v_grid[-1] + VEL_GRID_STEP/2,
                v_grid[0] - VEL_GRID_STEP/2,
                v_grid[-1] + VEL_GRID_STEP/2,
            ]

            im = plt.imshow(
                dchi2,
                origin="lower",
                extent=extent,
                aspect="equal",
                interpolation="nearest",
            )
            plt.colorbar(im, label="Delta chi^2")

            plt.plot(
                target["v_A"],
                target["v_B"],
                marker="*",
                markersize=14,
                linestyle="None",
                label="Injected (A,B)"
            )
            plt.plot(
                target["v_B"],
                target["v_A"],
                marker="*",
                markersize=14,
                linestyle="None",
                label="Injected swapped"
            )
            plt.plot(
                [VEL_GRID_MIN, VEL_GRID_MAX],
                [VEL_GRID_MIN, VEL_GRID_MAX],
                linestyle="--",
                linewidth=1,
                label="V1 = V2"
            )

            plt.xlabel("Component 1 velocity (km/s)")
            plt.ylabel("Component 2 velocity (km/s)")
            plt.title(
                "{} | {} | {} | S/N(resel)={:.0f}".format(
                    target_name,
                    template_case,
                    window_name,
                    snr
                )
            )
            plt.legend(fontsize=8)
            plt.tight_layout()

            plt.savefig(
                OUTPUT_DIR /
                "chi2map_{}_{}_{}_sn{:03d}.png".format(
                    target_name,
                    template_case,
                    window_name,
                    int(snr)
                ),
                dpi=180,
            )
            plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("KCWI/KCRM CRD injection/recovery simulation")
    print("=================================================")
    print("Target: {}".format(TARGET_NAME))
    print("Target redshift (metadata): {:.6f}".format(TARGET_REDSHIFT))
    print("Injected V_A, V_B: {:+.1f}, {:+.1f} km/s".format(TARGET_V_A, TARGET_V_B))
    print("Injected DeltaV: {:.1f} km/s".format(abs(TARGET_V_A - TARGET_V_B)))
    print("Injected sigma_A, sigma_B: {:.1f}, {:.1f} km/s".format(
        TARGET_SIGMA_A, TARGET_SIGMA_B
    ))
    print("Injected f_A, f_B: {:.2f}, {:.2f}".format(TARGET_FRAC_A, 1.0-TARGET_FRAC_A))
    print("Slicer: {}".format(SLICER))
    print("Instrument setup: {}".format(INSTRUMENT_LABEL))
    print("Nominal/conservative R: {:.0f}".format(INSTRUMENT_R_NOMINAL))
    print("Instrument FWHM used: {:.3f} A".format(FWHM_GAL_A))
    print("Spectral pixel: {:.3f} A/pix".format(KCWI_LINEAR_PIXEL_A))
    print("Processes: {}".format(N_PROCESSES))
    print("S/N per resel tested: {}".format(SNR_RESEL_VALUES.tolist()))
    print("Template cases: {}".format(TEMPLATE_CASES_TO_RUN))
    print("Approx S/N per pixel: {}".format(
        [round(snr_resel_to_pixel(x), 1) for x in SNR_RESEL_VALUES]
    ))
    print("Velocity grid: {:.0f} to {:.0f} km/s in {:.0f} km/s cells".format(
        VEL_GRID_MIN, VEL_GRID_MAX, VEL_GRID_STEP
    ))

    # Target-specific safety checks for reused simulations. These do not change
    # any fits; they only warn when the unchanged recovery criteria/grid are a
    # poor match to the manually entered target.
    for _target_name in TARGET_NAMES_TO_RUN:
        _target = TARGETS[_target_name]
        _sep = abs(float(_target["v_A"]) - float(_target["v_B"]))
        _half = 0.5 * VEL_GRID_STEP
        if (
            min(float(_target["v_A"]), float(_target["v_B"])) < VEL_GRID_MIN - _half
            or max(float(_target["v_A"]), float(_target["v_B"])) > VEL_GRID_MAX + _half
        ):
            warnings.warn(
                "Target {} has V_A/V_B outside the current velocity grid; "
                "expand VEL_GRID_MIN/MAX before interpreting recovery.".format(_target_name)
            )
        if _sep < MIN_RECOVERED_SEPARATION:
            warnings.warn(
                "Target {} has injected DeltaV={:.1f} km/s, below the current "
                "MIN_RECOVERED_SEPARATION={:.1f} km/s success criterion.".format(
                    _target_name, _sep, MIN_RECOVERED_SEPARATION
                )
            )

    n_v = len(np.arange(VEL_GRID_MIN, VEL_GRID_MAX + 0.5*VEL_GRID_STEP, VEL_GRID_STEP))
    n_cells = n_v * n_v
    n_null_jobs = (
        len(TEMPLATE_CASES_TO_RUN) *
        len(FIT_WINDOWS) *
        len(SNR_RESEL_VALUES) *
        N_MC_NULL
    )
    n_crd_jobs = (
        len(build_crd_scenarios()) *
        len(POPULATION_NAMES_TO_RUN) *
        len(TEMPLATE_CASES_TO_RUN) *
        len(FIT_WINDOWS) *
        len(SNR_RESEL_VALUES) *
        N_MC_CRD
    )
    n_extra_jobs = (
        len(FIT_WINDOWS) * len(SNR_RESEL_VALUES)
        if RUN_EXAMPLE_CHI2_MAPS else 0
    )
    print("Grid cells per realization: {}".format(n_cells))
    print("Null realizations:          {}".format(n_null_jobs))
    print("CRD realizations:           {}".format(n_crd_jobs))
    print("Approx two-component fits:  {:,}".format((n_null_jobs + n_crd_jobs + n_extra_jobs) * n_cells))

    prepare_templates()

    # 1. Null / false-positive calibration.
    null_df, thresholds = run_null_experiment()

    # 2. True CRD injection/recovery.
    recovery_df = run_crd_experiment(thresholds)

    # 3. Summaries and recommended S/N.
    summary, rec_df = summarize_recovery(recovery_df)

    # 4. Diagnostic plots.
    plot_recovery_curves(summary)
    plot_velocity_errors(summary)

    # 5. Optional example velocity-likelihood maps.
    # Disabled for the pilot because they add an extra 289 fits for every
    # S/N/window combination. Turn on after the recovery transition is known.
    if RUN_EXAMPLE_CHI2_MAPS:
        save_example_chi2_maps(
            target_name=TARGET_NAMES_TO_RUN[0],
            pop_name=POPULATION_NAMES_TO_RUN[0],
            template_case="matched_control",
        )

    print("\nFinished.")
    print("Results written to: {}".format(OUTPUT_DIR.resolve()))
    print("Most important files:")
    print("  recommended_snr.csv")
    print("  recovery_summary.csv")
    print("  null_delta_chi2_thresholds.csv")
    print("  recovery_*.png")
    if RUN_EXAMPLE_CHI2_MAPS:
        print("  chi2map_*.png")


if __name__ == "__main__":
    main()
