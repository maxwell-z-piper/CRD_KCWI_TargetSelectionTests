#!/usr/bin/env python
# coding: utf-8

"""
KCRM RM2 + RH3 CRD INJECTION / RECOVERY WRAPPER
================================================

This script piggy-backs on:

    KCWI_CRD_injection_recovery_refined_configurable.py

and reruns the same Mitzkus-style two-component injection/recovery experiment
for the KCRM red arm using the slicer selected in target_config.txt with RM2 and/or RH3.

It can run BOTH versions of the experiment we have already developed:

    1. free_fraction
       Component light fractions are free, as in the refined baseline script.

    2. fixed_50_50
       The total stellar-template weights of the two components are forced to
       f_A = f_B = 0.50, as in the fixed-light-fraction diagnostic.

The script also adds a NEW velocity-dispersion recovery summary, because for
RM2/RH3 we specifically want to know whether the red arm can provide clean
V_A(x,y), V_B(x,y), sigma_A(x,y), and sigma_B(x,y) maps.

IMPORTANT MODELING NOTE
-----------------------
The synthetic spectra are kept in the galaxy REST FRAME. Instrumental pixel
sizes are therefore divided by (1+z), while resolving power R is unchanged.
The adopted constant FWHM is evaluated near the CaT science region. This is an
idealized information-content experiment: it still uses white Gaussian noise
and does NOT yet inject OH sky residuals, telluric absorption, covariance, or a
wavelength-dependent KCRM LSF.

Put this file in the same directory as:

    KCWI_CRD_injection_recovery_refined_configurable.py

Outputs are written into separate directories for every setup/fraction mode.
"""

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import KCWI_CRD_injection_recovery_refined_configfile as base


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


# We plan to use RH3. RM2 remains available by adding it back to this list.
RED_SETUPS_TO_RUN = ["RH3"]

# Slicer properties relevant to the spectral simulation. KCWI/KCRM is
# slit-width limited: Medium has half Small's R; Large has one quarter.
# Normal detector operation is 1x1 for Small and 2x2 for Medium/Large.
SLICER_CONFIGS = {
    "Small": {"resolution_factor": 1.00, "detector_binning": 1},
    "Medium": {"resolution_factor": 0.50, "detector_binning": 2},
    "Large": {"resolution_factor": 0.25, "detector_binning": 2},
}

_slicer_lookup = {key.lower(): key for key in SLICER_CONFIGS}
_slicer_key = _slicer_lookup.get(str(SLICER).strip().lower())
if _slicer_key is None:
    raise ValueError(
        "Unknown SLICER={!r}. Use 'Small', 'Medium', or 'Large'.".format(SLICER)
    )
SLICER = _slicer_key

# Replace the imported base script's target with the shared-config target.
base.TARGETS = {
    TARGET_NAME: {
        "v_A": float(TARGET_V_A),
        "v_B": float(TARGET_V_B),
        "sigma_A": float(TARGET_SIGMA_A),
        "sigma_B": float(TARGET_SIGMA_B),
        "frac_A": float(TARGET_FRAC_A),
    }
}
base.TARGET_NAMES_TO_RUN = [TARGET_NAME]
base.TARGET_NAME = TARGET_NAME
base.TARGET_REDSHIFT = Z_TARGET
base.TARGET_V_A = TARGET_V_A
base.TARGET_V_B = TARGET_V_B
base.TARGET_SIGMA_A = TARGET_SIGMA_A
base.TARGET_SIGMA_B = TARGET_SIGMA_B
base.TARGET_FRAC_A = TARGET_FRAC_A

TARGET_TAG = "".join(c if (c.isalnum() or c in "-_") else "_" for c in TARGET_NAME)

# Run the original free-light-fraction experiment and/or the fixed-50/50 test.
# For a faster first pass, change this to ["free_fraction"].
FRACTION_MODES_TO_RUN = ["free_fraction", "fixed_50_50"]

# Use the matched-control basis for the instrument comparison. This removes
# stellar-template mismatch as a confounding variable. The previous blue test
# showed almost no difference between matched and sparse-mismatched cases.
TEMPLATE_CASES_TO_RUN = ["matched_control"]

# The red arm should plausibly reach its transition at lower S/N than BL, so
# extend the grid downward while retaining 30--50 for direct comparison.
# S/N is PER SPECTRAL RESOLUTION ELEMENT.
SNR_RESEL_VALUES = np.array([10, 15, 20, 25, 30, 35, 40, 50], dtype=float)

# A useful first red-arm run. Increase to 50/100 after locating the transition.
N_MC_NULL = 20
N_MC_CRD = 50

# Set True for the same larger Monte Carlo statistics as the refined blue run.
RUN_HIGH_STATISTICS = False
if RUN_HIGH_STATISTICS:
    N_MC_NULL = 50
    N_MC_CRD = 100

# Same three-process setup as before.
N_PROCESSES = 3

# Fixed-fraction diagnostic value.
FIXED_FRACTION_A = float(TARGET_FRAC_A)

# Do not add the optional expensive example chi2 maps during this comparison.
RUN_EXAMPLE_CHI2_MAPS = False

# Output parent directory.
OUTPUT_PARENT = Path(
    "KCRM_RM2_RH3_injection_recovery_results_{}_{}".format(
        TARGET_TAG, SLICER
    )
)


# =============================================================================
# KCRM RED CONFIGURATIONS
# =============================================================================
#
# Small-slicer lab specifications are the reference values below; R is scaled by slicer:
#
# RM2:
#   dispersion = 0.50 A / unbinned detector pixel
#   R > 5600
#   instantaneous coverage ~1750--2000 A
#
# RH3:
#   dispersion = 0.223 A / unbinned detector pixel
#   R > 13000
#   instantaneous coverage ~750--850 A
#
# R_small values are conservative Small-slicer reference values; Medium and Large
# are scaled to one-half and one-quarter of Small, respectively.
#
# The fitting windows are REST FRAME science windows, chosen to emphasize the
# strongest old/intermediate-age stellar kinematic information in each setup.
#
# RM2 window:
#   broad red kinematic region including Na I ~8190, the Ca triplet, Mg I 8807,
#   and numerous weaker Fe/Ti features. A central wavelength around ~8200 A
#   observed should place this region comfortably on RM2 for IC25; confirm the
#   exact detector edges with the KCRM configuration tool before observing.
#
# RH3 window:
#   CaT-focused high-resolution region. The official example central wavelength
#   9040 A gives ~8630--9380 A observed, corresponding to ~8465--9201 A in the
#   IC25 rest frame, so the 8470--8900 A fitting interval is safely inside it.
# =============================================================================

RED_CONFIGS = {
    "RM2": {
        "R_small": 5600.0,
        "dispersion_obs_A_per_unbinned_pix": 0.50,
        "lambda_ref_rest_A": 8650.0,
        "wave_gal_min_rest_A": 7850.0,
        "wave_gal_max_rest_A": 9000.0,
        "fit_window_name": "RM2_red_kinematics",
        "fit_window_rest_A": (7900.0, 8950.0),
    },
    "RH3": {
        "R_small": 13000.0,
        "dispersion_obs_A_per_unbinned_pix": 0.223,
        "lambda_ref_rest_A": 8650.0,
        "wave_gal_min_rest_A": 8450.0,
        "wave_gal_max_rest_A": 9000.0,
        "fit_window_name": "RH3_CaT_kinematics",
        "fit_window_rest_A": (8470.0, 8900.0),
    },
}


# =============================================================================
# SAVE ORIGINAL BASELINE FUNCTION BEFORE MONKEY-PATCHING
# =============================================================================

ORIGINAL_TWO_COMPONENT_CELL_FIT = base.two_component_cell_fit




# =============================================================================
# RED-WAVELENGTH NORMALIZATION / SYNTHETIC SPECTRUM PATCHES
# =============================================================================
#
# The blue baseline script normalizes its synthetic spectra in 5000--5350 A.
# That interval is not present in the red-arm simulations, so we replace the
# normalization helpers with red-safe versions that use the active fitting
# window. This is essential when reusing the blue script at CaT wavelengths.
# =============================================================================

def normalize_template_red(template, lam, norm_range=None):
    """Normalize a red-arm template to median unity in the active fit window."""
    template = np.asarray(template, dtype=float)
    lam = np.asarray(lam, dtype=float)

    if norm_range is None:
        # There is only one active fit window in each red configuration.
        lo, hi = list(base.FIT_WINDOWS.values())[0]
    else:
        lo, hi = norm_range

    use = (lam >= lo) & (lam <= hi) & np.isfinite(template)
    if np.sum(use) < 10:
        use = np.isfinite(template)

    med = np.nanmedian(template[use])
    if not np.isfinite(med) or med == 0:
        raise ValueError("Red-arm template normalization failed.")
    return template / med


def make_two_component_noiseless_red(pop_name, v_A, v_B, sigma_A, sigma_B, frac_A):
    """Red-safe version of the baseline two-component spectrum generator."""
    scenario = base.POPULATION_SCENARIOS[pop_name]
    sps = base.G["sps"]
    keep = base.G["keep_temp"]

    temp_A_all, age_A_actual, metal_A_actual = base.get_ssp_template(
        sps, scenario["age_A"], scenario["metal_A"]
    )
    temp_B_all, age_B_actual, metal_B_actual = base.get_ssp_template(
        sps, scenario["age_B"], scenario["metal_B"]
    )

    temp_A = normalize_template_red(temp_A_all[keep], base.G["lam_temp"])
    temp_B = normalize_template_red(temp_B_all[keep], base.G["lam_temp"])

    A_pad = base.apply_gaussian_losvd(
        temp_A, base.G["loglam_temp"], v_A, sigma_A, base.G["velscale"]
    )
    B_pad = base.apply_gaussian_losvd(
        temp_B, base.G["loglam_temp"], v_B, sigma_B, base.G["velscale"]
    )

    A = base.interpolate_to_galaxy_grid(A_pad)
    B = base.interpolate_to_galaxy_grid(B_pad)
    model = frac_A * A + (1.0 - frac_A) * B

    lo, hi = list(base.FIT_WINDOWS.values())[0]
    norm = (base.G["lam_gal"] >= lo) & (base.G["lam_gal"] <= hi)
    scale = np.nanmedian(model[norm])
    if not np.isfinite(scale) or scale == 0:
        raise ValueError("Red-arm combined-spectrum normalization failed.")

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


def make_one_component_noiseless_red():
    """Red-safe version of the one-component null-spectrum generator."""
    sps = base.G["sps"]
    keep = base.G["keep_temp"]

    temp_all, age_actual, metal_actual = base.get_ssp_template(
        sps, base.NULL_AGE, base.NULL_METAL
    )
    temp = normalize_template_red(temp_all[keep], base.G["lam_temp"])

    model_pad = base.apply_gaussian_losvd(
        temp, base.G["loglam_temp"],
        base.NULL_V, base.NULL_SIGMA, base.G["velscale"]
    )
    model = base.interpolate_to_galaxy_grid(model_pad)

    lo, hi = list(base.FIT_WINDOWS.values())[0]
    norm = (base.G["lam_gal"] >= lo) & (base.G["lam_gal"] <= hi)
    scale = np.nanmedian(model[norm])
    if not np.isfinite(scale) or scale == 0:
        raise ValueError("Red-arm null-spectrum normalization failed.")
    model /= scale

    return model, age_actual, metal_actual


# =============================================================================
# FIXED-FRACTION TWO-COMPONENT CELL FIT
# =============================================================================

def two_component_cell_fit_fixed_fraction(
        galaxy, noise, goodpixels, v1_start, v2_start, template_case):
    """
    Same Mitzkus-style cell fit as the refined script, except the total
    template-weight fraction of component 0 is fixed to FIXED_FRACTION_A.

    V1, V2, sigma1, sigma2, and the SSP mixture within each component remain
    free. This is the red-arm equivalent of the fixed-light-fraction test.
    """
    fit_case = base.G["fit_cases"][template_case]

    scale = np.nanmedian(galaxy[goodpixels])
    gal = galaxy / scale
    err = noise / scale

    half = 0.5 * base.VEL_GRID_STEP
    start = [[v1_start, base.SIGMA_START], [v2_start, base.SIGMA_START]]
    bounds = [
        [[v1_start - half, v1_start + half], [base.SIGMA_MIN, base.SIGMA_MAX]],
        [[v2_start - half, v2_start + half], [base.SIGMA_MIN, base.SIGMA_MAX]],
    ]

    pp = base.ppxf(
        fit_case["templates_fit"],
        gal,
        err,
        base.G["velscale"],
        start,
        moments=[2, 2],
        component=fit_case["component_fit"],
        fraction=FIXED_FRACTION_A,
        bounds=bounds,
        goodpixels=goodpixels,
        degree=base.ADEGREE,
        mdegree=base.MDEGREE,
        lam=base.G["lam_gal"],
        lam_temp=base.G["lam_temp"],
        quiet=True,
    )

    chi2_total = np.sum(
        ((gal[goodpixels] - pp.bestfit[goodpixels]) / err[goodpixels])**2
    )

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


# =============================================================================
# CONFIGURE ONE RED SETUP
# =============================================================================

def configure_red_setup(setup_name, fraction_mode):
    """Overwrite only the globals that differ from the blue refined run."""
    cfg = RED_CONFIGS[setup_name]
    slicer_cfg = SLICER_CONFIGS[SLICER]

    # Apply the slit-width resolution scaling for the selected slicer.
    resolving_power = (
        float(cfg["R_small"])
        * float(slicer_cfg["resolution_factor"])
    )

    # Conservative constant FWHM evaluated near the CaT region.
    # R is invariant to redshift, so lambda_rest / R gives rest-frame FWHM.
    fwhm_rest = cfg["lambda_ref_rest_A"] / resolving_power

    # The tabulated dispersion is per UNBINNED detector pixel. Medium/Large are
    # normally used 2x2, so convert to the effective observed A/pixel first,
    # then divide by (1+z) because the synthetic spectra are rest-frame.
    detector_binning = int(slicer_cfg["detector_binning"])
    pixel_obs = (
        float(cfg["dispersion_obs_A_per_unbinned_pix"])
        * detector_binning
    )
    pixel_rest = pixel_obs / (1.0 + Z_TARGET)

    # Approximate number of detector pixels per spectral resolution element.
    pixels_per_resel = fwhm_rest / pixel_rest

    base.SLICER = SLICER
    base.INSTRUMENT_LABEL = setup_name
    base.INSTRUMENT_R_NOMINAL = float(resolving_power)
    base.FWHM_GAL_A = float(fwhm_rest)
    base.KCWI_LINEAR_PIXEL_A = float(pixel_rest)
    base.PIXELS_PER_RESEL = float(pixels_per_resel)

    base.WAVE_GAL_MIN = float(cfg["wave_gal_min_rest_A"])
    base.WAVE_GAL_MAX = float(cfg["wave_gal_max_rest_A"])
    base.WAVE_TEMPLATE_PAD = 150.0

    base.FIT_WINDOWS = {
        cfg["fit_window_name"]: tuple(cfg["fit_window_rest_A"])
    }

    # No nebular-line masks are needed for this pure-stellar injection test.
    # A later realism test should add telluric/OH masks or, preferably, inject
    # wavelength-dependent noise/sky residuals from the KCRM ETC/real data.
    base.MASK_REGIONS_A = []

    base.SNR_RESEL_VALUES = SNR_RESEL_VALUES.copy()
    base.N_MC_NULL = int(N_MC_NULL)
    base.N_MC_CRD = int(N_MC_CRD)
    base.N_PROCESSES = int(N_PROCESSES)

    base.TARGET_NAMES_TO_RUN = [TARGET_NAME]
    base.POPULATION_NAMES_TO_RUN = ["identical_population"]
    base.TEMPLATE_CASES_TO_RUN = list(TEMPLATE_CASES_TO_RUN)
    base.RUN_STRESS_TEST = False
    base.RUN_EXAMPLE_CHI2_MAPS = bool(RUN_EXAMPLE_CHI2_MAPS)

    # Preserve the same velocity search and recovery definitions.
    base.VEL_GRID_MIN = -160.0
    base.VEL_GRID_MAX = +160.0
    base.VEL_GRID_STEP = 20.0
    base.SIGMA_START = 60.0
    base.SIGMA_MIN = 5.0
    base.SIGMA_MAX = 180.0
    base.ADEGREE = 4
    base.MDEGREE = 0

    base.ABSOLUTE_VELOCITY_TOLERANCE = 20.0
    base.DETECTION_SEPARATION_TOLERANCE = 25.0
    base.RELATIVE_SEPARATION_TOLERANCE = 15.0
    base.MIN_COMPONENT_LIGHT = 0.15
    base.MIN_RECOVERED_SEPARATION = 50.0
    base.NULL_PERCENTILE = 95.0

    # Select free-fraction or fixed-50/50 fitting.
    if fraction_mode == "free_fraction":
        base.two_component_cell_fit = ORIGINAL_TWO_COMPONENT_CELL_FIT
    elif fraction_mode == "fixed_50_50":
        base.two_component_cell_fit = two_component_cell_fit_fixed_fraction
    else:
        raise ValueError("Unknown fraction mode: {}".format(fraction_mode))

    # Separate output directory for every combination.
    base.OUTPUT_DIR = (
        OUTPUT_PARENT /
        "{}_{}".format(setup_name, fraction_mode)
    )

    # Patch the blue script's hard-coded 5000--5350 A normalization so the
    # synthetic red spectra are normalized inside the active RM2/RH3 window.
    base.normalize_template = normalize_template_red
    base.make_two_component_noiseless = make_two_component_noiseless_red
    base.make_one_component_noiseless = make_one_component_noiseless_red

    # Clear previously prepared wavelength/template state before preparing the
    # next grating. base.main() will rebuild everything using these new globals.
    base.G.clear()

    return {
        "setup": setup_name,
        "slicer": SLICER,
        "detector_binning": detector_binning,
        "fraction_mode": fraction_mode,
        "R": resolving_power,
        "fwhm_rest_A": fwhm_rest,
        "pixel_rest_A": pixel_rest,
        "pixels_per_resel": pixels_per_resel,
        "fit_window_name": cfg["fit_window_name"],
        "fit_lo_A": cfg["fit_window_rest_A"][0],
        "fit_hi_A": cfg["fit_window_rest_A"][1],
        "output_dir": str(base.OUTPUT_DIR),
    }


# =============================================================================
# ADD SIGMA-RECOVERY SUMMARY
# =============================================================================

def make_sigma_summary(output_dir, setup_name, fraction_mode):
    """
    The original refined script summarizes velocity recovery but not sigma
    recovery. Add the quantities we need to judge whether RM2/RH3 can make
    clean component velocity-dispersion maps.
    """
    output_dir = Path(output_dir)
    path = output_dir / "crd_recovery_realizations.csv"
    df = pd.read_csv(path)

    df["dsigma_A"] = df["sigma_A_rec"] - df["sigma_A_true"]
    df["dsigma_B"] = df["sigma_B_rec"] - df["sigma_B_true"]
    df["abs_dsigma_A"] = np.abs(df["dsigma_A"])
    df["abs_dsigma_B"] = np.abs(df["dsigma_B"])

    df["sigma_both_within_10"] = (
        (df["abs_dsigma_A"] <= 10.0) &
        (df["abs_dsigma_B"] <= 10.0)
    )
    df["sigma_both_within_20"] = (
        (df["abs_dsigma_A"] <= 20.0) &
        (df["abs_dsigma_B"] <= 20.0)
    )

    group_cols = ["template_case", "window", "snr_resel"]
    s = df.groupby(group_cols).agg(
        n=("snr_resel", "size"),
        median_abs_dsigma_A=("abs_dsigma_A", "median"),
        median_abs_dsigma_B=("abs_dsigma_B", "median"),
        p84_abs_dsigma_A=("abs_dsigma_A", lambda x: np.nanpercentile(x, 84)),
        p84_abs_dsigma_B=("abs_dsigma_B", lambda x: np.nanpercentile(x, 84)),
        sigma_both_within_10_fraction=("sigma_both_within_10", "mean"),
        sigma_both_within_20_fraction=("sigma_both_within_20", "mean"),
        median_sigma_A_rec=("sigma_A_rec", "median"),
        median_sigma_B_rec=("sigma_B_rec", "median"),
    ).reset_index()

    s.insert(0, "red_setup", setup_name)
    s.insert(1, "fraction_mode", fraction_mode)
    s.to_csv(output_dir / "sigma_recovery_summary.csv", index=False)

    # One simple diagnostic plot per run.
    plt.figure(figsize=(8.5, 5.5))
    plt.plot(s["snr_resel"], s["median_abs_dsigma_A"], marker="o", label="Disk A")
    plt.plot(s["snr_resel"], s["median_abs_dsigma_B"], marker="o", label="Disk B")
    plt.axhline(10.0, linestyle="--", linewidth=1, label="10 km/s error")
    plt.axhline(20.0, linestyle=":", linewidth=1, label="20 km/s error")
    plt.xlabel("S/N per KCRM resolution element")
    plt.ylabel("Median |sigma_rec - sigma_true| (km/s)")
    plt.title("{} | {} | velocity-dispersion recovery".format(
        setup_name, fraction_mode
    ))
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "sigma_velocity_error.png", dpi=180)
    plt.close()

    return s


# =============================================================================
# RUN ONE SETUP / FRACTION MODE
# =============================================================================

def run_one(setup_name, fraction_mode):
    meta = configure_red_setup(setup_name, fraction_mode)

    print("\n" + "#" * 78)
    print("KCRM RED TEST: {} | {}".format(setup_name, fraction_mode))
    print("#" * 78)
    print("R used:                    {:.0f}".format(meta["R"]))
    print("Rest-frame FWHM used:      {:.4f} A".format(meta["fwhm_rest_A"]))
    print("Rest-frame pixel size:     {:.4f} A/pix".format(meta["pixel_rest_A"]))
    print("Pixels per resel:          {:.2f}".format(meta["pixels_per_resel"]))
    print("Rest-frame fitting window: {:.0f}-{:.0f} A".format(
        meta["fit_lo_A"], meta["fit_hi_A"]
    ))
    print("S/N grid:                  {}".format(SNR_RESEL_VALUES.tolist()))
    print("N null / S/N:              {}".format(N_MC_NULL))
    print("N CRD / S/N:               {}".format(N_MC_CRD))
    print("Output:                    {}".format(meta["output_dir"]))

    # The null experiment MUST be rerun for each grating/fraction model because
    # the LSF, spectral sampling, wavelength range, and model flexibility change.
    base.main()

    output_dir = Path(meta["output_dir"])

    recovery = pd.read_csv(output_dir / "recovery_summary.csv")
    recovery.insert(0, "red_setup", setup_name)
    recovery.insert(1, "fraction_mode", fraction_mode)
    recovery.insert(2, "R_used", meta["R"])
    recovery.insert(3, "FWHM_rest_A_used", meta["fwhm_rest_A"])
    recovery.insert(4, "pixel_rest_A_used", meta["pixel_rest_A"])
    recovery.to_csv(output_dir / "recovery_summary_with_setup.csv", index=False)

    sigma = make_sigma_summary(output_dir, setup_name, fraction_mode)

    return meta, recovery, sigma


# =============================================================================
# MAIN + COMBINED OUTPUTS
# =============================================================================

def main():
    OUTPUT_PARENT.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("KCRM RED-ARM INJECTION / RECOVERY")
    print("=" * 78)
    print("Target: {}".format(TARGET_NAME))
    print("Slicer: {}".format(SLICER))
    print("Red setup(s): {}".format(RED_SETUPS_TO_RUN))
    print("Injected V_A, V_B: {:+.1f}, {:+.1f} km/s".format(TARGET_V_A, TARGET_V_B))
    print("Injected DeltaV: {:.1f} km/s".format(abs(TARGET_V_A - TARGET_V_B)))
    print("Injected sigma_A, sigma_B: {:.1f}, {:.1f} km/s".format(
        TARGET_SIGMA_A, TARGET_SIGMA_B
    ))

    all_meta = []
    all_recovery = []
    all_sigma = []

    for setup_name in RED_SETUPS_TO_RUN:
        for fraction_mode in FRACTION_MODES_TO_RUN:
            meta, recovery, sigma = run_one(setup_name, fraction_mode)
            all_meta.append(meta)
            all_recovery.append(recovery)
            all_sigma.append(sigma)

    meta_df = pd.DataFrame(all_meta)
    meta_df.to_csv(OUTPUT_PARENT / "red_setup_metadata.csv", index=False)

    recovery_df = pd.concat(all_recovery, ignore_index=True)
    recovery_df.to_csv(OUTPUT_PARENT / "combined_recovery_summary.csv", index=False)

    sigma_df = pd.concat(all_sigma, ignore_index=True)
    sigma_df.to_csv(OUTPUT_PARENT / "combined_sigma_recovery_summary.csv", index=False)

    # -------------------------------------------------------------------------
    # Combined absolute-velocity recovery comparison
    # -------------------------------------------------------------------------
    plt.figure(figsize=(9.0, 6.0))
    for (setup, fracmode), sub in recovery_df.groupby(["red_setup", "fraction_mode"]):
        sub = sub.sort_values("snr_resel")
        plt.plot(
            sub["snr_resel"],
            sub["absolute_kinematic_fraction"],
            marker="o",
            label="{} | {}".format(setup, fracmode),
        )
    plt.axhline(0.90, linestyle="--", linewidth=1, label="90%")
    plt.ylim(-0.02, 1.02)
    plt.xlabel("S/N per KCRM resolution element")
    plt.ylabel("Absolute V_A/V_B recovery fraction")
    plt.title("KCRM red-arm: absolute two-disk velocity recovery")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_PARENT / "combined_absolute_velocity_recovery.png", dpi=180)
    plt.close()

    # -------------------------------------------------------------------------
    # Combined DeltaV recovery comparison
    # -------------------------------------------------------------------------
    plt.figure(figsize=(9.0, 6.0))
    for (setup, fracmode), sub in recovery_df.groupby(["red_setup", "fraction_mode"]):
        sub = sub.sort_values("snr_resel")
        plt.plot(
            sub["snr_resel"],
            sub["relative_kinematic_fraction"],
            marker="o",
            label="{} | {}".format(setup, fracmode),
        )
    plt.axhline(0.90, linestyle="--", linewidth=1, label="90%")
    plt.ylim(-0.02, 1.02)
    plt.xlabel("S/N per KCRM resolution element")
    plt.ylabel("Relative DeltaV recovery fraction")
    plt.title("KCRM red-arm: LOSVD-separation recovery")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_PARENT / "combined_relative_velocity_recovery.png", dpi=180)
    plt.close()

    # -------------------------------------------------------------------------
    # Combined sigma recovery comparison
    # -------------------------------------------------------------------------
    plt.figure(figsize=(9.0, 6.0))
    for (setup, fracmode), sub in sigma_df.groupby(["red_setup", "fraction_mode"]):
        sub = sub.sort_values("snr_resel")
        # Average the two median absolute sigma errors only for this quick plot.
        y = 0.5 * (sub["median_abs_dsigma_A"] + sub["median_abs_dsigma_B"])
        plt.plot(
            sub["snr_resel"], y,
            marker="o",
            label="{} | {}".format(setup, fracmode),
        )
    plt.axhline(10.0, linestyle="--", linewidth=1, label="10 km/s")
    plt.axhline(20.0, linestyle=":", linewidth=1, label="20 km/s")
    plt.xlabel("S/N per KCRM resolution element")
    plt.ylabel("Mean of median |sigma_rec - sigma_true| (km/s)")
    plt.title("KCRM red-arm: two-disk velocity-dispersion recovery")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_PARENT / "combined_sigma_velocity_error.png", dpi=180)
    plt.close()

    print("\nFinished all requested KCRM red-arm experiments.")
    print("Combined results written to: {}".format(OUTPUT_PARENT.resolve()))
    print("Most useful combined files:")
    print("  combined_recovery_summary.csv")
    print("  combined_sigma_recovery_summary.csv")
    print("  combined_absolute_velocity_recovery.png")
    print("  combined_relative_velocity_recovery.png")
    print("  combined_sigma_velocity_error.png")


if __name__ == "__main__":
    main()
