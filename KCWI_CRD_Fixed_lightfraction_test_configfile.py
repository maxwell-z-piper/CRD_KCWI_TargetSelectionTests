#!/usr/bin/env python
# coding: utf-8

"""
KCWI BL FIXED LIGHT-FRACTION TEST
=======================================

PURPOSE
-------
This script performs Test #1 of the KCWI CRD recovery experiment.

The previous injection/recovery experiment found that the separation between
the two stellar LOSVDs,

    DeltaV = |V_A - V_B|

was recovered very accurately, but the absolute velocities V_A and V_B were
often shifted together by ~15--20 km/s.

One possible explanation is a degeneracy between:

    component light fraction
        f_A, f_B

and

    common velocity midpoint
        (V_A + V_B) / 2.

For the default target configuration, the TRUE injected light fractions are:

    f_A = 0.50
    f_B = 0.50

This diagnostic test therefore fixes the total stellar-template weight of
component A to exactly 0.50 during every two-component pPXF fit.

The following remain FREE:

    V_A
    V_B
    sigma_A
    sigma_B

The SSP mixture WITHIN each component also remains free.

Everything else remains as close as possible to the previous refined run:

    - target properties are set manually in the USER SETTINGS block below

    - identical stellar populations in both disks

    - matched-control fitting basis
      (includes the exact XSL SSP used to generate the fake galaxy)

    - 4800--5500 A fitting region

    - S/N per KCWI resolution element:
          30, 35, 40, 45, 50

    - 50 one-component NULL realizations per S/N

    - 100 two-component CRD realizations per S/N

    - full 17 x 17 Mitzkus-style velocity search for every realization

    - empirical 95th-percentile NULL Delta-chi2 threshold

    - three success definitions:
          two-LOSVD detection
          relative DeltaV recovery
          absolute V_A/V_B recovery


IMPORTANT
---------
This script imports:

    KCWI_CRD_injection_recovery_refined.py

so that file must be in the same directory.

The original refined script is NOT modified.

All output from this experiment is written to:

    KCWI_CRD_fixed_fraction_test_results/


INTERPRETATION
--------------
If fixing f_A = 0.50 causes the absolute-velocity recovery fraction to increase
substantially relative to the previous matched-control result, that is strong
evidence that the previous common-mode velocity offsets were caused primarily
by the light-fraction / velocity-midpoint degeneracy.

For reference, the previous free-fraction matched-control absolute-velocity
recovery fractions were approximately:

    S/N 30 : 0.29
    S/N 35 : 0.34
    S/N 40 : 0.46
    S/N 45 : 0.54
    S/N 50 : 0.51

Those are the values we want to compare against.
"""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np
import pandas as pd

from pathlib import Path

# Import the complete refined analysis that we already tested.
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


# Apply the selected BL slicer model and target to the imported base script.
base.configure_bl_instrument(SLICER)
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
base.TARGET_REDSHIFT = TARGET_REDSHIFT
base.TARGET_V_A = TARGET_V_A
base.TARGET_V_B = TARGET_V_B
base.TARGET_SIGMA_A = TARGET_SIGMA_A
base.TARGET_SIGMA_B = TARGET_SIGMA_B
base.TARGET_FRAC_A = TARGET_FRAC_A

TARGET_TAG = "".join(
    c if (c.isalnum() or c in "-_") else "_" for c in TARGET_NAME
)

# =============================================================================
# CONFIGURATION FOR THIS EXPERIMENT
# =============================================================================

# -------------------------------------------------------------------------
# Use ONLY the matched-control template case.
#
# This basis contains the exact XSL SSP used to generate the synthetic
# galaxy, which means template mismatch is deliberately removed from this
# particular diagnostic experiment.
# -------------------------------------------------------------------------

base.TEMPLATE_CASES_TO_RUN = [
    "matched_control"
]


# -------------------------------------------------------------------------
# Use the same S/N range as the previous refined run so that the results
# can be compared directly point by point.
#
# S/N is PER KCWI RESOLUTION ELEMENT.
# -------------------------------------------------------------------------

base.SNR_RESEL_VALUES = np.array(
    [
        30,
        35,
        40,
        45,
        50,
    ],
    dtype=float
)


# -------------------------------------------------------------------------
# Use the same Monte Carlo statistics as the refined run.
# -------------------------------------------------------------------------

base.N_MC_NULL = 50
base.N_MC_CRD = 100


# -------------------------------------------------------------------------
# Keep the identical-population experiment for the manually selected target.
# -------------------------------------------------------------------------

base.TARGET_NAMES_TO_RUN = [TARGET_NAME]

base.POPULATION_NAMES_TO_RUN = [
    "identical_population"
]


# -------------------------------------------------------------------------
# Do NOT run the optional unequal-light-fraction or unequal-sigma stress
# tests yet.
#
# We want this experiment to differ from the previous one in ONE important
# respect only: whether the light fraction is fixed.
# -------------------------------------------------------------------------

base.RUN_STRESS_TEST = False


# -------------------------------------------------------------------------
# Keep the same wavelength interval.
#
# This contains:
#
#     Hbeta
#     Fe5015
#     Mg b
#     Fe5270
#     Fe5335
#
# and is the region we identified as the primary kinematic window.
# -------------------------------------------------------------------------

base.FIT_WINDOWS = {
    "red_kinematics": (
        4800.0,
        5500.0,
    ),
}


# -------------------------------------------------------------------------
# Keep the same full Mitzkus-style velocity search.
#
# -160 ... +160 km/s
# in 20 km/s cells
#
# gives:
#
#     17 x 17 = 289
#
# two-component pPXF fits for EVERY noisy realization.
# -------------------------------------------------------------------------

base.VEL_GRID_MIN = -160.0
base.VEL_GRID_MAX = +160.0
base.VEL_GRID_STEP = 20.0


# -------------------------------------------------------------------------
# Keep the same sigma freedom.
#
# IMPORTANT:
#
# We are NOT fixing sigma in this test.
#
# We are testing ONLY whether fixing the light fraction improves recovery
# of V_A and V_B.
# -------------------------------------------------------------------------

base.SIGMA_START = 60.0
base.SIGMA_MIN = 5.0
base.SIGMA_MAX = 180.0


# -------------------------------------------------------------------------
# Keep the same polynomial setup.
# -------------------------------------------------------------------------

base.ADEGREE = 4
base.MDEGREE = 0


# -------------------------------------------------------------------------
# Keep the same recovery thresholds.
# -------------------------------------------------------------------------

base.ABSOLUTE_VELOCITY_TOLERANCE = 20.0

base.DETECTION_SEPARATION_TOLERANCE = 25.0

base.RELATIVE_SEPARATION_TOLERANCE = 15.0

base.MIN_COMPONENT_LIGHT = 0.15

base.MIN_RECOVERED_SEPARATION = 50.0

base.NULL_PERCENTILE = 95.0


# -------------------------------------------------------------------------
# The actual diagnostic change.
# -------------------------------------------------------------------------

FIXED_FRACTION_A = float(TARGET_FRAC_A)


# -------------------------------------------------------------------------
# Use the same multiprocessing setting.
# -------------------------------------------------------------------------

base.N_PROCESSES = 3


# -------------------------------------------------------------------------
# Write these results somewhere NEW so the previous free-fraction results
# are preserved.
# -------------------------------------------------------------------------

base.OUTPUT_DIR = Path(
    "KCWI_CRD_fixed_fraction_test_results_{}_{}".format(
        TARGET_TAG, SLICER
    )
)


# -------------------------------------------------------------------------
# Keep the expensive additional example chi2 maps off initially.
# -------------------------------------------------------------------------

base.RUN_EXAMPLE_CHI2_MAPS = False


# =============================================================================
# REPLACEMENT TWO-COMPONENT CELL FIT
# =============================================================================

def two_component_cell_fit_fixed_fraction(
        galaxy,
        noise,
        goodpixels,
        v1_start,
        v2_start,
        template_case):
    """
    Fit ONE cell of the Mitzkus-style (V1,V2) velocity grid.

    THIS IS THE ONLY IMPORTANT FITTING CHANGE FROM THE PREVIOUS RUN.

    Previously
    ----------
    The relative total template weights of the two stellar components were
    free. pPXF could choose, for example:

        f_A = 0.35
        f_B = 0.65

    even though the simulated spectrum actually contained:

        f_A = 0.50
        f_B = 0.50.


    Now
    ---
    We impose:

        f_A = 0.50
        f_B = 0.50

    using pPXF's `fraction` keyword.


    Still free
    ----------
    For every velocity-grid cell, pPXF still optimizes:

        V_1
        sigma_1

        V_2
        sigma_2

    The SSP template mixture WITHIN each component is also still free.


    Velocity-grid behavior
    ----------------------
    As before, each grid point is not merely an initial guess.

    For a cell centered at:

        V1_start
        V2_start

    the two velocities can move only within:

        +/- VEL_GRID_STEP / 2

    around those cell centers.

    For the current 20 km/s grid:

        V1_start - 10 <= V1 <= V1_start + 10

        V2_start - 10 <= V2 <= V2_start + 10

    This preserves the global Mitzkus-style search and prevents every
    starting point from simply falling into the same local solution.
    """

    # Get the matched-control template basis prepared by the original script.
    fit_case = base.G[
        "fit_cases"
    ][
        template_case
    ]


    # =====================================================================
    # Normalize galaxy and noise together
    # =====================================================================

    scale = np.nanmedian(
        galaxy[
            goodpixels
        ]
    )

    gal = (
        galaxy
        /
        scale
    )

    err = (
        noise
        /
        scale
    )


    # =====================================================================
    # Bounds for this individual velocity cell
    # =====================================================================

    half = (
        0.5
        *
        base.VEL_GRID_STEP
    )


    # Starting LOSVDs.
    start = [

        [
            v1_start,
            base.SIGMA_START,
        ],

        [
            v2_start,
            base.SIGMA_START,
        ],

    ]


    # Each velocity stays inside its own grid cell.
    #
    # sigma remains completely free over SIGMA_MIN -> SIGMA_MAX.
    bounds = [

        [
            [
                v1_start - half,
                v1_start + half,
            ],

            [
                base.SIGMA_MIN,
                base.SIGMA_MAX,
            ],
        ],

        [
            [
                v2_start - half,
                v2_start + half,
            ],

            [
                base.SIGMA_MIN,
                base.SIGMA_MAX,
            ],
        ],

    ]


    # =====================================================================
    # TWO-COMPONENT pPXF FIT
    # =====================================================================
    #
    # The key new line is:
    #
    #       fraction = FIXED_FRACTION_A
    #
    # With two kinematic components, pPXF constrains:
    #
    #     sum(weights in component 0)
    #     ---------------------------
    #        sum(all stellar weights)
    #
    # to equal this number.
    #
    # Therefore:
    #
    #       fraction = 0.50
    #
    # requires equal total template weight in the two components.
    #
    # It does NOT dictate which SSPs the components use.
    # =====================================================================

    pp = base.ppxf(

        fit_case[
            "templates_fit"
        ],

        gal,

        err,

        base.G[
            "velscale"
        ],

        start,

        moments=[
            2,
            2,
        ],

        component=fit_case[
            "component_fit"
        ],

        # -------------------------------------------------------------
        # NEW CONSTRAINT
        # -------------------------------------------------------------
        fraction=FIXED_FRACTION_A,

        bounds=bounds,

        goodpixels=goodpixels,

        degree=base.ADEGREE,

        mdegree=base.MDEGREE,

        lam=base.G[
            "lam_gal"
        ],

        lam_temp=base.G[
            "lam_temp"
        ],

        quiet=True,
    )


    # =====================================================================
    # Calculate TOTAL chi2 over the same good pixels
    # =====================================================================

    chi2_total = np.sum(

        (

            (
                gal[
                    goodpixels
                ]

                -

                pp.bestfit[
                    goodpixels
                ]
            )

            /

            err[
                goodpixels
            ]

        )**2

    )


    # =====================================================================
    # Recover template-weight fractions
    # =====================================================================
    #
    # These should now be approximately:
    #
    #     0.50
    #     0.50
    #
    # We still calculate them explicitly as a sanity check.
    # =====================================================================

    w = np.asarray(
        pp.weights,
        dtype=float
    )


    n = fit_case[
        "n_basis"
    ]


    # Component 0 weights.
    w1 = np.sum(

        np.clip(

            w[
                :n
            ],

            0.0,

            None,

        )

    )


    # Component 1 weights.
    w2 = np.sum(

        np.clip(

            w[
                n:2*n
            ],

            0.0,

            None,

        )

    )


    wt = (
        w1
        +
        w2
    )


    if wt > 0:

        frac1 = (
            w1
            /
            wt
        )

        frac2 = (
            w2
            /
            wt
        )

    else:

        frac1 = np.nan
        frac2 = np.nan


    # =====================================================================
    # Return the exact same dictionary structure expected by the original
    # brute-force velocity-grid function.
    # =====================================================================

    return {

        "pp":
            pp,

        "chi2_total":
            float(
                chi2_total
            ),

        "v1":
            float(
                pp.sol[0][0]
            ),

        "sig1":
            float(
                pp.sol[0][1]
            ),

        "v2":
            float(
                pp.sol[1][0]
            ),

        "sig2":
            float(
                pp.sol[1][1]
            ),

        "frac1":
            float(
                frac1
            ),

        "frac2":
            float(
                frac2
            ),

    }


# =============================================================================
# REPLACE THE ORIGINAL CELL-FIT FUNCTION
# =============================================================================
#
# The existing:
#
#     brute_force_velocity_grid()
#
# calls:
#
#     two_component_cell_fit()
#
# every time it evaluates one of the 289 velocity cells.
#
# By replacing the function in the imported module here, every NULL and CRD
# realization automatically uses the fixed-50/50 model.
# =============================================================================

base.two_component_cell_fit = (
    two_component_cell_fit_fixed_fraction
)


# =============================================================================
# RUN THE EXPERIMENT
# =============================================================================

if __name__ == "__main__":

    print("")
    print("=" * 78)
    print("KCWI FIXED-LIGHT-FRACTION DIAGNOSTIC")
    print("=" * 78)
    print("Target: {}".format(TARGET_NAME))
    print("Slicer: {}".format(SLICER))
    print("Injected V_A, V_B: {:+.1f}, {:+.1f} km/s".format(TARGET_V_A, TARGET_V_B))
    print("Injected DeltaV: {:.1f} km/s".format(abs(TARGET_V_A - TARGET_V_B)))
    print("Injected sigma_A, sigma_B: {:.1f}, {:.1f} km/s".format(
        TARGET_SIGMA_A, TARGET_SIGMA_B
    ))

    print("")
    print("Only substantive fitting change:")
    print("")
    print(
        "    f_A fixed = {:.2f}".format(
            FIXED_FRACTION_A
        )
    )

    print(
        "    f_B fixed = {:.2f}".format(
            1.0 - FIXED_FRACTION_A
        )
    )

    print("")
    print("Still free:")
    print("")
    print("    V_A")
    print("    V_B")
    print("    sigma_A")
    print("    sigma_B")
    print("    SSP mixture within each component")

    print("")
    print(
        "Template case: matched_control"
    )

    print(
        "S/N/resel: {}".format(
            base.SNR_RESEL_VALUES.tolist()
        )
    )

    print(
        "Null realizations per S/N: {}".format(
            base.N_MC_NULL
        )
    )

    print(
        "CRD realizations per S/N: {}".format(
            base.N_MC_CRD
        )
    )

    print("")
    print(
        "IMPORTANT: the NULL experiment is rerun from scratch."
    )

    print(
        "The old free-fraction Delta-chi2 thresholds are NOT reused."
    )

    print("")


    # -------------------------------------------------------------------------
    # Run the complete original pipeline using the new fixed-fraction
    # two-component cell fit.
    # -------------------------------------------------------------------------

    base.main()


    # =============================================================================
    # EXTRA SANITY CHECK
    # =============================================================================
    #
    # Read the completed recovery table and verify that pPXF actually enforced
    # the expected approximately 50/50 template-weight division.
    # =============================================================================

    recovery_file = (

        base.OUTPUT_DIR

        /

        "crd_recovery_realizations.csv"

    )


    if recovery_file.is_file():

        df = pd.read_csv(
            recovery_file
        )


        print("")
        print("=" * 78)
        print("FIXED-FRACTION SANITY CHECK")
        print("=" * 78)

        print("")
        print(
            "Median recovered component fractions:"
        )


        fraction_check = (

            df

            .groupby(
                "snr_resel"
            )[
                [
                    "frac_A_rec",
                    "frac_B_rec",
                ]
            ]

            .median()

        )


        print("")
        print(
            fraction_check.to_string()
        )


        print("")
        print(
            "These should be approximately {:.2f} / {:.2f} at every S/N.".format(
                FIXED_FRACTION_A, 1.0 - FIXED_FRACTION_A
            )
        )


        # =====================================================================
        # Print the most important comparison table directly
        # =====================================================================

        summary_file = (

            base.OUTPUT_DIR

            /

            "recovery_summary.csv"

        )


        if summary_file.is_file():

            summary = pd.read_csv(
                summary_file
            )


            columns_to_show = [

                "snr_resel",

                "detection_fraction",

                "relative_kinematic_fraction",

                "absolute_kinematic_fraction",

                "median_abs_dv_A",

                "median_abs_dv_B",

                "median_abs_dsep",

                "median_abs_common_offset",

            ]


            print("")
            print("=" * 78)
            print("MOST IMPORTANT FIXED-FRACTION RESULTS")
            print("=" * 78)
            print("")


            print(

                summary[
                    columns_to_show
                ]

                .sort_values(
                    "snr_resel"
                )

                .to_string(
                    index=False
                )

            )


            print("")
            if TARGET_NAME.upper().replace(" ", "") == "IC25":
                print(
                    "Compare absolute_kinematic_fraction against the previous"
                )

                print(
                    "FREE-fraction matched-control values:"
                )

                print("")

                print(
                    "    S/N 30 : 0.29"
                )

                print(
                    "    S/N 35 : 0.34"
                )

                print(
                    "    S/N 40 : 0.46"
                )

                print(
                    "    S/N 45 : 0.54"
                )

                print(
                    "    S/N 50 : 0.51"
                )

                print("")

                print(
                    "A large increase would implicate the f_A / velocity-midpoint"
                )

                print(
                    "degeneracy as the main source of the common velocity offset."
                )
            else:
                print("")
                print(
                    "No previous free-fraction comparison table is embedded for "
                    "this target. Compare against a free-fraction run with the "
                    "same target and slicer."
                )

    else:

        print("")
        print(
            "WARNING: recovery output file was not found:"
        )

        print(
            recovery_file
        )
