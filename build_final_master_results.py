from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

OUTPUTS = ROOT / "outputs"

RESULTS_DIR = (
    ROOT
    / "results"
    / "tables"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. INPUT FILES
# ============================================================

PIS_UTILITY_FILE = (
    OUTPUTS
    / "market1501_pis_osnet_reid"
    / "pis_osnet_reid_results.csv"
)

PIS_PRIVACY_FILE = (
    OUTPUTS
    / "market1501_pis_privacy"
    / "pis_privacy_results.csv"
)

PIS_SIMILARITY_FILE = (
    OUTPUTS
    / "market1501_pis_privacy"
    / "pis_feature_similarity.csv"
)

DONOR_FILE = (
    OUTPUTS
    / "market1501_pis_donor_leakage"
    / "donor_leakage_results.csv"
)

ADAPTIVE_FILE = (
    OUTPUTS
    / "market1501_pis_adaptive_privacy"
    / "adaptive_privacy_attack_results.csv"
)

CONVENTIONAL_FILE = (
    OUTPUTS
    / "market1501_privacy_baseline_evaluation"
    / "privacy_baseline_results.csv"
)

QUALITY_FILE = (
    OUTPUTS
    / "market1501_image_quality"
    / "image_quality_summary.csv"
)

FID_FILE = (
    OUTPUTS
    / "market1501_fid_evaluation"
    / "fid_results.csv"
)


# ============================================================
# 3. VERIFY INPUT FILES
# ============================================================

required_files = [
    PIS_UTILITY_FILE,
    PIS_PRIVACY_FILE,
    PIS_SIMILARITY_FILE,
    DONOR_FILE,
    ADAPTIVE_FILE,
    CONVENTIONAL_FILE,
    QUALITY_FILE,
    FID_FILE,
]


for path in required_files:

    if not path.exists():

        raise FileNotFoundError(
            f"Missing required result file:\n"
            f"{path}"
        )


print("=" * 100)
print("Phase 5A - Final Results Consolidation")
print("=" * 100)
print()

print("All result files verified.")
print()


# ============================================================
# 4. LOAD RESULTS
# ============================================================

pis_utility = pd.read_csv(
    PIS_UTILITY_FILE
)

pis_privacy = pd.read_csv(
    PIS_PRIVACY_FILE
)

pis_similarity = pd.read_csv(
    PIS_SIMILARITY_FILE
)

donor = pd.read_csv(
    DONOR_FILE
)

adaptive = pd.read_csv(
    ADAPTIVE_FILE
)

conventional = pd.read_csv(
    CONVENTIONAL_FILE
)

quality = pd.read_csv(
    QUALITY_FILE
)

fid = pd.read_csv(
    FID_FILE
)


# ============================================================
# 5. ONLY USE COMBINED QUERY + GALLERY IMAGE QUALITY
# ============================================================

quality_combined = quality[
    quality["split"] == "combined"
].copy()


# ============================================================
# 6. HELPERS
# ============================================================

def get_alpha_row(
    dataframe,
    alpha
):

    mask = np.isclose(
        dataframe["alpha"].astype(float),
        alpha
    )

    rows = dataframe[
        mask
    ]

    if len(rows) != 1:

        raise RuntimeError(
            f"Expected exactly one row "
            f"for alpha={alpha}, "
            f"found {len(rows)}."
        )

    return rows.iloc[0]


def get_quality(
    method
):

    rows = quality_combined[
        quality_combined[
            "method"
        ] == method
    ]

    if rows.empty:

        return {
            "ssim": np.nan,
            "ssim_std": np.nan,
            "psnr": np.nan,
            "psnr_std": np.nan,
        }

    row = rows.iloc[0]

    return {
        "ssim":
            row[
                "mean_ssim"
            ],

        "ssim_std":
            row[
                "std_ssim"
            ],

        "psnr":
            row[
                "mean_psnr"
            ],

        "psnr_std":
            row[
                "std_psnr"
            ],
    }


def get_fid(
    method
):

    rows = fid[
        fid[
            "method"
        ] == method
    ]

    if rows.empty:

        return np.nan

    return float(
        rows.iloc[0][
            "fid"
        ]
    )


# ============================================================
# 7. MASTER ROWS
# ============================================================

rows = []


# ============================================================
# 8. ORIGINAL MARKET-1501 REFERENCE
# ============================================================

original = pis_utility[
    pis_utility[
        "dataset"
    ] == "Original"
].iloc[0]


rows.append(
    {
        "method":
            "Original",

        "category":
            "Reference",

        "alpha":
            np.nan,

        "utility_rank1":
            original["Rank-1"],

        "utility_mAP":
            original["mAP"],

        "source_attack_rank1":
            original["Rank-1"],

        "source_attack_mAP":
            original["mAP"],

        "reverse_attack_rank1":
            original["Rank-1"],

        "adaptive_attack_rank1":
            np.nan,

        "adaptive_attack_mAP":
            np.nan,

        "adaptive_original_baseline_rank1":
            np.nan,

        "donor_rank1":
            np.nan,

        "donor_mAP":
            np.nan,

        "feature_cosine":
            1.0,

        "feature_euclidean":
            0.0,

        "ssim":
            1.0,

        "ssim_std":
            0.0,

        "psnr_db":
            np.nan,

        "psnr_std":
            np.nan,

        "fid":
            0.0,

        "protocol_note":
            (
                "Original-to-original "
                "Market-1501 reference."
            ),
    }
)


# ============================================================
# 9. CONVENTIONAL BASELINES
# ============================================================

for _, conventional_row in (
    conventional.iterrows()
):

    method_name = (
        conventional_row[
            "name"
        ]
    )

    quality_values = get_quality(
        method_name
    )


    rows.append(
        {
            "method":
                conventional_row[
                    "label"
                ],

            "category":
                conventional_row[
                    "group"
                ],

            "alpha":
                np.nan,

            "utility_rank1":
                conventional_row[
                    "utility_rank1"
                ],

            "utility_mAP":
                conventional_row[
                    "utility_mAP"
                ],

            "source_attack_rank1":
                conventional_row[
                    "source_attack_rank1"
                ],

            "source_attack_mAP":
                conventional_row[
                    "source_attack_mAP"
                ],

            "reverse_attack_rank1":
                conventional_row[
                    "reverse_attack_rank1"
                ],

            "adaptive_attack_rank1":
                np.nan,

            "adaptive_attack_mAP":
                np.nan,

            "adaptive_original_baseline_rank1":
                np.nan,

            "donor_rank1":
                np.nan,

            "donor_mAP":
                np.nan,

            "feature_cosine":
                conventional_row[
                    "feature_cosine"
                ],

            "feature_euclidean":
                conventional_row[
                    "feature_euclidean"
                ],

            "ssim":
                quality_values[
                    "ssim"
                ],

            "ssim_std":
                quality_values[
                    "ssim_std"
                ],

            "psnr_db":
                quality_values[
                    "psnr"
                ],

            "psnr_std":
                quality_values[
                    "psnr_std"
                ],

            "fid":
                get_fid(
                    method_name
                ),

            "protocol_note":
                (
                    "Utility and fixed source attack "
                    "use full Market-1501 test set. "
                    "Adaptive attack not evaluated."
                ),
        }
    )


# ============================================================
# 10. PIS RESULTS
# ============================================================

PIS_ALPHAS = [
    0.00,
    0.25,
    0.50,
    0.75,
    1.00,
]


for alpha in PIS_ALPHAS:

    utility_row = get_alpha_row(
        pis_utility.dropna(
            subset=[
                "alpha"
            ]
        ),
        alpha
    )

    privacy_row = get_alpha_row(
        pis_privacy,
        alpha
    )

    similarity_row = get_alpha_row(
        pis_similarity,
        alpha
    )

    donor_row = get_alpha_row(
        donor,
        alpha
    )

    adaptive_row = get_alpha_row(
        adaptive,
        alpha
    )


    method_name = (
        f"pis_{alpha:.2f}"
    )


    quality_values = get_quality(
        method_name
    )


    rows.append(
        {
            "method":
                f"PIS α={alpha:.2f}",

            "category":
                "PIS",

            "alpha":
                alpha,

            "utility_rank1":
                utility_row[
                    "Rank-1"
                ],

            "utility_mAP":
                utility_row[
                    "mAP"
                ],

            "source_attack_rank1":
                privacy_row[
                    "oq_sg_rank1"
                ],

            "source_attack_mAP":
                privacy_row[
                    "oq_sg_mAP"
                ],

            "reverse_attack_rank1":
                privacy_row[
                    "sq_og_rank1"
                ],

            "adaptive_attack_rank1":
                adaptive_row[
                    "adapted_oq_sg_rank1"
                ],

            "adaptive_attack_mAP":
                adaptive_row[
                    "adapted_oq_sg_mAP"
                ],

            "adaptive_original_baseline_rank1":
                adaptive_row[
                    "original_target_rank1"
                ],

            "donor_rank1":
                donor_row[
                    "Rank-1"
                ],

            "donor_mAP":
                donor_row[
                    "mAP"
                ],

            "feature_cosine":
                similarity_row[
                    "combined_mean_cosine"
                ],

            "feature_euclidean":
                similarity_row[
                    "combined_mean_euclidean"
                ],

            "ssim":
                quality_values[
                    "ssim"
                ],

            "ssim_std":
                quality_values[
                    "ssim_std"
                ],

            "psnr_db":
                quality_values[
                    "psnr"
                ],

            "psnr_std":
                quality_values[
                    "psnr_std"
                ],

            "fid":
                get_fid(
                    method_name
                ),

            "protocol_note":
                (
                    "Utility, fixed source attack, "
                    "and donor attack use full "
                    "Market-1501 evaluation. "
                    "Adaptive attack uses separate "
                    "held-out 375-ID target split."
                ),
        }
    )


# ============================================================
# 11. CREATE MASTER TABLE
# ============================================================

master = pd.DataFrame(
    rows
)


# ============================================================
# 12. SAVE FULL PRECISION
# ============================================================

MASTER_CSV = (
    RESULTS_DIR
    / "final_master_results.csv"
)


master.to_csv(
    MASTER_CSV,
    index=False
)


# ============================================================
# 13. SAVE ROUNDED TABLE
# ============================================================

rounded = master.copy()


columns_to_round_4 = [
    "feature_cosine",
    "feature_euclidean",
    "ssim",
    "ssim_std",
]


columns_to_round_2 = [
    "utility_rank1",
    "utility_mAP",
    "source_attack_rank1",
    "source_attack_mAP",
    "reverse_attack_rank1",
    "adaptive_attack_rank1",
    "adaptive_attack_mAP",
    "adaptive_original_baseline_rank1",
    "donor_rank1",
    "donor_mAP",
    "psnr_db",
    "psnr_std",
    "fid",
]


for column in columns_to_round_4:

    rounded[
        column
    ] = rounded[
        column
    ].round(
        4
    )


for column in columns_to_round_2:

    rounded[
        column
    ] = rounded[
        column
    ].round(
        2
    )


ROUNDED_CSV = (
    RESULTS_DIR
    / "final_master_results_rounded.csv"
)


rounded.to_csv(
    ROUNDED_CSV,
    index=False
)


# ============================================================
# 14. HEADLINE COMPARISON
# ============================================================

headline_methods = [
    "Original",
    "Resize control",
    "Blur r=8",
    "Pixel 8x16",
    "PIS α=0.50",
    "PIS α=0.75",
    "PIS α=1.00",
]


headline = rounded[
    rounded[
        "method"
    ].isin(
        headline_methods
    )
].copy()


headline["method"] = pd.Categorical(
    headline["method"],
    categories=headline_methods,
    ordered=True
)


headline = headline.sort_values(
    "method"
)


HEADLINE_CSV = (
    RESULTS_DIR
    / "headline_results.csv"
)


headline.to_csv(
    HEADLINE_CSV,
    index=False
)


# ============================================================
# 15. MARKDOWN VERSION
# ============================================================

MARKDOWN_FILE = (
    RESULTS_DIR
    / "final_master_results.md"
)


display_columns = [
    "method",
    "utility_rank1",
    "utility_mAP",
    "source_attack_rank1",
    "source_attack_mAP",
    "adaptive_attack_rank1",
    "donor_rank1",
    "feature_cosine",
    "ssim",
    "psnr_db",
    "fid",
]


with open(
    MARKDOWN_FILE,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "# Final Experimental Results\n\n"
    )

    file.write(
        "Higher utility Rank-1/mAP is better. "
        "Lower source attack, adaptive attack, "
        "donor leakage and FID are preferable "
        "for their respective interpretations.\n\n"
    )

    file.write(
        rounded[
            display_columns
        ].to_markdown(
            index=False
        )
    )

    file.write(
        "\n\n"
    )

    file.write(
        "## Protocol note\n\n"
    )

    file.write(
        "Adaptive privacy results use a "
        "separate held-out 375-identity target "
        "split and should not be treated as "
        "directly numerically equivalent to "
        "the full 750-identity fixed-attack "
        "protocol.\n"
    )


# ============================================================
# 16. PRINT HEADLINE TABLE
# ============================================================

print("=" * 120)

print(
    "HEADLINE FINAL RESULTS"
)

print("=" * 120)

print()


print(
    headline[
        [
            "method",
            "utility_rank1",
            "utility_mAP",
            "source_attack_rank1",
            "adaptive_attack_rank1",
            "donor_rank1",
            "ssim",
            "psnr_db",
            "fid",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# 17. FINISH
# ============================================================

print()

print("=" * 100)

print(
    "Phase 5A final results consolidation completed."
)

print("=" * 100)

print()

print(
    "Saved:"
)

print(
    MASTER_CSV
)

print(
    ROUNDED_CSV
)

print(
    HEADLINE_CSV
)

print(
    MARKDOWN_FILE
)