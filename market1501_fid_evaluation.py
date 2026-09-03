import os
import glob
import re
import csv
import gc

import numpy as np

import torch

import matplotlib.pyplot as plt


# ============================================================
# 1. PYTORCH-FID IMPORT
# ============================================================

try:

    from pytorch_fid.inception import (
        InceptionV3
    )

    from pytorch_fid.fid_score import (
        get_activations,
        calculate_frechet_distance
    )

except ImportError:

    raise RuntimeError(
        "\n"
        "pytorch-fid is not installed.\n\n"
        "Install it with:\n"
        "python -m pip install pytorch-fid\n"
    )


# ============================================================
# 2. CONFIGURATION
# ============================================================

MARKET1501_ROOT = os.path.expanduser(
    "~/Desktop/ResearchResources/Phase1/deep-person-reid/"
    "reid-data/market1501/Market-1501-v15.09.15"
)


ORIGINAL_QUERY_DIR = os.path.join(
    MARKET1501_ROOT,
    "query"
)


ORIGINAL_GALLERY_DIR = os.path.join(
    MARKET1501_ROOT,
    "bounding_box_test"
)


PIS_ROOT = os.path.expanduser(
    "~/Desktop/ResearchResources/Phase2/"
    "disentanglement-pytorch/"
    "outputs/"
    "market1501_full_pis"
)


BASELINE_ROOT = os.path.expanduser(
    "~/Desktop/ResearchResources/Phase2/"
    "disentanglement-pytorch/"
    "outputs/"
    "market1501_privacy_baselines"
)


# ============================================================
# 3. METHODS
# ============================================================

METHODS = [
    {
        "name":
            "pis_0.00",

        "label":
            "PIS α=0.00 (VAE recon)",

        "group":
            "PIS",

        "alpha":
            0.00,

        "query_dir":
            os.path.join(
                PIS_ROOT,
                "alpha_0.00",
                "query"
            ),

        "gallery_dir":
            os.path.join(
                PIS_ROOT,
                "alpha_0.00",
                "bounding_box_test"
            )
    },

    {
        "name":
            "pis_0.25",

        "label":
            "PIS α=0.25",

        "group":
            "PIS",

        "alpha":
            0.25,

        "query_dir":
            os.path.join(
                PIS_ROOT,
                "alpha_0.25",
                "query"
            ),

        "gallery_dir":
            os.path.join(
                PIS_ROOT,
                "alpha_0.25",
                "bounding_box_test"
            )
    },

    {
        "name":
            "pis_0.50",

        "label":
            "PIS α=0.50",

        "group":
            "PIS",

        "alpha":
            0.50,

        "query_dir":
            os.path.join(
                PIS_ROOT,
                "alpha_0.50",
                "query"
            ),

        "gallery_dir":
            os.path.join(
                PIS_ROOT,
                "alpha_0.50",
                "bounding_box_test"
            )
    },

    {
        "name":
            "pis_0.75",

        "label":
            "PIS α=0.75",

        "group":
            "PIS",

        "alpha":
            0.75,

        "query_dir":
            os.path.join(
                PIS_ROOT,
                "alpha_0.75",
                "query"
            ),

        "gallery_dir":
            os.path.join(
                PIS_ROOT,
                "alpha_0.75",
                "bounding_box_test"
            )
    },

    {
        "name":
            "pis_1.00",

        "label":
            "PIS α=1.00",

        "group":
            "PIS",

        "alpha":
            1.00,

        "query_dir":
            os.path.join(
                PIS_ROOT,
                "alpha_1.00",
                "query"
            ),

        "gallery_dir":
            os.path.join(
                PIS_ROOT,
                "alpha_1.00",
                "bounding_box_test"
            )
    },

    {
        "name":
            "resized_control",

        "label":
            "Resize control",

        "group":
            "Control",

        "alpha":
            None,

        "query_dir":
            os.path.join(
                BASELINE_ROOT,
                "resized_control",
                "query"
            ),

        "gallery_dir":
            os.path.join(
                BASELINE_ROOT,
                "resized_control",
                "bounding_box_test"
            )
    },

    {
        "name":
            "blur_r8",

        "label":
            "Blur r=8",

        "group":
            "Blur",

        "alpha":
            None,

        "query_dir":
            os.path.join(
                BASELINE_ROOT,
                "blur_r8",
                "query"
            ),

        "gallery_dir":
            os.path.join(
                BASELINE_ROOT,
                "blur_r8",
                "bounding_box_test"
            )
    },

    {
        "name":
            "pixel_8x16",

        "label":
            "Pixel 8x16",

        "group":
            "Pixelation",

        "alpha":
            None,

        "query_dir":
            os.path.join(
                BASELINE_ROOT,
                "pixel_8x16",
                "query"
            ),

        "gallery_dir":
            os.path.join(
                BASELINE_ROOT,
                "pixel_8x16",
                "bounding_box_test"
            )
    }
]


# ============================================================
# 4. FID CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# 2048 = standard Inception pool3 feature dimensionality.
# ------------------------------------------------------------

FID_DIMS = 2048


# ------------------------------------------------------------
# RTX 2050 usually has limited VRAM.
#
# 16 is intentionally conservative.
#
# If CUDA runs out of memory, change this to 8.
# ------------------------------------------------------------

BATCH_SIZE = 16


NUM_WORKERS = 4


# ============================================================
# 5. OUTPUT
# ============================================================

OUTPUT_DIR = (
    "outputs/"
    "market1501_fid_evaluation"
)


STATS_DIR = os.path.join(
    OUTPUT_DIR,
    "cached_statistics"
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


os.makedirs(
    STATS_DIR,
    exist_ok=True
)


RESULTS_CSV = os.path.join(
    OUTPUT_DIR,
    "fid_results.csv"
)


FID_PLOT = os.path.join(
    OUTPUT_DIR,
    "fid_comparison.png"
)


PIS_FID_PLOT = os.path.join(
    OUTPUT_DIR,
    "pis_fid_vs_alpha.png"
)


# ============================================================
# 6. DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("=" * 92)

print(
    "Phase 4D - Market-1501 FID Evaluation"
)

print("=" * 92)


print(
    "Device:",
    device
)


if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


print()


print(
    f"FID feature dimensions: "
    f"{FID_DIMS}"
)


print(
    f"Batch size: "
    f"{BATCH_SIZE}"
)


print()


# ============================================================
# 7. MARKET-1501 IMAGE COLLECTION
# ============================================================

filename_pattern = re.compile(
    r"([-\d]+)_c(\d)"
)


def collect_images(
    query_dir,
    gallery_dir
):

    all_paths = []


    for image_dir in [
        query_dir,
        gallery_dir
    ]:

        image_paths = sorted(
            glob.glob(
                os.path.join(
                    image_dir,
                    "*.jpg"
                )
            )
        )


        for path in image_paths:

            filename = os.path.basename(
                path
            )


            match = filename_pattern.search(
                filename
            )


            if match is None:

                continue


            pid = int(
                match.group(1)
            )


            # ------------------------------------------------
            # Match all previous evaluation scripts:
            # ignore Market-1501 junk PID -1.
            # ------------------------------------------------

            if pid == -1:

                continue


            all_paths.append(
                path
            )


    return all_paths


# ============================================================
# 8. ORIGINAL REFERENCE IMAGE LIST
# ============================================================

original_files = collect_images(
    ORIGINAL_QUERY_DIR,
    ORIGINAL_GALLERY_DIR
)


print(
    f"Original reference images: "
    f"{len(original_files)}"
)


if len(
    original_files
) != 19281:

    print(
        "WARNING: expected 19,281 combined "
        "query/gallery images."
    )


print()


# ============================================================
# 9. VERIFY METHOD DATASETS
# ============================================================

for method in METHODS:

    if not os.path.isdir(
        method[
            "query_dir"
        ]
    ):

        raise FileNotFoundError(
            "Missing directory:\n"
            f"{method['query_dir']}"
        )


    if not os.path.isdir(
        method[
            "gallery_dir"
        ]
    ):

        raise FileNotFoundError(
            "Missing directory:\n"
            f"{method['gallery_dir']}"
        )


print(
    "All transformed datasets verified."
)


print()


# ============================================================
# 10. BUILD FID INCEPTION MODEL
# ============================================================

print("=" * 92)

print(
    "Loading FID Inception-v3 feature extractor..."
)

print("=" * 92)

print()


block_index = (
    InceptionV3
    .BLOCK_INDEX_BY_DIM[
        FID_DIMS
    ]
)


fid_model = InceptionV3(
    [
        block_index
    ]
).to(
    device
)


fid_model.eval()


print(
    "FID Inception model loaded."
)


print()


# ============================================================
# 11. COMPUTE / CACHE FEATURE STATISTICS
# ============================================================

def compute_statistics(
    files,
    cache_path,
    label
):

    # --------------------------------------------------------
    # Caching is useful because Inception feature extraction
    # is the expensive part of FID.
    #
    # If the script is interrupted, completed methods will
    # not need to be processed again.
    # --------------------------------------------------------

    if os.path.exists(
        cache_path
    ):

        print(
            f"Loading cached statistics: "
            f"{label}"
        )


        cached = np.load(
            cache_path
        )


        return (
            cached[
                "mu"
            ],

            cached[
                "sigma"
            ]
        )


    print(
        f"Extracting Inception features: "
        f"{label}"
    )


    print(
        f"  Images: "
        f"{len(files)}"
    )


    activations = get_activations(
        files,
        fid_model,
        batch_size=BATCH_SIZE,
        dims=FID_DIMS,
        device=device,
        num_workers=NUM_WORKERS
    )


    print(
        "  Computing mean and covariance..."
    )


    mu = np.mean(
        activations,
        axis=0
    )


    sigma = np.cov(
        activations,
        rowvar=False
    )


    np.savez_compressed(
        cache_path,
        mu=mu,
        sigma=sigma
    )


    del activations


    gc.collect()


    if torch.cuda.is_available():

        torch.cuda.empty_cache()


    print(
        f"  Saved statistics:"
    )


    print(
        f"  {cache_path}"
    )


    print()


    return (
        mu,
        sigma
    )


# ============================================================
# 12. ORIGINAL REFERENCE STATISTICS
# ============================================================

print("=" * 92)

print(
    "ORIGINAL MARKET-1501 REFERENCE"
)

print("=" * 92)

print()


original_stats_path = os.path.join(
    STATS_DIR,
    "original_market1501.npz"
)


(
    original_mu,
    original_sigma
) = compute_statistics(
    original_files,
    original_stats_path,
    "Original Market-1501"
)


# ============================================================
# 13. RUN FID FOR EACH METHOD
# ============================================================

results = []


for method in METHODS:

    print()

    print("=" * 92)

    print(
        f"FID: {method['label']}"
    )

    print("=" * 92)

    print()


    transformed_files = collect_images(
        method[
            "query_dir"
        ],
        method[
            "gallery_dir"
        ]
    )


    print(
        f"Transformed images: "
        f"{len(transformed_files)}"
    )


    if len(
        transformed_files
    ) != len(
        original_files
    ):

        raise RuntimeError(
            "Image count mismatch for "
            f"{method['label']}:\n"
            f"Original: {len(original_files)}\n"
            f"Transformed: {len(transformed_files)}"
        )


    method_stats_path = os.path.join(
        STATS_DIR,
        (
            f"{method['name']}.npz"
        )
    )


    (
        transformed_mu,
        transformed_sigma
    ) = compute_statistics(
        transformed_files,
        method_stats_path,
        method[
            "label"
        ]
    )


    print(
        "Computing Fréchet distance..."
    )


    fid_value = (
        calculate_frechet_distance(
            original_mu,
            original_sigma,
            transformed_mu,
            transformed_sigma
        )
    )


    result = {
        "method":
            method[
                "name"
            ],

        "label":
            method[
                "label"
            ],

        "group":
            method[
                "group"
            ],

        "alpha":
            (
                ""
                if method[
                    "alpha"
                ] is None
                else method[
                    "alpha"
                ]
            ),

        "num_original_images":
            len(
                original_files
            ),

        "num_transformed_images":
            len(
                transformed_files
            ),

        "fid":
            float(
                fid_value
            )
    }


    results.append(
        result
    )


    print()


    print(
        f"{method['label']}"
    )


    print(
        f"FID: "
        f"{fid_value:.4f}"
    )


# ============================================================
# 14. FINAL TABLE
# ============================================================

print()

print("=" * 92)

print(
    "FINAL MARKET-1501 FID RESULTS"
)

print("=" * 92)

print()


print(
    f"{'Method':<30}"
    f"{'FID ↓':>14}"
)


print(
    "-" * 46
)


for result in results:

    print(
        f"{result['label']:<30}"
        f"{result['fid']:>14.4f}"
    )


# ============================================================
# 15. SAVE RESULTS
# ============================================================

with open(
    RESULTS_CSV,
    "w",
    newline=""
) as csv_file:

    writer = csv.DictWriter(
        csv_file,
        fieldnames=[
            "method",
            "label",
            "group",
            "alpha",
            "num_original_images",
            "num_transformed_images",
            "fid"
        ]
    )


    writer.writeheader()


    writer.writerows(
        results
    )


# ============================================================
# 16. ALL-METHOD FID PLOT
# ============================================================

labels = [
    result[
        "label"
    ]
    for result in results
]


fid_values = [
    result[
        "fid"
    ]
    for result in results
]


x_positions = list(
    range(
        len(
            labels
        )
    )
)


plt.figure(
    figsize=(
        13,
        7
    )
)


plt.bar(
    x_positions,
    fid_values
)


plt.xticks(
    x_positions,
    labels,
    rotation=35,
    ha="right"
)


plt.ylabel(
    "FID ↓"
)


plt.xlabel(
    "Transformation Method"
)


plt.title(
    "Distribution-Level Distance to Original Market-1501"
)


plt.grid(
    True,
    alpha=0.3,
    axis="y"
)


plt.tight_layout()


plt.savefig(
    FID_PLOT,
    dpi=150
)


plt.close()


# ============================================================
# 17. PIS-ONLY FID CURVE
# ============================================================

pis_results = [
    result

    for result in results

    if result[
        "group"
    ] == "PIS"
]


pis_results = sorted(
    pis_results,
    key=lambda result:
        float(
            result[
                "alpha"
            ]
        )
)


pis_alphas = [
    float(
        result[
            "alpha"
        ]
    )
    for result in pis_results
]


pis_fids = [
    result[
        "fid"
    ]
    for result in pis_results
]


plt.figure(
    figsize=(
        10,
        6
    )
)


plt.plot(
    pis_alphas,
    pis_fids,
    marker="o"
)


plt.xlabel(
    "PIS Shift Strength α"
)


plt.ylabel(
    "FID ↓"
)


plt.title(
    "Distributional Fidelity vs PIS Shift Strength"
)


plt.grid(
    True,
    alpha=0.3
)


plt.tight_layout()


plt.savefig(
    PIS_FID_PLOT,
    dpi=150
)


plt.close()


# ============================================================
# 18. PIS-SPECIFIC CHANGE
# ============================================================

pis_zero = next(
    result

    for result in pis_results

    if abs(
        float(
            result[
                "alpha"
            ]
        )
        -
        0.0
    ) < 1e-8
)


pis_one = next(
    result

    for result in pis_results

    if abs(
        float(
            result[
                "alpha"
            ]
        )
        -
        1.0
    ) < 1e-8
)


print()

print("=" * 92)

print(
    "PIS DISTRIBUTIONAL CHANGE"
)

print("=" * 92)

print()


print(
    f"VAE reconstruction FID "
    f"(α=0.00): "
    f"{pis_zero['fid']:.4f}"
)


print(
    f"Full PIS FID "
    f"(α=1.00): "
    f"{pis_one['fid']:.4f}"
)


print(
    f"Additional FID change from "
    f"identity shifting: "
    f"{pis_one['fid'] - pis_zero['fid']:+.4f}"
)


# ============================================================
# 19. FINISH
# ============================================================

print()

print("=" * 92)

print(
    "Phase 4D FID evaluation completed."
)

print("=" * 92)

print()


print(
    "Saved:"
)


print(
    RESULTS_CSV
)


print(
    FID_PLOT
)


print(
    PIS_FID_PLOT
)


print()


print(
    "Cached Inception statistics:"
)


print(
    STATS_DIR
)