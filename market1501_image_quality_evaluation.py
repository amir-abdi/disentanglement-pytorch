import os
import glob
import csv

import numpy as np

import torch
import torch.nn.functional as F

from torch.utils.data import (
    Dataset,
    DataLoader
)

from torchvision import transforms

from PIL import Image

import matplotlib.pyplot as plt


# ============================================================
# 1. CONFIGURATION
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
# 2. IMAGE SIZE
# ============================================================

IMAGE_HEIGHT = 128
IMAGE_WIDTH = 64


# ============================================================
# 3. METHODS
# ============================================================

METHODS = [
    # --------------------------------------------------------
    # PIS
    # --------------------------------------------------------

    {
        "name": "pis_0.00",
        "label": "PIS α=0.00 (VAE recon)",
        "group": "PIS",
        "root": os.path.join(
            PIS_ROOT,
            "alpha_0.00"
        )
    },

    {
        "name": "pis_0.25",
        "label": "PIS α=0.25",
        "group": "PIS",
        "root": os.path.join(
            PIS_ROOT,
            "alpha_0.25"
        )
    },

    {
        "name": "pis_0.50",
        "label": "PIS α=0.50",
        "group": "PIS",
        "root": os.path.join(
            PIS_ROOT,
            "alpha_0.50"
        )
    },

    {
        "name": "pis_0.75",
        "label": "PIS α=0.75",
        "group": "PIS",
        "root": os.path.join(
            PIS_ROOT,
            "alpha_0.75"
        )
    },

    {
        "name": "pis_1.00",
        "label": "PIS α=1.00",
        "group": "PIS",
        "root": os.path.join(
            PIS_ROOT,
            "alpha_1.00"
        )
    },


    # --------------------------------------------------------
    # Conventional controls / baselines
    # --------------------------------------------------------

    {
        "name": "resized_control",
        "label": "Resize control",
        "group": "Control",
        "root": os.path.join(
            BASELINE_ROOT,
            "resized_control"
        )
    },

    {
        "name": "blur_r2",
        "label": "Blur r=2",
        "group": "Blur",
        "root": os.path.join(
            BASELINE_ROOT,
            "blur_r2"
        )
    },

    {
        "name": "blur_r4",
        "label": "Blur r=4",
        "group": "Blur",
        "root": os.path.join(
            BASELINE_ROOT,
            "blur_r4"
        )
    },

    {
        "name": "blur_r8",
        "label": "Blur r=8",
        "group": "Blur",
        "root": os.path.join(
            BASELINE_ROOT,
            "blur_r8"
        )
    },

    {
        "name": "pixel_32x64",
        "label": "Pixel 32x64",
        "group": "Pixelation",
        "root": os.path.join(
            BASELINE_ROOT,
            "pixel_32x64"
        )
    },

    {
        "name": "pixel_16x32",
        "label": "Pixel 16x32",
        "group": "Pixelation",
        "root": os.path.join(
            BASELINE_ROOT,
            "pixel_16x32"
        )
    },

    {
        "name": "pixel_8x16",
        "label": "Pixel 8x16",
        "group": "Pixelation",
        "root": os.path.join(
            BASELINE_ROOT,
            "pixel_8x16"
        )
    }
]


# ============================================================
# 4. RUNTIME
# ============================================================

BATCH_SIZE = 128

NUM_WORKERS = 4


# ============================================================
# 5. OUTPUT
# ============================================================

OUTPUT_DIR = (
    "outputs/"
    "market1501_image_quality"
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


SUMMARY_CSV = os.path.join(
    OUTPUT_DIR,
    "image_quality_summary.csv"
)


PER_IMAGE_CSV = os.path.join(
    OUTPUT_DIR,
    "image_quality_per_image.csv"
)


SSIM_PLOT = os.path.join(
    OUTPUT_DIR,
    "ssim_comparison.png"
)


PSNR_PLOT = os.path.join(
    OUTPUT_DIR,
    "psnr_comparison.png"
)


PIS_SSIM_PLOT = os.path.join(
    OUTPUT_DIR,
    "pis_ssim_vs_alpha.png"
)


PIS_PSNR_PLOT = os.path.join(
    OUTPUT_DIR,
    "pis_psnr_vs_alpha.png"
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
    "Phase 4C - Market-1501 Image Quality Evaluation"
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


# ============================================================
# 7. VERIFY PATHS
# ============================================================

for method in METHODS:

    query_path = os.path.join(
        method[
            "root"
        ],
        "query"
    )


    gallery_path = os.path.join(
        method[
            "root"
        ],
        "bounding_box_test"
    )


    if not os.path.isdir(
        query_path
    ):

        raise FileNotFoundError(
            f"Missing query directory:\n"
            f"{query_path}"
        )


    if not os.path.isdir(
        gallery_path
    ):

        raise FileNotFoundError(
            f"Missing gallery directory:\n"
            f"{gallery_path}"
        )


print(
    "All transformed dataset paths verified."
)


print()


# ============================================================
# 8. IMAGE TRANSFORM
# ============================================================

to_tensor = transforms.ToTensor()


def load_reference_image(
    path
):

    image = Image.open(
        path
    ).convert(
        "RGB"
    )


    # --------------------------------------------------------
    # Canonical comparison resolution.
    #
    # PIL size = (width, height)
    # --------------------------------------------------------

    image = image.resize(
        (
            IMAGE_WIDTH,
            IMAGE_HEIGHT
        ),
        resample=Image.Resampling.BILINEAR
    )


    return to_tensor(
        image
    )


def load_transformed_image(
    path
):

    image = Image.open(
        path
    ).convert(
        "RGB"
    )


    if image.size != (
        IMAGE_WIDTH,
        IMAGE_HEIGHT
    ):

        image = image.resize(
            (
                IMAGE_WIDTH,
                IMAGE_HEIGHT
            ),
            resample=Image.Resampling.BILINEAR
        )


    return to_tensor(
        image
    )


# ============================================================
# 9. PAIR DATASET
# ============================================================

class ImagePairDataset(
    Dataset
):

    def __init__(
        self,
        original_dir,
        transformed_dir,
        split_name
    ):

        self.original_dir = (
            original_dir
        )


        self.transformed_dir = (
            transformed_dir
        )


        self.split_name = (
            split_name
        )


        transformed_paths = sorted(
            glob.glob(
                os.path.join(
                    transformed_dir,
                    "*.jpg"
                )
            )
        )


        if len(
            transformed_paths
        ) == 0:

            raise RuntimeError(
                "No transformed images found:\n"
                f"{transformed_dir}"
            )


        self.samples = []


        for transformed_path in transformed_paths:

            filename = os.path.basename(
                transformed_path
            )


            original_path = os.path.join(
                original_dir,
                filename
            )


            if not os.path.exists(
                original_path
            ):

                raise FileNotFoundError(
                    "Matching original image missing:\n"
                    f"{original_path}"
                )


            self.samples.append(
                (
                    original_path,
                    transformed_path,
                    filename
                )
            )


    def __len__(
        self
    ):

        return len(
            self.samples
        )


    def __getitem__(
        self,
        index
    ):

        (
            original_path,
            transformed_path,
            filename
        ) = self.samples[
            index
        ]


        original = load_reference_image(
            original_path
        )


        transformed = load_transformed_image(
            transformed_path
        )


        return (
            original,
            transformed,
            filename
        )


# ============================================================
# 10. PSNR
# ============================================================

def compute_psnr(
    original,
    transformed
):

    # --------------------------------------------------------
    # Inputs are expected in [0, 1].
    #
    # PSNR =
    # 10 log10(MAX^2 / MSE)
    #
    # MAX = 1.
    # --------------------------------------------------------

    mse = (
        (
            original
            -
            transformed
        )
        ** 2
    ).mean(
        dim=(
            1,
            2,
            3
        )
    )


    mse = torch.clamp(
        mse,
        min=1e-12
    )


    psnr = (
        10.0
        *
        torch.log10(
            1.0
            /
            mse
        )
    )


    return psnr


# ============================================================
# 11. STANDARD SSIM
# ============================================================

def create_gaussian_window(
    window_size,
    sigma,
    channels,
    device
):

    coordinates = torch.arange(
        window_size,
        dtype=torch.float32,
        device=device
    )


    coordinates = (
        coordinates
        -
        window_size
        //
        2
    )


    gaussian_1d = torch.exp(
        -(
            coordinates
            **
            2
        )
        /
        (
            2
            *
            sigma
            **
            2
        )
    )


    gaussian_1d = (
        gaussian_1d
        /
        gaussian_1d.sum()
    )


    gaussian_2d = torch.outer(
        gaussian_1d,
        gaussian_1d
    )


    window = gaussian_2d.view(
        1,
        1,
        window_size,
        window_size
    )


    window = window.expand(
        channels,
        1,
        window_size,
        window_size
    ).contiguous()


    return window


def compute_ssim(
    original,
    transformed,
    window_size=11,
    sigma=1.5
):

    channels = original.shape[
        1
    ]


    window = create_gaussian_window(
        window_size,
        sigma,
        channels,
        original.device
    )


    padding = (
        window_size
        //
        2
    )


    # --------------------------------------------------------
    # Reflection padding avoids introducing artificial black
    # borders into local SSIM statistics.
    # --------------------------------------------------------

    original_padded = F.pad(
        original,
        (
            padding,
            padding,
            padding,
            padding
        ),
        mode="reflect"
    )


    transformed_padded = F.pad(
        transformed,
        (
            padding,
            padding,
            padding,
            padding
        ),
        mode="reflect"
    )


    mu_original = F.conv2d(
        original_padded,
        window,
        groups=channels
    )


    mu_transformed = F.conv2d(
        transformed_padded,
        window,
        groups=channels
    )


    mu_original_sq = (
        mu_original
        **
        2
    )


    mu_transformed_sq = (
        mu_transformed
        **
        2
    )


    mu_cross = (
        mu_original
        *
        mu_transformed
    )


    original_squared = (
        original
        **
        2
    )


    transformed_squared = (
        transformed
        **
        2
    )


    cross_product = (
        original
        *
        transformed
    )


    original_squared = F.pad(
        original_squared,
        (
            padding,
            padding,
            padding,
            padding
        ),
        mode="reflect"
    )


    transformed_squared = F.pad(
        transformed_squared,
        (
            padding,
            padding,
            padding,
            padding
        ),
        mode="reflect"
    )


    cross_product = F.pad(
        cross_product,
        (
            padding,
            padding,
            padding,
            padding
        ),
        mode="reflect"
    )


    sigma_original_sq = (
        F.conv2d(
            original_squared,
            window,
            groups=channels
        )
        -
        mu_original_sq
    )


    sigma_transformed_sq = (
        F.conv2d(
            transformed_squared,
            window,
            groups=channels
        )
        -
        mu_transformed_sq
    )


    sigma_cross = (
        F.conv2d(
            cross_product,
            window,
            groups=channels
        )
        -
        mu_cross
    )


    # --------------------------------------------------------
    # Standard SSIM constants for image range [0, 1].
    # --------------------------------------------------------

    c1 = (
        0.01
        **
        2
    )


    c2 = (
        0.03
        **
        2
    )


    numerator = (
        (
            2
            *
            mu_cross
            +
            c1
        )
        *
        (
            2
            *
            sigma_cross
            +
            c2
        )
    )


    denominator = (
        (
            mu_original_sq
            +
            mu_transformed_sq
            +
            c1
        )
        *
        (
            sigma_original_sq
            +
            sigma_transformed_sq
            +
            c2
        )
    )


    ssim_map = (
        numerator
        /
        (
            denominator
            +
            1e-12
        )
    )


    # --------------------------------------------------------
    # One SSIM value per image.
    # --------------------------------------------------------

    return ssim_map.mean(
        dim=(
            1,
            2,
            3
        )
    )


# ============================================================
# 12. EVALUATE ONE SPLIT
# ============================================================

def evaluate_split(
    original_dir,
    transformed_dir,
    split_name,
    method_name,
    method_label,
    group,
    per_image_writer
):

    dataset = ImagePairDataset(
        original_dir,
        transformed_dir,
        split_name
    )


    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )


    all_psnr = []

    all_ssim = []


    processed = 0


    with torch.inference_mode():

        for (
            original,
            transformed,
            filenames
        ) in loader:

            original = original.to(
                device,
                non_blocking=True
            )


            transformed = transformed.to(
                device,
                non_blocking=True
            )


            psnr_values = compute_psnr(
                original,
                transformed
            )


            ssim_values = compute_ssim(
                original,
                transformed
            )


            psnr_cpu = (
                psnr_values
                .cpu()
                .numpy()
            )


            ssim_cpu = (
                ssim_values
                .cpu()
                .numpy()
            )


            all_psnr.extend(
                psnr_cpu.tolist()
            )


            all_ssim.extend(
                ssim_cpu.tolist()
            )


            # ------------------------------------------------
            # Save image-level results for later statistics.
            # ------------------------------------------------

            for (
                filename,
                psnr_value,
                ssim_value
            ) in zip(
                filenames,
                psnr_cpu,
                ssim_cpu
            ):

                per_image_writer.writerow(
                    {
                        "method":
                            method_name,

                        "label":
                            method_label,

                        "group":
                            group,

                        "split":
                            split_name,

                        "filename":
                            filename,

                        "psnr":
                            float(
                                psnr_value
                            ),

                        "ssim":
                            float(
                                ssim_value
                            )
                    }
                )


            processed += len(
                filenames
            )


            if (
                processed % 1000
                <
                len(
                    filenames
                )
            ):

                print(
                    f"  {split_name}: "
                    f"{processed}/"
                    f"{len(dataset)}"
                )


    all_psnr = np.asarray(
        all_psnr,
        dtype=np.float64
    )


    all_ssim = np.asarray(
        all_ssim,
        dtype=np.float64
    )


    return {
        "count":
            len(
                all_psnr
            ),

        "mean_psnr":
            float(
                np.mean(
                    all_psnr
                )
            ),

        "std_psnr":
            float(
                np.std(
                    all_psnr
                )
            ),

        "mean_ssim":
            float(
                np.mean(
                    all_ssim
                )
            ),

        "std_ssim":
            float(
                np.std(
                    all_ssim
                )
            ),

        "psnr_values":
            all_psnr,

        "ssim_values":
            all_ssim
    }


# ============================================================
# 13. RUN ALL METHODS
# ============================================================

summary_rows = []


with open(
    PER_IMAGE_CSV,
    "w",
    newline=""
) as per_image_file:

    per_image_writer = csv.DictWriter(
        per_image_file,
        fieldnames=[
            "method",
            "label",
            "group",
            "split",
            "filename",
            "psnr",
            "ssim"
        ]
    )


    per_image_writer.writeheader()


    for method in METHODS:

        print()

        print("=" * 92)

        print(
            f"EVALUATING: "
            f"{method['label']}"
        )

        print("=" * 92)

        print()


        query_transformed_dir = os.path.join(
            method[
                "root"
            ],
            "query"
        )


        gallery_transformed_dir = os.path.join(
            method[
                "root"
            ],
            "bounding_box_test"
        )


        # ====================================================
        # QUERY
        # ====================================================

        query_result = evaluate_split(
            ORIGINAL_QUERY_DIR,
            query_transformed_dir,
            "query",
            method[
                "name"
            ],
            method[
                "label"
            ],
            method[
                "group"
            ],
            per_image_writer
        )


        # ====================================================
        # GALLERY
        # ====================================================

        gallery_result = evaluate_split(
            ORIGINAL_GALLERY_DIR,
            gallery_transformed_dir,
            "gallery",
            method[
                "name"
            ],
            method[
                "label"
            ],
            method[
                "group"
            ],
            per_image_writer
        )


        # ====================================================
        # COMBINED
        # ====================================================

        combined_psnr = np.concatenate(
            [
                query_result[
                    "psnr_values"
                ],

                gallery_result[
                    "psnr_values"
                ]
            ]
        )


        combined_ssim = np.concatenate(
            [
                query_result[
                    "ssim_values"
                ],

                gallery_result[
                    "ssim_values"
                ]
            ]
        )


        combined_result = {
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

            "split":
                "combined",

            "count":
                len(
                    combined_psnr
                ),

            "mean_psnr":
                float(
                    np.mean(
                        combined_psnr
                    )
                ),

            "std_psnr":
                float(
                    np.std(
                        combined_psnr
                    )
                ),

            "mean_ssim":
                float(
                    np.mean(
                        combined_ssim
                    )
                ),

            "std_ssim":
                float(
                    np.std(
                        combined_ssim
                    )
                )
        }


        summary_rows.append(
            {
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

                "split":
                    "query",

                "count":
                    query_result[
                        "count"
                    ],

                "mean_psnr":
                    query_result[
                        "mean_psnr"
                    ],

                "std_psnr":
                    query_result[
                        "std_psnr"
                    ],

                "mean_ssim":
                    query_result[
                        "mean_ssim"
                    ],

                "std_ssim":
                    query_result[
                        "std_ssim"
                    ]
            }
        )


        summary_rows.append(
            {
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

                "split":
                    "gallery",

                "count":
                    gallery_result[
                        "count"
                    ],

                "mean_psnr":
                    gallery_result[
                        "mean_psnr"
                    ],

                "std_psnr":
                    gallery_result[
                        "std_psnr"
                    ],

                "mean_ssim":
                    gallery_result[
                        "mean_ssim"
                    ],

                "std_ssim":
                    gallery_result[
                        "std_ssim"
                    ]
            }
        )


        summary_rows.append(
            combined_result
        )


        print()

        print(
            f"{method['label']} COMBINED RESULTS"
        )


        print(
            "-" * 60
        )


        print(
            f"Images: "
            f"{combined_result['count']}"
        )


        print(
            f"PSNR: "
            f"{combined_result['mean_psnr']:.3f} "
            f"± "
            f"{combined_result['std_psnr']:.3f} dB"
        )


        print(
            f"SSIM: "
            f"{combined_result['mean_ssim']:.4f} "
            f"± "
            f"{combined_result['std_ssim']:.4f}"
        )


# ============================================================
# 14. SAVE SUMMARY
# ============================================================

with open(
    SUMMARY_CSV,
    "w",
    newline=""
) as summary_file:

    writer = csv.DictWriter(
        summary_file,
        fieldnames=[
            "method",
            "label",
            "group",
            "split",
            "count",
            "mean_psnr",
            "std_psnr",
            "mean_ssim",
            "std_ssim"
        ]
    )


    writer.writeheader()


    writer.writerows(
        summary_rows
    )


# ============================================================
# 15. EXTRACT COMBINED RESULTS
# ============================================================

combined_rows = [
    row
    for row in summary_rows
    if row[
        "split"
    ] == "combined"
]


# ============================================================
# 16. FINAL TABLE
# ============================================================

print()

print("=" * 92)

print(
    "FINAL IMAGE QUALITY RESULTS"
)

print("=" * 92)

print()


print(
    f"{'Method':<27}"
    f"{'PSNR (dB)':>16}"
    f"{'SSIM':>14}"
)


print(
    "-" * 60
)


for row in combined_rows:

    print(
        f"{row['label']:<27}"
        f"{row['mean_psnr']:>10.3f}"
        f" ± {row['std_psnr']:<5.3f}"
        f"{row['mean_ssim']:>9.4f}"
    )


# ============================================================
# 17. ALL-METHOD SSIM PLOT
# ============================================================

labels = [
    row[
        "label"
    ]
    for row in combined_rows
]


ssim_values = [
    row[
        "mean_ssim"
    ]
    for row in combined_rows
]


x_positions = list(
    range(
        len(
            labels
        )
    )
)


plt.figure(
    figsize=(14, 7)
)


plt.bar(
    x_positions,
    ssim_values
)


plt.xticks(
    x_positions,
    labels,
    rotation=35,
    ha="right"
)


plt.ylabel(
    "Mean SSIM"
)


plt.title(
    "Original-to-Transformed Structural Similarity"
)


plt.grid(
    True,
    alpha=0.3,
    axis="y"
)


plt.tight_layout()


plt.savefig(
    SSIM_PLOT,
    dpi=150
)


plt.close()


# ============================================================
# 18. ALL-METHOD PSNR PLOT
# ============================================================

psnr_values = [
    row[
        "mean_psnr"
    ]
    for row in combined_rows
]


plt.figure(
    figsize=(14, 7)
)


plt.bar(
    x_positions,
    psnr_values
)


plt.xticks(
    x_positions,
    labels,
    rotation=35,
    ha="right"
)


plt.ylabel(
    "Mean PSNR (dB)"
)


plt.title(
    "Original-to-Transformed PSNR"
)


plt.grid(
    True,
    alpha=0.3,
    axis="y"
)


plt.tight_layout()


plt.savefig(
    PSNR_PLOT,
    dpi=150
)


plt.close()


# ============================================================
# 19. PIS-ONLY CURVES
# ============================================================

pis_rows = [
    row
    for row in combined_rows
    if row[
        "group"
    ] == "PIS"
]


pis_alphas = [
    0.00,
    0.25,
    0.50,
    0.75,
    1.00
]


pis_ssim = [
    row[
        "mean_ssim"
    ]
    for row in pis_rows
]


pis_psnr = [
    row[
        "mean_psnr"
    ]
    for row in pis_rows
]


plt.figure(
    figsize=(10, 6)
)


plt.plot(
    pis_alphas,
    pis_ssim,
    marker="o"
)


plt.xlabel(
    "PIS Shift Strength α"
)


plt.ylabel(
    "Mean SSIM"
)


plt.title(
    "Structural Fidelity vs PIS Shift Strength"
)


plt.grid(
    True,
    alpha=0.3
)


plt.tight_layout()


plt.savefig(
    PIS_SSIM_PLOT,
    dpi=150
)


plt.close()


plt.figure(
    figsize=(10, 6)
)


plt.plot(
    pis_alphas,
    pis_psnr,
    marker="o"
)


plt.xlabel(
    "PIS Shift Strength α"
)


plt.ylabel(
    "Mean PSNR (dB)"
)


plt.title(
    "Pixel Fidelity vs PIS Shift Strength"
)


plt.grid(
    True,
    alpha=0.3
)


plt.tight_layout()


plt.savefig(
    PIS_PSNR_PLOT,
    dpi=150
)


plt.close()


# ============================================================
# 20. FINISH
# ============================================================

print()

print("=" * 92)

print(
    "Phase 4C image quality evaluation completed."
)

print("=" * 92)

print()


print(
    "Saved:"
)


print(
    SUMMARY_CSV
)


print(
    PER_IMAGE_CSV
)


print(
    SSIM_PLOT
)


print(
    PSNR_PLOT
)


print(
    PIS_SSIM_PLOT
)


print(
    PIS_PSNR_PLOT
)