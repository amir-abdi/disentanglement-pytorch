import os
import glob
import re
import csv

from PIL import (
    Image,
    ImageFilter
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

MARKET1501_ROOT = os.path.expanduser(
    "~/Desktop/ResearchResources/Phase1/deep-person-reid/"
    "reid-data/market1501/Market-1501-v15.09.15"
)


QUERY_DIR = os.path.join(
    MARKET1501_ROOT,
    "query"
)


GALLERY_DIR = os.path.join(
    MARKET1501_ROOT,
    "bounding_box_test"
)


# ------------------------------------------------------------
# Match the spatial resolution used by the VAE/PIS pipeline.
#
# PIL uses:
#     (width, height)
# ------------------------------------------------------------

CANONICAL_SIZE = (
    64,
    128
)


# ============================================================
# 2. BASELINE STRENGTHS
# ============================================================

BLUR_RADII = [
    2.0,
    4.0,
    8.0
]


# ------------------------------------------------------------
# Pixelation:
#
# First shrink the 64x128 canonical image to the following
# size, then enlarge back to 64x128 using nearest-neighbour.
#
# Mild:
#     32x64
#
# Medium:
#     16x32
#
# Strong:
#     8x16
# ------------------------------------------------------------

PIXELATION_SIZES = [
    (
        32,
        64
    ),

    (
        16,
        32
    ),

    (
        8,
        16
    )
]


JPEG_QUALITY = 95


# ============================================================
# 3. OUTPUT
# ============================================================

OUTPUT_ROOT = (
    "outputs/"
    "market1501_privacy_baselines"
)


CONFIG_CSV = os.path.join(
    OUTPUT_ROOT,
    "baseline_configurations.csv"
)


os.makedirs(
    OUTPUT_ROOT,
    exist_ok=True
)


# ============================================================
# 4. MARKET-1501 PARSING
# ============================================================

pattern = re.compile(
    r"([-\d]+)_c(\d)"
)


def collect_market1501_images(
    image_dir
):

    paths = sorted(
        glob.glob(
            os.path.join(
                image_dir,
                "*.jpg"
            )
        )
    )


    samples = []


    for path in paths:

        filename = os.path.basename(
            path
        )


        match = pattern.search(
            filename
        )


        if match is None:

            continue


        pid = int(
            match.group(1)
        )


        camid = int(
            match.group(2)
        )


        # ----------------------------------------------------
        # Keep the evaluation convention consistent with the
        # PIS datasets: ignore junk PID -1.
        # ----------------------------------------------------

        if pid == -1:

            continue


        samples.append(
            {
                "path":
                    path,

                "filename":
                    filename,

                "pid":
                    pid,

                "camid":
                    camid
            }
        )


    return samples


# ============================================================
# 5. LOAD SPLIT METADATA
# ============================================================

query_samples = collect_market1501_images(
    QUERY_DIR
)


gallery_samples = collect_market1501_images(
    GALLERY_DIR
)


print("=" * 88)

print(
    "Phase 4A - Market-1501 Conventional Privacy Baselines"
)

print("=" * 88)

print()


print(
    f"Query images: "
    f"{len(query_samples)}"
)


print(
    f"Gallery images: "
    f"{len(gallery_samples)}"
)


print()


# ============================================================
# 6. IMAGE HELPERS
# ============================================================

def load_canonical_image(
    path
):

    image = Image.open(
        path
    ).convert(
        "RGB"
    )


    # --------------------------------------------------------
    # Match the resolution entering our VAE/PIS system.
    # --------------------------------------------------------

    image = image.resize(
        CANONICAL_SIZE,
        resample=Image.Resampling.BILINEAR
    )


    return image


def save_jpeg(
    image,
    path
):

    image.save(
        path,
        format="JPEG",
        quality=JPEG_QUALITY
    )


# ============================================================
# 7. BASELINE TRANSFORMATIONS
# ============================================================

def resized_control(
    image
):

    return image.copy()


def gaussian_blur(
    image,
    radius
):

    return image.filter(
        ImageFilter.GaussianBlur(
            radius=radius
        )
    )


def pixelate(
    image,
    low_resolution
):

    # --------------------------------------------------------
    # Downsample with BOX averaging.
    # --------------------------------------------------------

    small = image.resize(
        low_resolution,
        resample=Image.Resampling.BOX
    )


    # --------------------------------------------------------
    # Upsample with nearest neighbour to produce visible
    # pixel blocks.
    # --------------------------------------------------------

    pixelated = small.resize(
        CANONICAL_SIZE,
        resample=Image.Resampling.NEAREST
    )


    return pixelated


# ============================================================
# 8. DEFINE ALL BASELINES
# ============================================================

BASELINES = []


# ------------------------------------------------------------
# Control
# ------------------------------------------------------------

BASELINES.append(
    {
        "name":
            "resized_control",

        "method":
            "resize_only",

        "strength":
            "none"
    }
)


# ------------------------------------------------------------
# Gaussian blur
# ------------------------------------------------------------

for radius in BLUR_RADII:

    BASELINES.append(
        {
            "name":
                f"blur_r{radius:g}",

            "method":
                "gaussian_blur",

            "strength":
                radius
        }
    )


# ------------------------------------------------------------
# Pixelation
# ------------------------------------------------------------

for (
    width,
    height
) in PIXELATION_SIZES:

    BASELINES.append(
        {
            "name":
                f"pixel_{width}x{height}",

            "method":
                "pixelation",

            "strength":
                f"{width}x{height}",

            "width":
                width,

            "height":
                height
        }
    )


print(
    "Baseline configurations:"
)


for baseline in BASELINES:

    print(
        f"  {baseline['name']}"
    )


print()


# ============================================================
# 9. SAVE BASELINE CONFIGURATION
# ============================================================

with open(
    CONFIG_CSV,
    "w",
    newline=""
) as csv_file:

    writer = csv.writer(
        csv_file
    )


    writer.writerow(
        [
            "name",
            "method",
            "strength",
            "output_width",
            "output_height"
        ]
    )


    for baseline in BASELINES:

        writer.writerow(
            [
                baseline[
                    "name"
                ],

                baseline[
                    "method"
                ],

                baseline[
                    "strength"
                ],

                CANONICAL_SIZE[
                    0
                ],

                CANONICAL_SIZE[
                    1
                ]
            ]
        )


# ============================================================
# 10. CREATE OUTPUT DIRECTORIES
# ============================================================

for baseline in BASELINES:

    baseline_root = os.path.join(
        OUTPUT_ROOT,
        baseline[
            "name"
        ]
    )


    os.makedirs(
        os.path.join(
            baseline_root,
            "query"
        ),
        exist_ok=True
    )


    os.makedirs(
        os.path.join(
            baseline_root,
            "bounding_box_test"
        ),
        exist_ok=True
    )


# ============================================================
# 11. APPLY ONE BASELINE
# ============================================================

def apply_baseline(
    image,
    baseline
):

    method = baseline[
        "method"
    ]


    if method == "resize_only":

        return resized_control(
            image
        )


    elif method == "gaussian_blur":

        return gaussian_blur(
            image,
            radius=float(
                baseline[
                    "strength"
                ]
            )
        )


    elif method == "pixelation":

        return pixelate(
            image,
            (
                baseline[
                    "width"
                ],
                baseline[
                    "height"
                ]
            )
        )


    else:

        raise ValueError(
            f"Unknown baseline method: "
            f"{method}"
        )


# ============================================================
# 12. GENERATE ONE SPLIT
# ============================================================

def generate_split(
    samples,
    split_directory_name
):

    print()

    print("=" * 88)

    print(
        f"GENERATING SPLIT: "
        f"{split_directory_name}"
    )

    print("=" * 88)

    print()


    total_images = len(
        samples
    )


    for index, sample in enumerate(
        samples,
        start=1
    ):

        canonical_image = (
            load_canonical_image(
                sample[
                    "path"
                ]
            )
        )


        for baseline in BASELINES:

            output_image = apply_baseline(
                canonical_image,
                baseline
            )


            save_path = os.path.join(
                OUTPUT_ROOT,
                baseline[
                    "name"
                ],
                split_directory_name,
                sample[
                    "filename"
                ]
            )


            save_jpeg(
                output_image,
                save_path
            )


        if (
            index % 500 == 0
            or
            index == total_images
        ):

            print(
                f"Processed "
                f"{index}/"
                f"{total_images} "
                f"{split_directory_name} images"
            )


    print()


    print(
        f"Completed "
        f"{split_directory_name}."
    )


# ============================================================
# 13. GENERATE QUERY
# ============================================================

generate_split(
    query_samples,
    "query"
)


# ============================================================
# 14. GENERATE GALLERY
# ============================================================

generate_split(
    gallery_samples,
    "bounding_box_test"
)


# ============================================================
# 15. VERIFY OUTPUT COUNTS
# ============================================================

print()

print("=" * 88)

print(
    "VERIFYING GENERATED BASELINES"
)

print("=" * 88)

print()


for baseline in BASELINES:

    name = baseline[
        "name"
    ]


    query_outputs = glob.glob(
        os.path.join(
            OUTPUT_ROOT,
            name,
            "query",
            "*.jpg"
        )
    )


    gallery_outputs = glob.glob(
        os.path.join(
            OUTPUT_ROOT,
            name,
            "bounding_box_test",
            "*.jpg"
        )
    )


    print(
        name
    )


    print(
        f"  Query: "
        f"{len(query_outputs)} / "
        f"{len(query_samples)}"
    )


    print(
        f"  Gallery: "
        f"{len(gallery_outputs)} / "
        f"{len(gallery_samples)}"
    )


    if (
        len(
            query_outputs
        )
        !=
        len(
            query_samples
        )
    ):

        raise RuntimeError(
            f"Query output count mismatch "
            f"for {name}"
        )


    if (
        len(
            gallery_outputs
        )
        !=
        len(
            gallery_samples
        )
    ):

        raise RuntimeError(
            f"Gallery output count mismatch "
            f"for {name}"
        )


# ============================================================
# 16. FINISH
# ============================================================

print()

print("=" * 88)

print(
    "Phase 4A conventional baseline generation completed."
)

print("=" * 88)

print()


print(
    "Output root:"
)


print(
    OUTPUT_ROOT
)


print()


print(
    "Saved configuration:"
)


print(
    CONFIG_CSV
)


print()


print(
    "Generated:"
)


for baseline in BASELINES:

    print(
        f"  {baseline['name']}/"
    )


print()


print(
    "NEXT:"
)


print(
    "Evaluate utility and original-identity linkability "
    "for all conventional baselines using the same OSNet "
    "protocol used for the PIS experiments."
)