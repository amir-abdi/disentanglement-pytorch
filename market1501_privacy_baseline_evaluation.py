import os
import glob
import re
import csv
import random

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

import torchreid


# ============================================================
# 1. REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# 2. PATHS
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


BASELINE_ROOT = os.path.expanduser(
    "~/Desktop/ResearchResources/Phase2/"
    "disentanglement-pytorch/"
    "outputs/"
    "market1501_privacy_baselines"
)


OSNET_CHECKPOINT = os.path.expanduser(
    "~/Desktop/ResearchResources/Phase1/"
    "deep-person-reid/"
    "log/"
    "phase1_gpu_test/"
    "model/"
    "model.pth.tar-1"
)


# ------------------------------------------------------------
# Existing PIS results.
#
# These are optional. If found, the script will create a
# combined PIS vs conventional-baseline privacy-utility plot.
# ------------------------------------------------------------

PIS_UTILITY_CSV = os.path.expanduser(
    "~/Desktop/ResearchResources/Phase2/"
    "disentanglement-pytorch/"
    "outputs/"
    "market1501_pis_osnet_reid/"
    "pis_osnet_reid_results.csv"
)

PIS_PRIVACY_CSV = os.path.expanduser(
    "~/Desktop/ResearchResources/Phase2/"
    "disentanglement-pytorch/"
    "outputs/"
    "market1501_pis_privacy/"
    "pis_privacy_results.csv"
)


# ============================================================
# 3. BASELINES
# ============================================================

BASELINES = [
    {
        "name":
            "resized_control",

        "label":
            "Resize control",

        "group":
            "Control"
    },

    {
        "name":
            "blur_r2",

        "label":
            "Blur r=2",

        "group":
            "Blur"
    },

    {
        "name":
            "blur_r4",

        "label":
            "Blur r=4",

        "group":
            "Blur"
    },

    {
        "name":
            "blur_r8",

        "label":
            "Blur r=8",

        "group":
            "Blur"
    },

    {
        "name":
            "pixel_32x64",

        "label":
            "Pixel 32x64",

        "group":
            "Pixelation"
    },

    {
        "name":
            "pixel_16x32",

        "label":
            "Pixel 16x32",

        "group":
            "Pixelation"
    },

    {
        "name":
            "pixel_8x16",

        "label":
            "Pixel 8x16",

        "group":
            "Pixelation"
    }
]


# ============================================================
# 4. OSNET / EVALUATION CONFIGURATION
# ============================================================

MODEL_NAME = "osnet_x1_0"

NUM_CLASSES = 751

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 128

BATCH_SIZE = 64

NUM_WORKERS = 4

QUERY_CHUNK_SIZE = 128


RANKS = [
    1,
    5,
    10,
    20
]


# ============================================================
# 5. OUTPUT
# ============================================================

OUTPUT_DIR = os.path.expanduser(
    "~/Desktop/ResearchResources/Phase2/"
    "disentanglement-pytorch/"
    "outputs/"
    "market1501_privacy_baseline_evaluation"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


RESULTS_CSV = os.path.join(
    OUTPUT_DIR,
    "privacy_baseline_results.csv"
)

UTILITY_PLOT = os.path.join(
    OUTPUT_DIR,
    "baseline_utility_rank1.png"
)

PRIVACY_PLOT = os.path.join(
    OUTPUT_DIR,
    "baseline_source_attack_rank1.png"
)

SIMILARITY_PLOT = os.path.join(
    OUTPUT_DIR,
    "baseline_feature_similarity.png"
)

TRADEOFF_PLOT = os.path.join(
    OUTPUT_DIR,
    "baseline_privacy_utility_tradeoff.png"
)

COMBINED_TRADEOFF_PLOT = os.path.join(
    OUTPUT_DIR,
    "pis_vs_conventional_privacy_utility.png"
)


# ============================================================
# 6. DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("=" * 96)

print(
    "Phase 4B - Conventional Privacy Baseline Evaluation"
)

print("=" * 96)

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

if not os.path.exists(
    OSNET_CHECKPOINT
):

    raise FileNotFoundError(
        "OSNet checkpoint not found:\n"
        f"{OSNET_CHECKPOINT}"
    )


for baseline in BASELINES:

    query_dir = os.path.join(
        BASELINE_ROOT,
        baseline[
            "name"
        ],
        "query"
    )

    gallery_dir = os.path.join(
        BASELINE_ROOT,
        baseline[
            "name"
        ],
        "bounding_box_test"
    )

    if not os.path.isdir(
        query_dir
    ):

        raise FileNotFoundError(
            f"Missing query directory:\n"
            f"{query_dir}"
        )

    if not os.path.isdir(
        gallery_dir
    ):

        raise FileNotFoundError(
            f"Missing gallery directory:\n"
            f"{gallery_dir}"
        )


print(
    "All baseline paths verified."
)

print()


# ============================================================
# 8. PREPROCESSING
# ============================================================

transform = transforms.Compose(
    [
        transforms.Resize(
            (
                IMAGE_HEIGHT,
                IMAGE_WIDTH
            )
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406
            ],

            std=[
                0.229,
                0.224,
                0.225
            ]
        )
    ]
)


# ============================================================
# 9. MARKET-1501 DATASET
# ============================================================

class Market1501Dataset(
    Dataset
):

    def __init__(
        self,
        image_dir,
        split_name
    ):

        image_paths = sorted(
            glob.glob(
                os.path.join(
                    image_dir,
                    "*.jpg"
                )
            )
        )


        if len(
            image_paths
        ) == 0:

            raise RuntimeError(
                f"No JPG images found in:\n"
                f"{image_dir}"
            )


        pattern = re.compile(
            r"([-\d]+)_c(\d)"
        )


        self.samples = []


        for path in image_paths:

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


            if pid == -1:

                continue


            self.samples.append(
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


        print(
            f"{split_name}: "
            f"{len(self.samples)} images"
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

        sample = self.samples[
            index
        ]


        image = Image.open(
            sample[
                "path"
            ]
        ).convert(
            "RGB"
        )


        image = transform(
            image
        )


        return (
            image,
            sample[
                "pid"
            ],
            sample[
                "camid"
            ],
            sample[
                "filename"
            ]
        )


# ============================================================
# 10. BUILD OSNET
# ============================================================

print("=" * 96)

print(
    "Loading OSNet..."
)

print("=" * 96)

print()


model = torchreid.models.build_model(
    name=MODEL_NAME,
    num_classes=NUM_CLASSES,
    loss="softmax",
    pretrained=False
).to(
    device
)


checkpoint = torch.load(
    OSNET_CHECKPOINT,
    map_location=device,
    weights_only=False
)


if (
    isinstance(
        checkpoint,
        dict
    )
    and
    "state_dict"
    in checkpoint
):

    state_dict = checkpoint[
        "state_dict"
    ]

elif (
    isinstance(
        checkpoint,
        dict
    )
    and
    "model_state_dict"
    in checkpoint
):

    state_dict = checkpoint[
        "model_state_dict"
    ]

else:

    state_dict = checkpoint


clean_state_dict = {}


for key, value in state_dict.items():

    if key.startswith(
        "module."
    ):

        key = key[
            7:
        ]


    clean_state_dict[
        key
    ] = value


model.load_state_dict(
    clean_state_dict,
    strict=False
)


model.eval()


for parameter in model.parameters():

    parameter.requires_grad = False


print(
    "OSNet loaded and frozen."
)

print()


# ============================================================
# 11. FEATURE EXTRACTION
# ============================================================

def extract_features(
    image_dir,
    split_name
):

    dataset = Market1501Dataset(
        image_dir,
        split_name
    )


    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )


    feature_batches = []

    pid_batches = []

    camid_batches = []

    filenames = []


    with torch.inference_mode():

        for (
            images,
            pids,
            camids,
            batch_filenames
        ) in loader:

            images = images.to(
                device,
                non_blocking=True
            )


            features = model(
                images
            )


            features = F.normalize(
                features,
                p=2,
                dim=1
            )


            feature_batches.append(
                features.cpu()
            )


            pid_batches.append(
                pids.cpu()
            )


            camid_batches.append(
                camids.cpu()
            )


            filenames.extend(
                list(
                    batch_filenames
                )
            )


    return {
        "features":
            torch.cat(
                feature_batches,
                dim=0
            ).float(),

        "pids":
            torch.cat(
                pid_batches,
                dim=0
            ).numpy(),

        "camids":
            torch.cat(
                camid_batches,
                dim=0
            ).numpy(),

        "filenames":
            filenames
    }


# ============================================================
# 12. AVERAGE PRECISION
# ============================================================

def compute_average_precision(
    matches
):

    relevant = np.where(
        matches == 1
    )[0]


    if len(
        relevant
    ) == 0:

        return None


    cumulative = np.cumsum(
        matches
    )


    precision_values = []


    for index in relevant:

        precision_values.append(
            cumulative[
                index
            ]
            /
            (
                index + 1
            )
        )


    return float(
        np.mean(
            precision_values
        )
    )


# ============================================================
# 13. MARKET-1501 REID EVALUATION
# ============================================================

def evaluate_reid(
    query_features,
    gallery_features,
    query_pids,
    gallery_pids,
    query_camids,
    gallery_camids
):

    number_queries = query_features.shape[
        0
    ]


    number_gallery = gallery_features.shape[
        0
    ]


    cmc = np.zeros(
        number_gallery,
        dtype=np.float64
    )


    average_precisions = []


    valid_queries = 0


    gallery_gpu = gallery_features.to(
        device
    )


    for start in range(
        0,
        number_queries,
        QUERY_CHUNK_SIZE
    ):

        end = min(
            start
            +
            QUERY_CHUNK_SIZE,
            number_queries
        )


        query_gpu = query_features[
            start:end
        ].to(
            device
        )


        similarity = torch.mm(
            query_gpu,
            gallery_gpu.T
        )


        distances = (
            2.0
            -
            2.0
            *
            similarity
        ).cpu().numpy()


        for local_index in range(
            distances.shape[
                0
            ]
        ):

            query_index = (
                start
                +
                local_index
            )


            qpid = query_pids[
                query_index
            ]


            qcam = query_camids[
                query_index
            ]


            order = np.argsort(
                distances[
                    local_index
                ]
            )


            ordered_pids = gallery_pids[
                order
            ]


            ordered_camids = gallery_camids[
                order
            ]


            # ------------------------------------------------
            # Same Market-1501 protocol used throughout all
            # PIS experiments.
            # ------------------------------------------------

            remove = (
                (
                    ordered_pids
                    ==
                    qpid
                )
                &
                (
                    ordered_camids
                    ==
                    qcam
                )
            )


            keep = np.logical_not(
                remove
            )


            ordered_pids = ordered_pids[
                keep
            ]


            matches = (
                ordered_pids
                ==
                qpid
            ).astype(
                np.int32
            )


            if not np.any(
                matches
            ):

                continue


            valid_queries += 1


            first_match = np.where(
                matches == 1
            )[0][0]


            cmc[
                first_match:
            ] += 1


            ap = compute_average_precision(
                matches
            )


            if ap is not None:

                average_precisions.append(
                    ap
                )


    if valid_queries == 0:

        raise RuntimeError(
            "No valid ReID queries found."
        )


    cmc /= valid_queries


    result = {
        "valid_queries":
            valid_queries,

        "mAP":
            float(
                np.mean(
                    average_precisions
                )
                *
                100
            )
    }


    for rank in RANKS:

        result[
            f"Rank-{rank}"
        ] = (
            cmc[
                rank - 1
            ]
            *
            100
        )


    return result


# ============================================================
# 14. SAME-IMAGE FEATURE SIMILARITY
# ============================================================

def feature_similarity(
    original_outputs,
    transformed_outputs
):

    if (
        original_outputs[
            "filenames"
        ]
        !=
        transformed_outputs[
            "filenames"
        ]
    ):

        raise RuntimeError(
            "Original and transformed filenames "
            "do not match."
        )


    original_features = (
        original_outputs[
            "features"
        ]
    )


    transformed_features = (
        transformed_outputs[
            "features"
        ]
    )


    cosine = F.cosine_similarity(
        original_features,
        transformed_features,
        dim=1
    )


    euclidean = torch.norm(
        original_features
        -
        transformed_features,
        p=2,
        dim=1
    )


    return {
        "mean_cosine":
            float(
                cosine.mean().item()
            ),

        "mean_euclidean":
            float(
                euclidean.mean().item()
            )
    }


# ============================================================
# 15. EXTRACT ORIGINAL FEATURES ONCE
# ============================================================

print("=" * 96)

print(
    "Extracting original Market-1501 features"
)

print("=" * 96)

print()


original_query = extract_features(
    ORIGINAL_QUERY_DIR,
    "Original Query"
)


original_gallery = extract_features(
    ORIGINAL_GALLERY_DIR,
    "Original Gallery"
)


print()


# ============================================================
# 16. ORIGINAL BASELINE
# ============================================================

original_result = evaluate_reid(
    original_query[
        "features"
    ],

    original_gallery[
        "features"
    ],

    original_query[
        "pids"
    ],

    original_gallery[
        "pids"
    ],

    original_query[
        "camids"
    ],

    original_gallery[
        "camids"
    ]
)


print(
    "Original Market-1501:"
)

print(
    f"  Rank-1: "
    f"{original_result['Rank-1']:.2f}%"
)

print(
    f"  mAP: "
    f"{original_result['mAP']:.2f}%"
)

print()


# ============================================================
# 17. EVALUATE BASELINES
# ============================================================

results = []


for baseline in BASELINES:

    name = baseline[
        "name"
    ]


    label = baseline[
        "label"
    ]


    group = baseline[
        "group"
    ]


    print()

    print("=" * 96)

    print(
        f"EVALUATING: {label}"
    )

    print("=" * 96)

    print()


    query_dir = os.path.join(
        BASELINE_ROOT,
        name,
        "query"
    )


    gallery_dir = os.path.join(
        BASELINE_ROOT,
        name,
        "bounding_box_test"
    )


    transformed_query = extract_features(
        query_dir,
        f"{label} Query"
    )


    transformed_gallery = extract_features(
        gallery_dir,
        f"{label} Gallery"
    )


    # ========================================================
    # UTILITY
    #
    # transformed query -> transformed gallery
    # ========================================================

    utility = evaluate_reid(
        transformed_query[
            "features"
        ],

        transformed_gallery[
            "features"
        ],

        transformed_query[
            "pids"
        ],

        transformed_gallery[
            "pids"
        ],

        transformed_query[
            "camids"
        ],

        transformed_gallery[
            "camids"
        ]
    )


    # ========================================================
    # PRIVACY ATTACK A
    #
    # original query -> transformed gallery
    # ========================================================

    source_attack = evaluate_reid(
        original_query[
            "features"
        ],

        transformed_gallery[
            "features"
        ],

        original_query[
            "pids"
        ],

        transformed_gallery[
            "pids"
        ],

        original_query[
            "camids"
        ],

        transformed_gallery[
            "camids"
        ]
    )


    # ========================================================
    # PRIVACY ATTACK B
    #
    # transformed query -> original gallery
    # ========================================================

    reverse_attack = evaluate_reid(
        transformed_query[
            "features"
        ],

        original_gallery[
            "features"
        ],

        transformed_query[
            "pids"
        ],

        original_gallery[
            "pids"
        ],

        transformed_query[
            "camids"
        ],

        original_gallery[
            "camids"
        ]
    )


    # ========================================================
    # FEATURE SIMILARITY
    # ========================================================

    query_similarity = feature_similarity(
        original_query,
        transformed_query
    )


    gallery_similarity = feature_similarity(
        original_gallery,
        transformed_gallery
    )


    number_query = len(
        original_query[
            "filenames"
        ]
    )


    number_gallery = len(
        original_gallery[
            "filenames"
        ]
    )


    total_images = (
        number_query
        +
        number_gallery
    )


    combined_cosine = (
        query_similarity[
            "mean_cosine"
        ]
        *
        number_query

        +

        gallery_similarity[
            "mean_cosine"
        ]
        *
        number_gallery
    ) / total_images


    combined_euclidean = (
        query_similarity[
            "mean_euclidean"
        ]
        *
        number_query

        +

        gallery_similarity[
            "mean_euclidean"
        ]
        *
        number_gallery
    ) / total_images


    row = {
        "name":
            name,

        "label":
            label,

        "group":
            group,

        "utility_mAP":
            utility[
                "mAP"
            ],

        "utility_rank1":
            utility[
                "Rank-1"
            ],

        "utility_rank5":
            utility[
                "Rank-5"
            ],

        "utility_rank10":
            utility[
                "Rank-10"
            ],

        "utility_rank20":
            utility[
                "Rank-20"
            ],

        "source_attack_mAP":
            source_attack[
                "mAP"
            ],

        "source_attack_rank1":
            source_attack[
                "Rank-1"
            ],

        "source_attack_rank5":
            source_attack[
                "Rank-5"
            ],

        "source_attack_rank10":
            source_attack[
                "Rank-10"
            ],

        "reverse_attack_mAP":
            reverse_attack[
                "mAP"
            ],

        "reverse_attack_rank1":
            reverse_attack[
                "Rank-1"
            ],

        "feature_cosine":
            combined_cosine,

        "feature_euclidean":
            combined_euclidean
    }


    results.append(
        row
    )


    print(
        f"{label} RESULTS"
    )

    print(
        "-" * 68
    )


    print(
        "Utility:"
    )

    print(
        f"  Rank-1: "
        f"{utility['Rank-1']:.2f}%"
    )

    print(
        f"  mAP: "
        f"{utility['mAP']:.2f}%"
    )


    print()


    print(
        "Original Query -> Transformed Gallery:"
    )

    print(
        f"  Rank-1: "
        f"{source_attack['Rank-1']:.2f}%"
    )

    print(
        f"  mAP: "
        f"{source_attack['mAP']:.2f}%"
    )


    print()


    print(
        "Transformed Query -> Original Gallery:"
    )

    print(
        f"  Rank-1: "
        f"{reverse_attack['Rank-1']:.2f}%"
    )


    print()


    print(
        "Original <-> transformed features:"
    )

    print(
        f"  Cosine: "
        f"{combined_cosine:.4f}"
    )

    print(
        f"  Euclidean: "
        f"{combined_euclidean:.4f}"
    )


    del transformed_query
    del transformed_gallery


    if torch.cuda.is_available():

        torch.cuda.empty_cache()


# ============================================================
# 18. FINAL TABLE
# ============================================================

print()

print("=" * 116)

print(
    "FINAL CONVENTIONAL PRIVACY BASELINE RESULTS"
)

print("=" * 116)

print()


print(
    f"{'Method':<18}"
    f"{'Utility R1':>12}"
    f"{'Utility mAP':>14}"
    f"{'Attack R1':>12}"
    f"{'Attack mAP':>14}"
    f"{'Rev R1':>10}"
    f"{'Cosine':>10}"
)


print(
    "-" * 92
)


for row in results:

    print(
        f"{row['label']:<18}"
        f"{row['utility_rank1']:>11.2f}%"
        f"{row['utility_mAP']:>13.2f}%"
        f"{row['source_attack_rank1']:>11.2f}%"
        f"{row['source_attack_mAP']:>13.2f}%"
        f"{row['reverse_attack_rank1']:>9.2f}%"
        f"{row['feature_cosine']:>10.4f}"
    )


# ============================================================
# 19. SAVE CSV
# ============================================================

with open(
    RESULTS_CSV,
    "w",
    newline=""
) as csv_file:

    writer = csv.DictWriter(
        csv_file,
        fieldnames=list(
            results[
                0
            ].keys()
        )
    )


    writer.writeheader()


    writer.writerows(
        results
    )


# ============================================================
# 20. BASELINE UTILITY PLOT
# ============================================================

labels = [
    row[
        "label"
    ]
    for row in results
]


utility_values = [
    row[
        "utility_rank1"
    ]
    for row in results
]


x_positions = list(
    range(
        len(
            labels
        )
    )
)


plt.figure(
    figsize=(12, 6)
)


plt.bar(
    x_positions,
    utility_values
)


plt.axhline(
    original_result[
        "Rank-1"
    ],
    linestyle="--",
    label="Original"
)


plt.xticks(
    x_positions,
    labels,
    rotation=30,
    ha="right"
)


plt.ylabel(
    "Transformed-Domain Rank-1 (%)"
)


plt.title(
    "Conventional Baseline ReID Utility"
)


plt.legend()


plt.grid(
    True,
    alpha=0.3,
    axis="y"
)


plt.tight_layout()


plt.savefig(
    UTILITY_PLOT,
    dpi=150
)


plt.close()


# ============================================================
# 21. SOURCE ATTACK PLOT
# ============================================================

attack_values = [
    row[
        "source_attack_rank1"
    ]
    for row in results
]


plt.figure(
    figsize=(12, 6)
)


plt.bar(
    x_positions,
    attack_values
)


plt.axhline(
    original_result[
        "Rank-1"
    ],
    linestyle="--",
    label="Original -> Original"
)


plt.xticks(
    x_positions,
    labels,
    rotation=30,
    ha="right"
)


plt.ylabel(
    "Original-Identity Attack Rank-1 (%)"
)


plt.title(
    "Conventional Baseline Source-Identity Linkability"
)


plt.legend()


plt.grid(
    True,
    alpha=0.3,
    axis="y"
)


plt.tight_layout()


plt.savefig(
    PRIVACY_PLOT,
    dpi=150
)


plt.close()


# ============================================================
# 22. FEATURE SIMILARITY PLOT
# ============================================================

cosine_values = [
    row[
        "feature_cosine"
    ]
    for row in results
]


plt.figure(
    figsize=(12, 6)
)


plt.plot(
    x_positions,
    cosine_values,
    marker="o"
)


plt.xticks(
    x_positions,
    labels,
    rotation=30,
    ha="right"
)


plt.ylabel(
    "Original-to-Transformed OSNet Cosine"
)


plt.title(
    "Feature Similarity of Conventional Privacy Baselines"
)


plt.grid(
    True,
    alpha=0.3
)


plt.tight_layout()


plt.savefig(
    SIMILARITY_PLOT,
    dpi=150
)


plt.close()


# ============================================================
# 23. CONVENTIONAL PRIVACY-UTILITY PLOT
# ============================================================

plt.figure(
    figsize=(10, 7)
)


for row in results:

    plt.scatter(
        row[
            "source_attack_rank1"
        ],
        row[
            "utility_rank1"
        ],
        s=90
    )


    plt.annotate(
        row[
            "label"
        ],
        (
            row[
                "source_attack_rank1"
            ],
            row[
                "utility_rank1"
            ]
        ),
        xytext=(
            5,
            5
        ),
        textcoords="offset points"
    )


plt.xlabel(
    "Original-Identity Attack Rank-1 (%) ↓"
)


plt.ylabel(
    "Transformed-Domain Utility Rank-1 (%) ↑"
)


plt.title(
    "Conventional Privacy-Utility Operating Points"
)


plt.grid(
    True,
    alpha=0.3
)


plt.tight_layout()


plt.savefig(
    TRADEOFF_PLOT,
    dpi=150
)


plt.close()


# ============================================================
# 24. COMBINED PIS VS CONVENTIONAL PLOT
# ============================================================

if (
    os.path.exists(
        PIS_UTILITY_CSV
    )
    and
    os.path.exists(
        PIS_PRIVACY_CSV
    )
):

    pis_utility = {}

    with open(
        PIS_UTILITY_CSV,
        "r"
    ) as csv_file:

        reader = csv.DictReader(
            csv_file
        )


        for row in reader:

            if row[
                "alpha"
            ] == "":

                continue


            alpha = float(
                row[
                    "alpha"
                ]
            )


            pis_utility[
                alpha
            ] = float(
                row[
                    "Rank-1"
                ]
            )


    pis_privacy = {}

    with open(
        PIS_PRIVACY_CSV,
        "r"
    ) as csv_file:

        reader = csv.DictReader(
            csv_file
        )


        for row in reader:

            alpha = float(
                row[
                    "alpha"
                ]
            )


            pis_privacy[
                alpha
            ] = float(
                row[
                    "oq_sg_rank1"
                ]
            )


    plt.figure(
        figsize=(11, 8)
    )


    # --------------------------------------------------------
    # Conventional baselines
    # --------------------------------------------------------

    for row in results:

        plt.scatter(
            row[
                "source_attack_rank1"
            ],
            row[
                "utility_rank1"
            ],
            s=80
        )


        plt.annotate(
            row[
                "label"
            ],
            (
                row[
                    "source_attack_rank1"
                ],
                row[
                    "utility_rank1"
                ]
            ),
            xytext=(
                5,
                5
            ),
            textcoords="offset points"
        )


    # --------------------------------------------------------
    # PIS operating points
    # --------------------------------------------------------

    common_alphas = sorted(
        set(
            pis_utility.keys()
        ).intersection(
            pis_privacy.keys()
        )
    )


    pis_x = [
        pis_privacy[
            alpha
        ]
        for alpha in common_alphas
    ]


    pis_y = [
        pis_utility[
            alpha
        ]
        for alpha in common_alphas
    ]


    plt.plot(
        pis_x,
        pis_y,
        marker="o",
        label="PIS"
    )


    for alpha, x_value, y_value in zip(
        common_alphas,
        pis_x,
        pis_y
    ):

        plt.annotate(
            f"PIS α={alpha:.2f}",
            (
                x_value,
                y_value
            ),
            xytext=(
                5,
                5
            ),
            textcoords="offset points"
        )


    plt.xlabel(
        "Original-Identity Attack Rank-1 (%) ↓"
    )


    plt.ylabel(
        "Transformed-Domain ReID Rank-1 (%) ↑"
    )


    plt.title(
        "PIS vs Conventional Privacy Methods"
    )


    plt.legend()


    plt.grid(
        True,
        alpha=0.3
    )


    plt.tight_layout()


    plt.savefig(
        COMBINED_TRADEOFF_PLOT,
        dpi=150
    )


    plt.close()


    print()

    print(
        "Combined PIS vs conventional "
        "privacy-utility plot created."
    )


else:

    print()

    print(
        "PIS result CSVs not found. "
        "Combined comparison plot skipped."
    )


# ============================================================
# 25. FINISH
# ============================================================

print()

print("=" * 96)

print(
    "Phase 4B conventional privacy baseline evaluation completed."
)

print("=" * 96)

print()


print(
    "Original OSNet baseline:"
)

print(
    f"  Rank-1: "
    f"{original_result['Rank-1']:.2f}%"
)

print(
    f"  mAP: "
    f"{original_result['mAP']:.2f}%"
)

print()


print(
    "Saved:"
)

print(
    RESULTS_CSV
)

print(
    UTILITY_PLOT
)

print(
    PRIVACY_PLOT
)

print(
    SIMILARITY_PLOT
)

print(
    TRADEOFF_PLOT
)


if (
    os.path.exists(
        PIS_UTILITY_CSV
    )
    and
    os.path.exists(
        PIS_PRIVACY_CSV
    )
):

    print(
        COMBINED_TRADEOFF_PLOT
    )