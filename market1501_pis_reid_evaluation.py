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
# 1. CONFIGURATION
# ============================================================

RANDOM_SEED = 42


random.seed(
    RANDOM_SEED
)

np.random.seed(
    RANDOM_SEED
)

torch.manual_seed(
    RANDOM_SEED
)


if torch.cuda.is_available():

    torch.cuda.manual_seed_all(
        RANDOM_SEED
    )


# ============================================================
# ORIGINAL MARKET-1501
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


# ============================================================
# GENERATED PIS DATASETS
# ============================================================

PIS_ROOT = os.path.expanduser(
    "~/Desktop/ResearchResources/Phase2/"
    "disentanglement-pytorch/"
    "outputs/"
    "market1501_full_pis"
)


# ============================================================
# OSNET CHECKPOINT
# ============================================================

OSNET_CHECKPOINT = os.path.expanduser(
    "~/Desktop/ResearchResources/Phase1/"
    "deep-person-reid/"
    "log/"
    "phase1_gpu_test/"
    "model/"
    "model.pth.tar-1"
)


# ============================================================
# 2. DATASETS TO EVALUATE
# ============================================================

# ------------------------------------------------------------
# Original:
#
# real Market-1501 photographs
#
# alpha=0:
#
# VAE reconstruction only
#
# alpha>0:
#
# increasingly stronger PIS transformation
# ------------------------------------------------------------

DATASETS = [
    {
        "name":
            "Original",

        "alpha":
            None,

        "query_dir":
            ORIGINAL_QUERY_DIR,

        "gallery_dir":
            ORIGINAL_GALLERY_DIR
    },

    {
        "name":
            "Alpha 0.00",

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
            "Alpha 0.25",

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
            "Alpha 0.50",

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
            "Alpha 0.75",

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
            "Alpha 1.00",

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
    }
]


# ============================================================
# 3. OSNET CONFIGURATION
# ============================================================

MODEL_NAME = "osnet_x1_0"


# ------------------------------------------------------------
# We use 751 because that was the number of training
# identities in the Phase 1 Market-1501 model.
#
# In evaluation mode OSNet returns its embedding rather than
# classifier logits.
# ------------------------------------------------------------

NUM_CLASSES = 751


IMAGE_HEIGHT = 256

IMAGE_WIDTH = 128


BATCH_SIZE = 64

NUM_WORKERS = 4


QUERY_CHUNK_SIZE = 128


RANKS_TO_REPORT = [
    1,
    5,
    10,
    20
]


# ============================================================
# 4. OUTPUT
# ============================================================

OUTPUT_DIR = os.path.expanduser(
    "~/Desktop/ResearchResources/Phase2/"
    "disentanglement-pytorch/"
    "outputs/"
    "market1501_pis_osnet_reid"
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


CSV_PATH = os.path.join(
    OUTPUT_DIR,
    "pis_osnet_reid_results.csv"
)


RANK1_PLOT_PATH = os.path.join(
    OUTPUT_DIR,
    "rank1_vs_alpha.png"
)


MAP_PLOT_PATH = os.path.join(
    OUTPUT_DIR,
    "map_vs_alpha.png"
)


CMC_PLOT_PATH = os.path.join(
    OUTPUT_DIR,
    "cmc_metrics_vs_alpha.png"
)


# ============================================================
# 5. DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("=" * 88)

print(
    "Phase 3C - Market-1501 PIS Image-Level ReID Evaluation"
)

print("=" * 88)


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
    "OSNet checkpoint:"
)


print(
    OSNET_CHECKPOINT
)


print()


# ============================================================
# 6. VERIFY REQUIRED PATHS
# ============================================================

if not os.path.exists(
    OSNET_CHECKPOINT
):

    raise FileNotFoundError(
        "OSNet checkpoint not found:\n"
        f"{OSNET_CHECKPOINT}"
    )


for dataset_info in DATASETS:

    if not os.path.isdir(
        dataset_info[
            "query_dir"
        ]
    ):

        raise FileNotFoundError(
            "Query directory not found:\n"
            f"{dataset_info['query_dir']}"
        )


    if not os.path.isdir(
        dataset_info[
            "gallery_dir"
        ]
    ):

        raise FileNotFoundError(
            "Gallery directory not found:\n"
            f"{dataset_info['gallery_dir']}"
        )


print(
    "All required paths verified."
)


print()


# ============================================================
# 7. OSNET PREPROCESSING
# ============================================================

# ------------------------------------------------------------
# Torchreid OSNet normally uses ImageNet-style normalization.
# ------------------------------------------------------------

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
# 8. MARKET-1501 DATASET
# ============================================================

class Market1501ReIDDataset(
    Dataset
):

    def __init__(
        self,
        image_dir,
        transform=None,
        split_name="unknown"
    ):

        self.transform = transform

        self.split_name = split_name


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
                f"No JPG images found in "
                f"{split_name}:\n"
                f"{image_dir}"
            )


        pattern = re.compile(
            r"([-\d]+)_c(\d)"
        )


        self.samples = []


        for image_path in image_paths:

            filename = os.path.basename(
                image_path
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


            # ------------------------------------------------
            # Market-1501 junk identities.
            # ------------------------------------------------

            if pid == -1:

                continue


            self.samples.append(
                {
                    "path":
                        image_path,

                    "pid":
                        pid,

                    "camid":
                        camid
                }
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


        if self.transform is not None:

            image = self.transform(
                image
            )


        return (
            image,
            sample[
                "pid"
            ],
            sample[
                "camid"
            ]
        )


# ============================================================
# 9. BUILD OSNET
# ============================================================

print("=" * 88)

print(
    "Building OSNet model..."
)

print("=" * 88)

print()


model = torchreid.models.build_model(
    name=MODEL_NAME,
    num_classes=NUM_CLASSES,
    loss="softmax",
    pretrained=False
)


model = model.to(
    device
)


print(
    "OSNet architecture created."
)


# ============================================================
# 10. LOAD PHASE 1 CHECKPOINT
# ============================================================

checkpoint = torch.load(
    OSNET_CHECKPOINT,
    map_location=device,
    weights_only=False
)


# ------------------------------------------------------------
# Torchreid training checkpoints normally contain
# state_dict.
#
# We also support a direct state_dict just in case.
# ------------------------------------------------------------

if isinstance(
    checkpoint,
    dict
) and (
    "state_dict"
    in checkpoint
):

    state_dict = checkpoint[
        "state_dict"
    ]


elif isinstance(
    checkpoint,
    dict
) and (
    "model_state_dict"
    in checkpoint
):

    state_dict = checkpoint[
        "model_state_dict"
    ]


else:

    state_dict = checkpoint


# ------------------------------------------------------------
# Remove DataParallel prefix if present.
# ------------------------------------------------------------

clean_state_dict = {}


for key, value in state_dict.items():

    clean_key = key


    if clean_key.startswith(
        "module."
    ):

        clean_key = clean_key[
            7:
        ]


    clean_state_dict[
        clean_key
    ] = value


load_result = model.load_state_dict(
    clean_state_dict,
    strict=False
)


print(
    "OSNet checkpoint loaded."
)


print()


if len(
    load_result.missing_keys
) > 0:

    print(
        "Missing keys:"
    )


    for key in load_result.missing_keys:

        print(
            " ",
            key
        )


    print()


if len(
    load_result.unexpected_keys
) > 0:

    print(
        "Unexpected keys:"
    )


    for key in load_result.unexpected_keys:

        print(
            " ",
            key
        )


    print()


model.eval()


for parameter in model.parameters():

    parameter.requires_grad = False


print(
    "OSNet frozen in evaluation mode."
)


print()


# ============================================================
# 11. FEATURE EXTRACTION
# ============================================================

def extract_osnet_features(
    data_loader
):

    features = []

    all_pids = []

    all_camids = []


    with torch.inference_mode():

        for (
            images,
            pids,
            camids
        ) in data_loader:

            images = images.to(
                device,
                non_blocking=True
            )


            # ------------------------------------------------
            # In eval mode OSNet outputs embeddings.
            # ------------------------------------------------

            embeddings = model(
                images
            )


            embeddings = F.normalize(
                embeddings,
                p=2,
                dim=1
            )


            features.append(
                embeddings.cpu()
            )


            all_pids.append(
                pids.cpu()
            )


            all_camids.append(
                camids.cpu()
            )


    features = torch.cat(
        features,
        dim=0
    ).float()


    pids = torch.cat(
        all_pids,
        dim=0
    ).numpy()


    camids = torch.cat(
        all_camids,
        dim=0
    ).numpy()


    return (
        features,
        pids,
        camids
    )


# ============================================================
# 12. AVERAGE PRECISION
# ============================================================

def compute_average_precision(
    matches
):

    relevant_indices = np.where(
        matches == 1
    )[0]


    if len(
        relevant_indices
    ) == 0:

        return None


    cumulative_matches = np.cumsum(
        matches
    )


    precision_values = []


    for rank_index in relevant_indices:

        precision_at_rank = (
            cumulative_matches[
                rank_index
            ]
            /
            (
                rank_index
                +
                1
            )
        )


        precision_values.append(
            precision_at_rank
        )


    return float(
        np.mean(
            precision_values
        )
    )


# ============================================================
# 13. MARKET-1501 RETRIEVAL EVALUATION
# ============================================================

def evaluate_market1501(
    query_features,
    gallery_features,
    query_pids,
    gallery_pids,
    query_camids,
    gallery_camids
):

    number_queries = (
        query_features.shape[
            0
        ]
    )


    number_gallery = (
        gallery_features.shape[
            0
        ]
    )


    cmc = np.zeros(
        number_gallery,
        dtype=np.float64
    )


    average_precisions = []


    valid_queries = 0


    gallery_device = gallery_features.to(
        device
    )


    for start_index in range(
        0,
        number_queries,
        QUERY_CHUNK_SIZE
    ):

        end_index = min(
            start_index
            +
            QUERY_CHUNK_SIZE,
            number_queries
        )


        query_chunk = query_features[
            start_index:
            end_index
        ].to(
            device
        )


        # ----------------------------------------------------
        # Features are L2 normalized.
        #
        # Squared Euclidean distance:
        #
        # ||q-g||² = 2 - 2(q·g)
        # ----------------------------------------------------

        similarity = torch.mm(
            query_chunk,
            gallery_device.T
        )


        distances = (
            2.0
            -
            2.0
            *
            similarity
        )


        distances = (
            distances
            .cpu()
            .numpy()
        )


        for local_index in range(
            distances.shape[
                0
            ]
        ):

            query_index = (
                start_index
                +
                local_index
            )


            query_pid = query_pids[
                query_index
            ]


            query_camid = query_camids[
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
            # Standard Market-1501:
            #
            # same PID + same camera is excluded.
            # ------------------------------------------------

            remove = (
                (
                    ordered_pids
                    ==
                    query_pid
                )
                &
                (
                    ordered_camids
                    ==
                    query_camid
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
                query_pid
            ).astype(
                np.int32
            )


            if not np.any(
                matches
            ):

                continue


            valid_queries += 1


            # =================================================
            # CMC
            # =================================================

            first_match_index = (
                np.where(
                    matches == 1
                )[0][0]
            )


            cmc[
                first_match_index:
            ] += 1


            # =================================================
            # AVERAGE PRECISION
            # =================================================

            average_precision = (
                compute_average_precision(
                    matches
                )
            )


            if average_precision is not None:

                average_precisions.append(
                    average_precision
                )


    if valid_queries == 0:

        raise RuntimeError(
            "No valid Market-1501 queries."
        )


    cmc = (
        cmc
        /
        valid_queries
    )


    mean_ap = (
        float(
            np.mean(
                average_precisions
            )
        )
        *
        100
    )


    result = {
        "valid_queries":
            valid_queries,

        "mAP":
            mean_ap
    }


    for rank in RANKS_TO_REPORT:

        result[
            f"Rank-{rank}"
        ] = (
            cmc[
                rank
                -
                1
            ]
            *
            100
        )


    return result


# ============================================================
# 14. EVALUATE ONE IMAGE DATASET
# ============================================================

def evaluate_dataset(
    dataset_info
):

    dataset_name = (
        dataset_info[
            "name"
        ]
    )


    alpha = (
        dataset_info[
            "alpha"
        ]
    )


    query_dir = (
        dataset_info[
            "query_dir"
        ]
    )


    gallery_dir = (
        dataset_info[
            "gallery_dir"
        ]
    )


    print()

    print("=" * 88)


    print(
        f"EVALUATING: "
        f"{dataset_name}"
    )


    print("=" * 88)


    print(
        "Query:"
    )


    print(
        query_dir
    )


    print(
        "Gallery:"
    )


    print(
        gallery_dir
    )


    print()


    query_dataset = Market1501ReIDDataset(
        query_dir,
        transform=transform,
        split_name=(
            f"{dataset_name} Query"
        )
    )


    gallery_dataset = Market1501ReIDDataset(
        gallery_dir,
        transform=transform,
        split_name=(
            f"{dataset_name} Gallery"
        )
    )


    print(
        f"Query images: "
        f"{len(query_dataset)}"
    )


    print(
        f"Gallery images: "
        f"{len(gallery_dataset)}"
    )


    query_loader = DataLoader(
        query_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )


    gallery_loader = DataLoader(
        gallery_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )


    # ========================================================
    # EXTRACT QUERY FEATURES
    # ========================================================

    print(
        "Extracting query OSNet features..."
    )


    (
        query_features,
        query_pids,
        query_camids
    ) = extract_osnet_features(
        query_loader
    )


    # ========================================================
    # EXTRACT GALLERY FEATURES
    # ========================================================

    print(
        "Extracting gallery OSNet features..."
    )


    (
        gallery_features,
        gallery_pids,
        gallery_camids
    ) = extract_osnet_features(
        gallery_loader
    )


    print(
        "Running Market-1501 evaluation..."
    )


    result = evaluate_market1501(
        query_features,
        gallery_features,
        query_pids,
        gallery_pids,
        query_camids,
        gallery_camids
    )


    result[
        "dataset"
    ] = dataset_name


    result[
        "alpha"
    ] = alpha


    print()


    print(
        f"{dataset_name} RESULTS"
    )


    print(
        "-" * 52
    )


    print(
        f"Valid queries: "
        f"{result['valid_queries']}"
    )


    print(
        f"mAP: "
        f"{result['mAP']:.2f}%"
    )


    for rank in RANKS_TO_REPORT:

        print(
            f"Rank-{rank}: "
            f"{result[f'Rank-{rank}']:.2f}%"
        )


    print()


    del query_features

    del gallery_features


    if torch.cuda.is_available():

        torch.cuda.empty_cache()


    return result


# ============================================================
# 15. RUN ALL DATASETS
# ============================================================

results = []


for dataset_info in DATASETS:

    result = evaluate_dataset(
        dataset_info
    )


    results.append(
        result
    )


# ============================================================
# 16. FINAL RESULTS TABLE
# ============================================================

print()

print("=" * 96)

print(
    "FINAL PIS IMAGE-LEVEL OSNET REID RESULTS"
)

print("=" * 96)

print()


print(
    f"{'Dataset':<18}"
    f"{'mAP':>12}"
    f"{'Rank-1':>12}"
    f"{'Rank-5':>12}"
    f"{'Rank-10':>12}"
    f"{'Rank-20':>12}"
)


print(
    "-" * 78
)


for result in results:

    print(
        f"{result['dataset']:<18}"
        f"{result['mAP']:>11.2f}%"
        f"{result['Rank-1']:>11.2f}%"
        f"{result['Rank-5']:>11.2f}%"
        f"{result['Rank-10']:>11.2f}%"
        f"{result['Rank-20']:>11.2f}%"
    )


# ============================================================
# 17. SAVE RESULTS CSV
# ============================================================

with open(
    CSV_PATH,
    "w",
    newline=""
) as csv_file:

    writer = csv.writer(
        csv_file
    )


    writer.writerow(
        [
            "dataset",
            "alpha",
            "valid_queries",
            "mAP",
            "Rank-1",
            "Rank-5",
            "Rank-10",
            "Rank-20"
        ]
    )


    for result in results:

        writer.writerow(
            [
                result[
                    "dataset"
                ],

                (
                    ""
                    if result[
                        "alpha"
                    ] is None
                    else result[
                        "alpha"
                    ]
                ),

                result[
                    "valid_queries"
                ],

                result[
                    "mAP"
                ],

                result[
                    "Rank-1"
                ],

                result[
                    "Rank-5"
                ],

                result[
                    "Rank-10"
                ],

                result[
                    "Rank-20"
                ]
            ]
        )


print()

print(
    "Saved results to:"
)


print(
    CSV_PATH
)


# ============================================================
# 18. PREPARE PLOT AXIS
# ============================================================

# ------------------------------------------------------------
# x = -0.25 is used only to show the original image result
# separately from alpha=0 reconstruction.
# ------------------------------------------------------------

plot_x = [
    -0.25
]


plot_labels = [
    "Original"
]


for result in results[
    1:
]:

    plot_x.append(
        result[
            "alpha"
        ]
    )


    plot_labels.append(
        f"{result['alpha']:.2f}"
    )


# ============================================================
# 19. RANK-1 CURVE
# ============================================================

rank1_values = [
    result[
        "Rank-1"
    ]
    for result in results
]


plt.figure(
    figsize=(10, 6)
)


plt.plot(
    plot_x,
    rank1_values,
    marker="o"
)


plt.xticks(
    plot_x,
    plot_labels
)


plt.xlabel(
    "PIS Shift Strength α"
)


plt.ylabel(
    "OSNet Rank-1 (%)"
)


plt.title(
    "Image-Level ReID Utility vs PIS Shift Strength"
)


plt.grid(
    True,
    alpha=0.3
)


plt.tight_layout()


plt.savefig(
    RANK1_PLOT_PATH,
    dpi=150
)


plt.close()


# ============================================================
# 20. mAP CURVE
# ============================================================

map_values = [
    result[
        "mAP"
    ]
    for result in results
]


plt.figure(
    figsize=(10, 6)
)


plt.plot(
    plot_x,
    map_values,
    marker="o"
)


plt.xticks(
    plot_x,
    plot_labels
)


plt.xlabel(
    "PIS Shift Strength α"
)


plt.ylabel(
    "OSNet mAP (%)"
)


plt.title(
    "Image-Level ReID mAP vs PIS Shift Strength"
)


plt.grid(
    True,
    alpha=0.3
)


plt.tight_layout()


plt.savefig(
    MAP_PLOT_PATH,
    dpi=150
)


plt.close()


# ============================================================
# 21. MULTI-RANK CMC CURVE
# ============================================================

rank5_values = [
    result[
        "Rank-5"
    ]
    for result in results
]


rank10_values = [
    result[
        "Rank-10"
    ]
    for result in results
]


rank20_values = [
    result[
        "Rank-20"
    ]
    for result in results
]


plt.figure(
    figsize=(10, 6)
)


plt.plot(
    plot_x,
    rank1_values,
    marker="o",
    label="Rank-1"
)


plt.plot(
    plot_x,
    rank5_values,
    marker="o",
    label="Rank-5"
)


plt.plot(
    plot_x,
    rank10_values,
    marker="o",
    label="Rank-10"
)


plt.plot(
    plot_x,
    rank20_values,
    marker="o",
    label="Rank-20"
)


plt.xticks(
    plot_x,
    plot_labels
)


plt.xlabel(
    "PIS Shift Strength α"
)


plt.ylabel(
    "CMC Accuracy (%)"
)


plt.title(
    "ReID Utility Across PIS Shift Strengths"
)


plt.legend()


plt.grid(
    True,
    alpha=0.3
)


plt.tight_layout()


plt.savefig(
    CMC_PLOT_PATH,
    dpi=150
)


plt.close()


# ============================================================
# 22. RECONSTRUCTION COST
# ============================================================

original_result = results[
    0
]


reconstruction_result = results[
    1
]


print()

print("=" * 88)

print(
    "VAE RECONSTRUCTION COST"
)

print("=" * 88)

print()


rank1_reconstruction_change = (
    reconstruction_result[
        "Rank-1"
    ]
    -
    original_result[
        "Rank-1"
    ]
)


map_reconstruction_change = (
    reconstruction_result[
        "mAP"
    ]
    -
    original_result[
        "mAP"
    ]
)


print(
    f"Original Rank-1: "
    f"{original_result['Rank-1']:.2f}%"
)


print(
    f"Alpha 0 Rank-1: "
    f"{reconstruction_result['Rank-1']:.2f}%"
)


print(
    f"Reconstruction Rank-1 change: "
    f"{rank1_reconstruction_change:+.2f} percentage points"
)


print()


print(
    f"Original mAP: "
    f"{original_result['mAP']:.2f}%"
)


print(
    f"Alpha 0 mAP: "
    f"{reconstruction_result['mAP']:.2f}%"
)


print(
    f"Reconstruction mAP change: "
    f"{map_reconstruction_change:+.2f} percentage points"
)


# ============================================================
# 23. PIS-SPECIFIC UTILITY CHANGE
# ============================================================

print()

print("=" * 88)

print(
    "PIS-SPECIFIC UTILITY CHANGE RELATIVE TO ALPHA 0"
)

print("=" * 88)

print()


print(
    f"{'Alpha':<10}"
    f"{'Rank-1':>12}"
    f"{'Δ R1':>12}"
    f"{'mAP':>12}"
    f"{'Δ mAP':>12}"
)


print(
    "-" * 58
)


alpha_zero_rank1 = (
    reconstruction_result[
        "Rank-1"
    ]
)


alpha_zero_map = (
    reconstruction_result[
        "mAP"
    ]
)


for result in results[
    1:
]:

    rank1_delta = (
        result[
            "Rank-1"
        ]
        -
        alpha_zero_rank1
    )


    map_delta = (
        result[
            "mAP"
        ]
        -
        alpha_zero_map
    )


    print(
        f"{result['alpha']:<10.2f}"
        f"{result['Rank-1']:>11.2f}%"
        f"{rank1_delta:>+11.2f}"
        f"{result['mAP']:>11.2f}%"
        f"{map_delta:>+11.2f}"
    )


# ============================================================
# 24. FINISH
# ============================================================

print()

print("=" * 88)

print(
    "Phase 3C PIS image-level ReID evaluation completed."
)

print("=" * 88)

print()


print(
    "Saved:"
)


print(
    CSV_PATH
)


print(
    RANK1_PLOT_PATH
)


print(
    MAP_PLOT_PATH
)


print(
    CMC_PLOT_PATH
)