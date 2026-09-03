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

RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


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


PIS_ROOT = os.path.expanduser(
    "~/Desktop/ResearchResources/Phase2/"
    "disentanglement-pytorch/"
    "outputs/"
    "market1501_full_pis"
)


OSNET_CHECKPOINT = os.path.expanduser(
    "~/Desktop/ResearchResources/Phase1/"
    "deep-person-reid/"
    "log/"
    "phase1_gpu_test/"
    "model/"
    "model.pth.tar-1"
)


# Existing utility results from Phase 3C.
UTILITY_RESULTS_CSV = os.path.expanduser(
    "~/Desktop/ResearchResources/Phase2/"
    "disentanglement-pytorch/"
    "outputs/"
    "market1501_pis_osnet_reid/"
    "pis_osnet_reid_results.csv"
)


# ============================================================
# 3. PIS ALPHAS
# ============================================================

SHIFT_ALPHAS = [
    0.00,
    0.25,
    0.50,
    0.75,
    1.00
]


# ============================================================
# 4. MODEL / EVALUATION CONFIG
# ============================================================

MODEL_NAME = "osnet_x1_0"

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
# 5. OUTPUT
# ============================================================

OUTPUT_DIR = os.path.expanduser(
    "~/Desktop/ResearchResources/Phase2/"
    "disentanglement-pytorch/"
    "outputs/"
    "market1501_pis_privacy"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


PRIVACY_CSV_PATH = os.path.join(
    OUTPUT_DIR,
    "pis_privacy_results.csv"
)

SIMILARITY_CSV_PATH = os.path.join(
    OUTPUT_DIR,
    "pis_feature_similarity.csv"
)

ATTACK_RANK1_PLOT_PATH = os.path.join(
    OUTPUT_DIR,
    "cross_domain_rank1_vs_alpha.png"
)

ATTACK_MAP_PLOT_PATH = os.path.join(
    OUTPUT_DIR,
    "cross_domain_map_vs_alpha.png"
)

COSINE_PLOT_PATH = os.path.join(
    OUTPUT_DIR,
    "original_shifted_cosine_vs_alpha.png"
)

DISTANCE_PLOT_PATH = os.path.join(
    OUTPUT_DIR,
    "original_shifted_distance_vs_alpha.png"
)

TRADEOFF_PLOT_PATH = os.path.join(
    OUTPUT_DIR,
    "privacy_utility_tradeoff.png"
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
    "Phase 3D - Market-1501 PIS Privacy Evaluation"
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
    "OSNet checkpoint:"
)

print(
    OSNET_CHECKPOINT
)

print()


# ============================================================
# 7. PREPROCESSING
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
# 8. DATASET
# ============================================================

class Market1501Dataset(Dataset):

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


        if len(image_paths) == 0:

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


            if pid == -1:
                continue


            self.samples.append(
                {
                    "path":
                        image_path,

                    "filename":
                        filename,

                    "pid":
                        pid,

                    "camid":
                        camid
                }
            )


    def __len__(self):

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
            ],
            sample[
                "filename"
            ]
        )


# ============================================================
# 9. LOAD OSNET
# ============================================================

print("=" * 92)

print(
    "Loading OSNet..."
)

print("=" * 92)

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


if len(load_result.missing_keys) > 0:

    print(
        "Missing checkpoint keys:"
    )

    for key in load_result.missing_keys:
        print(
            " ",
            key
        )

    print()


if len(load_result.unexpected_keys) > 0:

    print(
        "Unexpected checkpoint keys:"
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
    "OSNet loaded and frozen."
)

print()


# ============================================================
# 10. FEATURE EXTRACTION
# ============================================================

def extract_features(
    image_dir,
    split_name
):

    dataset = Market1501Dataset(
        image_dir,
        transform=transform,
        split_name=split_name
    )


    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )


    all_features = []

    all_pids = []

    all_camids = []

    all_filenames = []


    with torch.inference_mode():

        for (
            images,
            pids,
            camids,
            filenames
        ) in loader:

            images = images.to(
                device,
                non_blocking=True
            )


            embeddings = model(
                images
            )


            embeddings = F.normalize(
                embeddings,
                p=2,
                dim=1
            )


            all_features.append(
                embeddings.cpu()
            )


            all_pids.append(
                pids.cpu()
            )


            all_camids.append(
                camids.cpu()
            )


            all_filenames.extend(
                list(
                    filenames
                )
            )


    return {
        "features":
            torch.cat(
                all_features,
                dim=0
            ).float(),

        "pids":
            torch.cat(
                all_pids,
                dim=0
            ).numpy(),

        "camids":
            torch.cat(
                all_camids,
                dim=0
            ).numpy(),

        "filenames":
            all_filenames
    }


# ============================================================
# 11. AVERAGE PRECISION
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


    precisions = []


    for rank_index in relevant_indices:

        precision = (
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


        precisions.append(
            precision
        )


    return float(
        np.mean(
            precisions
        )
    )


# ============================================================
# 12. CROSS-DOMAIN REID ATTACK
# ============================================================

def evaluate_cross_domain_reid(
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


            # Standard Market-1501 exclusion.
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


            first_match_index = np.where(
                matches == 1
            )[0][0]


            cmc[
                first_match_index:
            ] += 1


            average_precision = (
                compute_average_precision(
                    matches
                )
            )


            if average_precision is not None:

                average_precisions.append(
                    average_precision
                )


        print(
            f"  Processed "
            f"{end_index}/"
            f"{number_queries}",
            end="\r"
        )


    print()


    if valid_queries == 0:

        raise RuntimeError(
            "No valid queries found."
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
# 13. SAME-IMAGE FEATURE SIMILARITY
# ============================================================

def compute_same_image_similarity(
    original_outputs,
    shifted_outputs
):

    original_names = (
        original_outputs[
            "filenames"
        ]
    )


    shifted_names = (
        shifted_outputs[
            "filenames"
        ]
    )


    if original_names != shifted_names:

        raise RuntimeError(
            "Filename ordering mismatch between "
            "original and transformed datasets."
        )


    original_features = (
        original_outputs[
            "features"
        ]
    )


    shifted_features = (
        shifted_outputs[
            "features"
        ]
    )


    cosine_values = F.cosine_similarity(
        original_features,
        shifted_features,
        dim=1
    )


    euclidean_values = torch.norm(
        original_features
        -
        shifted_features,
        p=2,
        dim=1
    )


    return {
        "mean_cosine":
            cosine_values.mean().item(),

        "std_cosine":
            cosine_values.std(
                unbiased=False
            ).item(),

        "mean_euclidean":
            euclidean_values.mean().item(),

        "std_euclidean":
            euclidean_values.std(
                unbiased=False
            ).item(),

        "min_cosine":
            cosine_values.min().item(),

        "max_cosine":
            cosine_values.max().item()
    }


# ============================================================
# 14. EXTRACT ORIGINAL FEATURES ONCE
# ============================================================

print("=" * 92)

print(
    "Extracting ORIGINAL Market-1501 features"
)

print("=" * 92)

print()


print(
    "Original query..."
)

original_query = extract_features(
    ORIGINAL_QUERY_DIR,
    "Original Query"
)


print(
    "Original gallery..."
)

original_gallery = extract_features(
    ORIGINAL_GALLERY_DIR,
    "Original Gallery"
)


print()


# ============================================================
# 15. ORIGINAL -> ORIGINAL ATTACK BASELINE
# ============================================================

print("=" * 92)

print(
    "ORIGINAL -> ORIGINAL BASELINE"
)

print("=" * 92)

print()


original_baseline = evaluate_cross_domain_reid(
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
    f"mAP: "
    f"{original_baseline['mAP']:.2f}%"
)

print(
    f"Rank-1: "
    f"{original_baseline['Rank-1']:.2f}%"
)

print()


# ============================================================
# 16. EVALUATE EACH PIS ALPHA
# ============================================================

privacy_rows = []

similarity_rows = []


for alpha in SHIFT_ALPHAS:

    print()

    print("=" * 92)

    print(
        f"EVALUATING PRIVACY: alpha={alpha:.2f}"
    )

    print("=" * 92)

    print()


    shifted_query_dir = os.path.join(
        PIS_ROOT,
        f"alpha_{alpha:.2f}",
        "query"
    )


    shifted_gallery_dir = os.path.join(
        PIS_ROOT,
        f"alpha_{alpha:.2f}",
        "bounding_box_test"
    )


    # ========================================================
    # EXTRACT SHIFTED FEATURES
    # ========================================================

    print(
        "Extracting shifted query..."
    )


    shifted_query = extract_features(
        shifted_query_dir,
        f"Shifted Query alpha={alpha:.2f}"
    )


    print(
        "Extracting shifted gallery..."
    )


    shifted_gallery = extract_features(
        shifted_gallery_dir,
        f"Shifted Gallery alpha={alpha:.2f}"
    )


    # ========================================================
    # PRIVACY ATTACK A:
    #
    # Original query -> shifted gallery
    # ========================================================

    print()

    print(
        "Attack A: Original Query -> Shifted Gallery"
    )


    attack_original_to_shifted = (
        evaluate_cross_domain_reid(
            original_query[
                "features"
            ],

            shifted_gallery[
                "features"
            ],

            original_query[
                "pids"
            ],

            shifted_gallery[
                "pids"
            ],

            original_query[
                "camids"
            ],

            shifted_gallery[
                "camids"
            ]
        )
    )


    # ========================================================
    # PRIVACY ATTACK B:
    #
    # Shifted query -> original gallery
    # ========================================================

    print(
        "Attack B: Shifted Query -> Original Gallery"
    )


    attack_shifted_to_original = (
        evaluate_cross_domain_reid(
            shifted_query[
                "features"
            ],

            original_gallery[
                "features"
            ],

            shifted_query[
                "pids"
            ],

            original_gallery[
                "pids"
            ],

            shifted_query[
                "camids"
            ],

            original_gallery[
                "camids"
            ]
        )
    )


    # ========================================================
    # SAME-IMAGE FEATURE SIMILARITY
    # ========================================================

    query_similarity = (
        compute_same_image_similarity(
            original_query,
            shifted_query
        )
    )


    gallery_similarity = (
        compute_same_image_similarity(
            original_gallery,
            shifted_gallery
        )
    )


    combined_cosine = (
        (
            query_similarity[
                "mean_cosine"
            ]
            *
            len(
                original_query[
                    "filenames"
                ]
            )
        )
        +
        (
            gallery_similarity[
                "mean_cosine"
            ]
            *
            len(
                original_gallery[
                    "filenames"
                ]
            )
        )
    ) / (
        len(
            original_query[
                "filenames"
            ]
        )
        +
        len(
            original_gallery[
                "filenames"
            ]
        )
    )


    combined_euclidean = (
        (
            query_similarity[
                "mean_euclidean"
            ]
            *
            len(
                original_query[
                    "filenames"
                ]
            )
        )
        +
        (
            gallery_similarity[
                "mean_euclidean"
            ]
            *
            len(
                original_gallery[
                    "filenames"
                ]
            )
        )
    ) / (
        len(
            original_query[
                "filenames"
            ]
        )
        +
        len(
            original_gallery[
                "filenames"
            ]
        )
    )


    # ========================================================
    # STORE ATTACK RESULTS
    # ========================================================

    privacy_rows.append(
        {
            "alpha":
                alpha,

            "oq_sg_mAP":
                attack_original_to_shifted[
                    "mAP"
                ],

            "oq_sg_rank1":
                attack_original_to_shifted[
                    "Rank-1"
                ],

            "oq_sg_rank5":
                attack_original_to_shifted[
                    "Rank-5"
                ],

            "oq_sg_rank10":
                attack_original_to_shifted[
                    "Rank-10"
                ],

            "oq_sg_rank20":
                attack_original_to_shifted[
                    "Rank-20"
                ],

            "sq_og_mAP":
                attack_shifted_to_original[
                    "mAP"
                ],

            "sq_og_rank1":
                attack_shifted_to_original[
                    "Rank-1"
                ],

            "sq_og_rank5":
                attack_shifted_to_original[
                    "Rank-5"
                ],

            "sq_og_rank10":
                attack_shifted_to_original[
                    "Rank-10"
                ],

            "sq_og_rank20":
                attack_shifted_to_original[
                    "Rank-20"
                ]
        }
    )


    similarity_rows.append(
        {
            "alpha":
                alpha,

            "query_mean_cosine":
                query_similarity[
                    "mean_cosine"
                ],

            "query_mean_euclidean":
                query_similarity[
                    "mean_euclidean"
                ],

            "gallery_mean_cosine":
                gallery_similarity[
                    "mean_cosine"
                ],

            "gallery_mean_euclidean":
                gallery_similarity[
                    "mean_euclidean"
                ],

            "combined_mean_cosine":
                combined_cosine,

            "combined_mean_euclidean":
                combined_euclidean
        }
    )


    print()

    print(
        f"alpha={alpha:.2f}"
    )


    print(
        "Original Query -> Shifted Gallery:"
    )

    print(
        f"  mAP: "
        f"{attack_original_to_shifted['mAP']:.2f}%"
    )

    print(
        f"  Rank-1: "
        f"{attack_original_to_shifted['Rank-1']:.2f}%"
    )


    print(
        "Shifted Query -> Original Gallery:"
    )

    print(
        f"  mAP: "
        f"{attack_shifted_to_original['mAP']:.2f}%"
    )

    print(
        f"  Rank-1: "
        f"{attack_shifted_to_original['Rank-1']:.2f}%"
    )


    print(
        "Original <-> shifted OSNet similarity:"
    )

    print(
        f"  Mean cosine: "
        f"{combined_cosine:.4f}"
    )

    print(
        f"  Mean Euclidean: "
        f"{combined_euclidean:.4f}"
    )


    del shifted_query
    del shifted_gallery


    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# 17. FINAL PRIVACY TABLE
# ============================================================

print()

print("=" * 108)

print(
    "FINAL PIS ORIGINAL-IDENTITY LINKABILITY RESULTS"
)

print("=" * 108)

print()


print(
    f"{'Alpha':<8}"
    f"{'OQ->SG mAP':>14}"
    f"{'OQ->SG R1':>14}"
    f"{'SQ->OG mAP':>14}"
    f"{'SQ->OG R1':>14}"
    f"{'Cos Sim':>12}"
    f"{'Euclid':>12}"
)


print(
    "-" * 90
)


for privacy_row, similarity_row in zip(
    privacy_rows,
    similarity_rows
):

    print(
        f"{privacy_row['alpha']:<8.2f}"
        f"{privacy_row['oq_sg_mAP']:>13.2f}%"
        f"{privacy_row['oq_sg_rank1']:>13.2f}%"
        f"{privacy_row['sq_og_mAP']:>13.2f}%"
        f"{privacy_row['sq_og_rank1']:>13.2f}%"
        f"{similarity_row['combined_mean_cosine']:>12.4f}"
        f"{similarity_row['combined_mean_euclidean']:>12.4f}"
    )


# ============================================================
# 18. SAVE PRIVACY CSV
# ============================================================

with open(
    PRIVACY_CSV_PATH,
    "w",
    newline=""
) as csv_file:

    fieldnames = list(
        privacy_rows[
            0
        ].keys()
    )


    writer = csv.DictWriter(
        csv_file,
        fieldnames=fieldnames
    )


    writer.writeheader()

    writer.writerows(
        privacy_rows
    )


# ============================================================
# 19. SAVE SIMILARITY CSV
# ============================================================

with open(
    SIMILARITY_CSV_PATH,
    "w",
    newline=""
) as csv_file:

    fieldnames = list(
        similarity_rows[
            0
        ].keys()
    )


    writer = csv.DictWriter(
        csv_file,
        fieldnames=fieldnames
    )


    writer.writeheader()

    writer.writerows(
        similarity_rows
    )


# ============================================================
# 20. ATTACK RANK-1 PLOT
# ============================================================

alphas = [
    row[
        "alpha"
    ]
    for row in privacy_rows
]


oq_sg_rank1 = [
    row[
        "oq_sg_rank1"
    ]
    for row in privacy_rows
]


sq_og_rank1 = [
    row[
        "sq_og_rank1"
    ]
    for row in privacy_rows
]


plt.figure(
    figsize=(10, 6)
)


plt.plot(
    alphas,
    oq_sg_rank1,
    marker="o",
    label="Original Query -> Shifted Gallery"
)


plt.plot(
    alphas,
    sq_og_rank1,
    marker="o",
    label="Shifted Query -> Original Gallery"
)


plt.axhline(
    original_baseline[
        "Rank-1"
    ],
    linestyle="--",
    label="Original -> Original baseline"
)


plt.xlabel(
    "PIS Shift Strength α"
)


plt.ylabel(
    "Cross-Domain Rank-1 Attack Accuracy (%)"
)


plt.title(
    "Original-Identity Linkability vs PIS Shift Strength"
)


plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()


plt.savefig(
    ATTACK_RANK1_PLOT_PATH,
    dpi=150
)


plt.close()


# ============================================================
# 21. ATTACK mAP PLOT
# ============================================================

oq_sg_map = [
    row[
        "oq_sg_mAP"
    ]
    for row in privacy_rows
]


sq_og_map = [
    row[
        "sq_og_mAP"
    ]
    for row in privacy_rows
]


plt.figure(
    figsize=(10, 6)
)


plt.plot(
    alphas,
    oq_sg_map,
    marker="o",
    label="Original Query -> Shifted Gallery"
)


plt.plot(
    alphas,
    sq_og_map,
    marker="o",
    label="Shifted Query -> Original Gallery"
)


plt.axhline(
    original_baseline[
        "mAP"
    ],
    linestyle="--",
    label="Original -> Original baseline"
)


plt.xlabel(
    "PIS Shift Strength α"
)


plt.ylabel(
    "Cross-Domain mAP (%)"
)


plt.title(
    "Cross-Domain Identity Attack mAP vs PIS Strength"
)


plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()


plt.savefig(
    ATTACK_MAP_PLOT_PATH,
    dpi=150
)


plt.close()


# ============================================================
# 22. FEATURE COSINE PLOT
# ============================================================

cosine_values = [
    row[
        "combined_mean_cosine"
    ]
    for row in similarity_rows
]


plt.figure(
    figsize=(10, 6)
)


plt.plot(
    alphas,
    cosine_values,
    marker="o"
)


plt.xlabel(
    "PIS Shift Strength α"
)


plt.ylabel(
    "Mean Original-to-Shifted OSNet Cosine Similarity"
)


plt.title(
    "Original-Identity Feature Similarity vs PIS Strength"
)


plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()


plt.savefig(
    COSINE_PLOT_PATH,
    dpi=150
)


plt.close()


# ============================================================
# 23. FEATURE DISTANCE PLOT
# ============================================================

distance_values = [
    row[
        "combined_mean_euclidean"
    ]
    for row in similarity_rows
]


plt.figure(
    figsize=(10, 6)
)


plt.plot(
    alphas,
    distance_values,
    marker="o"
)


plt.xlabel(
    "PIS Shift Strength α"
)


plt.ylabel(
    "Mean Original-to-Shifted OSNet Euclidean Distance"
)


plt.title(
    "Original-to-Shifted Feature Distance vs PIS Strength"
)


plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()


plt.savefig(
    DISTANCE_PLOT_PATH,
    dpi=150
)


plt.close()


# ============================================================
# 24. OPTIONAL PRIVACY-UTILITY TRADE-OFF
# ============================================================

# ------------------------------------------------------------
# Read Phase 3C transformed-domain utility results.
#
# Y axis:
#     shifted -> shifted Rank-1
#
# X axis:
#     original -> shifted Rank-1 attack
#
# Ideal operating point:
#     high utility
#     low original-identity linkability.
# ------------------------------------------------------------

if os.path.exists(
    UTILITY_RESULTS_CSV
):

    utility_by_alpha = {}


    with open(
        UTILITY_RESULTS_CSV,
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


            utility_by_alpha[
                alpha
            ] = float(
                row[
                    "Rank-1"
                ]
            )


    tradeoff_alphas = []

    attack_values = []

    utility_values = []


    for row in privacy_rows:

        alpha = row[
            "alpha"
        ]


        if alpha not in utility_by_alpha:

            continue


        tradeoff_alphas.append(
            alpha
        )


        attack_values.append(
            row[
                "oq_sg_rank1"
            ]
        )


        utility_values.append(
            utility_by_alpha[
                alpha
            ]
        )


    plt.figure(
        figsize=(9, 6)
    )


    plt.scatter(
        attack_values,
        utility_values,
        s=90
    )


    for (
        alpha,
        attack_value,
        utility_value
    ) in zip(
        tradeoff_alphas,
        attack_values,
        utility_values
    ):

        plt.annotate(
            f"α={alpha:.2f}",
            (
                attack_value,
                utility_value
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
        "PIS Privacy-Utility Trade-off"
    )


    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()


    plt.savefig(
        TRADEOFF_PLOT_PATH,
        dpi=150
    )


    plt.close()


    print()

    print(
        "Privacy-utility trade-off plot created."
    )


else:

    print()

    print(
        "Utility results CSV not found, "
        "so combined trade-off plot was skipped."
    )


# ============================================================
# 25. FINISH
# ============================================================

print()

print("=" * 92)

print(
    "Phase 3D privacy evaluation completed."
)

print("=" * 92)

print()


print(
    "Saved:"
)

print(
    PRIVACY_CSV_PATH
)

print(
    SIMILARITY_CSV_PATH
)

print(
    ATTACK_RANK1_PLOT_PATH
)

print(
    ATTACK_MAP_PLOT_PATH
)

print(
    COSINE_PLOT_PATH
)

print(
    DISTANCE_PLOT_PATH
)


if os.path.exists(
    UTILITY_RESULTS_CSV
):

    print(
        TRADEOFF_PLOT_PATH
    )
