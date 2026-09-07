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

    torch.cuda.manual_seed_all(
        RANDOM_SEED
    )


# ============================================================
# 2. PATHS
# ============================================================

MARKET1501_ROOT = os.path.expanduser(
    "~/Desktop/ResearchResources/Phase1/deep-person-reid/"
    "reid-data/market1501/Market-1501-v15.09.15"
)


TRAIN_DIR = os.path.join(
    MARKET1501_ROOT,
    "bounding_box_train"
)


PIS_ROOT = os.path.expanduser(
    "~/Desktop/ResearchResources/Phase2/"
    "disentanglement-pytorch/"
    "outputs/"
    "market1501_full_pis"
)


DONOR_MAPPING_CSV = os.path.join(
    PIS_ROOT,
    "identity_shift_assignments.csv"
)


OSNET_CHECKPOINT = os.path.expanduser(
    "~/Desktop/ResearchResources/Phase1/"
    "deep-person-reid/"
    "log/"
    "phase1_gpu_test/"
    "model/"
    "model.pth.tar-1"
)


# ============================================================
# 3. SHIFT STRENGTHS
# ============================================================

SHIFT_ALPHAS = [
    0.00,
    0.25,
    0.50,
    0.75,
    1.00
]


# ============================================================
# 4. OSNET CONFIGURATION
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
    "market1501_pis_donor_leakage"
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


RESULTS_CSV_PATH = os.path.join(
    OUTPUT_DIR,
    "donor_leakage_results.csv"
)


SAMPLE_RESULTS_CSV_PATH = os.path.join(
    OUTPUT_DIR,
    "donor_leakage_sample_results.csv"
)


RANK1_PLOT_PATH = os.path.join(
    OUTPUT_DIR,
    "donor_rank1_vs_alpha.png"
)


MAP_PLOT_PATH = os.path.join(
    OUTPUT_DIR,
    "donor_map_vs_alpha.png"
)


SIMILARITY_PLOT_PATH = os.path.join(
    OUTPUT_DIR,
    "donor_similarity_vs_alpha.png"
)


MARGIN_PLOT_PATH = os.path.join(
    OUTPUT_DIR,
    "donor_similarity_margin_vs_alpha.png"
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
    "Phase 3E - PIS Donor-Identity Leakage Evaluation"
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

required_paths = [
    TRAIN_DIR,
    PIS_ROOT,
    DONOR_MAPPING_CSV,
    OSNET_CHECKPOINT
]


for required_path in required_paths:

    if not os.path.exists(
        required_path
    ):

        raise FileNotFoundError(
            "Required path not found:\n"
            f"{required_path}"
        )


print(
    "All required paths verified."
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
# 9. DATASET
# ============================================================

class Market1501Dataset(
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
                f"No JPG files found in:\n"
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


        unique_pids = sorted(
            set(
                sample[
                    "pid"
                ]
                for sample in self.samples
            )
        )


        print(
            f"{split_name}: "
            f"{len(self.samples)} images, "
            f"{len(unique_pids)} identities"
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
            ],
            sample[
                "filename"
            ]
        )


# ============================================================
# 10. LOAD SOURCE -> DONOR MAPPING
# ============================================================

source_to_donor = {}


with open(
    DONOR_MAPPING_CSV,
    "r"
) as csv_file:

    reader = csv.DictReader(
        csv_file
    )


    for row in reader:

        source_pid = int(
            row[
                "source_pid"
            ]
        )


        donor_pid = int(
            row[
                "donor_training_pid"
            ]
        )


        source_to_donor[
            source_pid
        ] = donor_pid


print()

print(
    f"Loaded donor assignments for "
    f"{len(source_to_donor)} source identities."
)


print()


# ============================================================
# 11. BUILD OSNET
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
    "OSNet loaded and frozen."
)


print()


# ============================================================
# 12. FEATURE EXTRACTION
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


            embeddings = model(
                images
            )


            embeddings = F.normalize(
                embeddings,
                p=2,
                dim=1
            )


            feature_batches.append(
                embeddings.cpu()
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
# 13. EXTRACT REAL DONOR TRAINING FEATURES ONCE
# ============================================================

print("=" * 92)

print(
    "Extracting real Market-1501 training features"
)

print("=" * 92)

print()


training_outputs = extract_features(
    TRAIN_DIR,
    "Real donor training gallery"
)


training_features = (
    training_outputs[
        "features"
    ]
)


training_pids = (
    training_outputs[
        "pids"
    ]
)


print()


# ============================================================
# 14. CHECK DONOR COVERAGE
# ============================================================

training_pid_set = set(
    training_pids.tolist()
)


missing_donors = sorted(
    {
        donor_pid

        for donor_pid in source_to_donor.values()

        if donor_pid not in training_pid_set
    }
)


if len(
    missing_donors
) > 0:

    raise RuntimeError(
        "Some donor identities do not exist in "
        "the training gallery:\n"
        f"{missing_donors}"
    )


print(
    "All assigned donor identities exist "
    "in the real training gallery."
)


print()


# ============================================================
# 15. AVERAGE PRECISION
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


        precisions.append(
            precision_at_rank
        )


    return float(
        np.mean(
            precisions
        )
    )


# ============================================================
# 16. DONOR RETRIEVAL ATTACK
# ============================================================

def evaluate_donor_retrieval(
    shifted_outputs,
    alpha
):

    query_features = (
        shifted_outputs[
            "features"
        ]
    )


    source_pids = (
        shifted_outputs[
            "pids"
        ]
    )


    filenames = (
        shifted_outputs[
            "filenames"
        ]
    )


    number_queries = query_features.shape[
        0
    ]


    number_gallery = training_features.shape[
        0
    ]


    cmc = np.zeros(
        number_gallery,
        dtype=np.float64
    )


    average_precisions = []


    sample_rows = []


    valid_queries = 0


    gallery_device = training_features.to(
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
        # L2-normalized features:
        #
        # cosine similarity = dot product
        #
        # squared Euclidean ranking is equivalent.
        # ----------------------------------------------------

        similarities = torch.mm(
            query_chunk,
            gallery_device.T
        )


        similarities_np = (
            similarities
            .cpu()
            .numpy()
        )


        for local_index in range(
            similarities_np.shape[
                0
            ]
        ):

            query_index = (
                start_index
                +
                local_index
            )


            source_pid = int(
                source_pids[
                    query_index
                ]
            )


            if source_pid not in source_to_donor:

                raise KeyError(
                    f"No donor assignment for "
                    f"source PID {source_pid}"
                )


            donor_pid = source_to_donor[
                source_pid
            ]


            similarity_vector = (
                similarities_np[
                    local_index
                ]
            )


            # ------------------------------------------------
            # Highest cosine similarity first.
            # ------------------------------------------------

            order = np.argsort(
                -similarity_vector
            )


            ordered_gallery_pids = (
                training_pids[
                    order
                ]
            )


            matches = (
                ordered_gallery_pids
                ==
                donor_pid
            ).astype(
                np.int32
            )


            if not np.any(
                matches
            ):

                continue


            valid_queries += 1


            first_match_index = int(
                np.where(
                    matches == 1
                )[0][0]
            )


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


            # =================================================
            # DONOR VS IMPOSTOR SIMILARITY
            # =================================================

            donor_mask = (
                training_pids
                ==
                donor_pid
            )


            non_donor_mask = np.logical_not(
                donor_mask
            )


            donor_similarities = (
                similarity_vector[
                    donor_mask
                ]
            )


            non_donor_similarities = (
                similarity_vector[
                    non_donor_mask
                ]
            )


            best_donor_similarity = float(
                np.max(
                    donor_similarities
                )
            )


            best_impostor_similarity = float(
                np.max(
                    non_donor_similarities
                )
            )


            mean_donor_similarity = float(
                np.mean(
                    donor_similarities
                )
            )


            donor_margin = (
                best_donor_similarity
                -
                best_impostor_similarity
            )


            top1_gallery_index = order[
                0
            ]


            top1_pid = int(
                training_pids[
                    top1_gallery_index
                ]
            )


            donor_is_top1 = (
                top1_pid
                ==
                donor_pid
            )


            sample_rows.append(
                {
                    "alpha":
                        alpha,

                    "filename":
                        filenames[
                            query_index
                        ],

                    "source_pid":
                        source_pid,

                    "assigned_donor_pid":
                        donor_pid,

                    "retrieved_top1_pid":
                        top1_pid,

                    "donor_is_top1":
                        int(
                            donor_is_top1
                        ),

                    "first_donor_rank":
                        first_match_index
                        +
                        1,

                    "average_precision":
                        average_precision,

                    "best_donor_cosine":
                        best_donor_similarity,

                    "mean_donor_cosine":
                        mean_donor_similarity,

                    "best_impostor_cosine":
                        best_impostor_similarity,

                    "donor_similarity_margin":
                        donor_margin
                }
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
            "No valid donor retrieval queries."
        )


    cmc /= valid_queries


    result = {
        "alpha":
            alpha,

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


    result[
        "mean_best_donor_cosine"
    ] = float(
        np.mean(
            [
                row[
                    "best_donor_cosine"
                ]
                for row in sample_rows
            ]
        )
    )


    result[
        "mean_donor_cosine"
    ] = float(
        np.mean(
            [
                row[
                    "mean_donor_cosine"
                ]
                for row in sample_rows
            ]
        )
    )


    result[
        "mean_best_impostor_cosine"
    ] = float(
        np.mean(
            [
                row[
                    "best_impostor_cosine"
                ]
                for row in sample_rows
            ]
        )
    )


    result[
        "mean_donor_similarity_margin"
    ] = float(
        np.mean(
            [
                row[
                    "donor_similarity_margin"
                ]
                for row in sample_rows
            ]
        )
    )


    return (
        result,
        sample_rows
    )


# ============================================================
# 17. EVALUATE ALL ALPHAS
# ============================================================

results = []

all_sample_rows = []


for alpha in SHIFT_ALPHAS:

    print()

    print("=" * 92)

    print(
        f"DONOR ATTACK: alpha={alpha:.2f}"
    )

    print("=" * 92)

    print()


    shifted_query_dir = os.path.join(
        PIS_ROOT,
        f"alpha_{alpha:.2f}",
        "query"
    )


    shifted_outputs = extract_features(
        shifted_query_dir,
        (
            f"PIS Query "
            f"alpha={alpha:.2f}"
        )
    )


    (
        result,
        sample_rows
    ) = evaluate_donor_retrieval(
        shifted_outputs,
        alpha
    )


    results.append(
        result
    )


    all_sample_rows.extend(
        sample_rows
    )


    print()

    print(
        f"alpha={alpha:.2f} DONOR RESULTS"
    )


    print(
        "-" * 56
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


    print(
        f"Mean best donor cosine: "
        f"{result['mean_best_donor_cosine']:.4f}"
    )


    print(
        f"Mean best impostor cosine: "
        f"{result['mean_best_impostor_cosine']:.4f}"
    )


    print(
        f"Mean donor similarity margin: "
        f"{result['mean_donor_similarity_margin']:.4f}"
    )


    del shifted_outputs


    if torch.cuda.is_available():

        torch.cuda.empty_cache()


# ============================================================
# 18. FINAL TABLE
# ============================================================

print()

print("=" * 112)

print(
    "FINAL PIS DONOR-IDENTITY LEAKAGE RESULTS"
)

print("=" * 112)

print()


print(
    f"{'Alpha':<8}"
    f"{'mAP':>10}"
    f"{'R1':>10}"
    f"{'R5':>10}"
    f"{'R10':>10}"
    f"{'R20':>10}"
    f"{'Donor Cos':>14}"
    f"{'Impostor Cos':>14}"
    f"{'Margin':>12}"
)


print(
    "-" * 100
)


for result in results:

    print(
        f"{result['alpha']:<8.2f}"
        f"{result['mAP']:>9.2f}%"
        f"{result['Rank-1']:>9.2f}%"
        f"{result['Rank-5']:>9.2f}%"
        f"{result['Rank-10']:>9.2f}%"
        f"{result['Rank-20']:>9.2f}%"
        f"{result['mean_best_donor_cosine']:>14.4f}"
        f"{result['mean_best_impostor_cosine']:>14.4f}"
        f"{result['mean_donor_similarity_margin']:>12.4f}"
    )


# ============================================================
# 19. SAVE SUMMARY CSV
# ============================================================

with open(
    RESULTS_CSV_PATH,
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
# 20. SAVE SAMPLE-LEVEL RESULTS
# ============================================================

with open(
    SAMPLE_RESULTS_CSV_PATH,
    "w",
    newline=""
) as csv_file:

    writer = csv.DictWriter(
        csv_file,
        fieldnames=list(
            all_sample_rows[
                0
            ].keys()
        )
    )


    writer.writeheader()


    writer.writerows(
        all_sample_rows
    )


# ============================================================
# 21. RANK-1 PLOT
# ============================================================

alphas = [
    result[
        "alpha"
    ]
    for result in results
]


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
    alphas,
    rank1_values,
    marker="o"
)


plt.axhline(
    100.0 / 751.0,
    linestyle="--",
    label="Approx. 1/751 identity chance"
)


plt.xlabel(
    "PIS Shift Strength α"
)


plt.ylabel(
    "Assigned-Donor Retrieval Rank-1 (%)"
)


plt.title(
    "Donor-Identity Leakage vs PIS Shift Strength"
)


plt.legend()


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
# 22. mAP PLOT
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
    alphas,
    map_values,
    marker="o"
)


plt.xlabel(
    "PIS Shift Strength α"
)


plt.ylabel(
    "Assigned-Donor Retrieval mAP (%)"
)


plt.title(
    "Donor Retrieval mAP vs PIS Shift Strength"
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
# 23. DONOR / IMPOSTOR COSINE PLOT
# ============================================================

donor_cosines = [
    result[
        "mean_best_donor_cosine"
    ]
    for result in results
]


impostor_cosines = [
    result[
        "mean_best_impostor_cosine"
    ]
    for result in results
]


plt.figure(
    figsize=(10, 6)
)


plt.plot(
    alphas,
    donor_cosines,
    marker="o",
    label="Best assigned-donor similarity"
)


plt.plot(
    alphas,
    impostor_cosines,
    marker="o",
    label="Best non-donor similarity"
)


plt.xlabel(
    "PIS Shift Strength α"
)


plt.ylabel(
    "OSNet Cosine Similarity"
)


plt.title(
    "Assigned Donor vs Strongest Impostor Similarity"
)


plt.legend()


plt.grid(
    True,
    alpha=0.3
)


plt.tight_layout()


plt.savefig(
    SIMILARITY_PLOT_PATH,
    dpi=150
)


plt.close()


# ============================================================
# 24. DONOR MARGIN PLOT
# ============================================================

margin_values = [
    result[
        "mean_donor_similarity_margin"
    ]
    for result in results
]


plt.figure(
    figsize=(10, 6)
)


plt.plot(
    alphas,
    margin_values,
    marker="o"
)


plt.axhline(
    0.0,
    linestyle="--"
)


plt.xlabel(
    "PIS Shift Strength α"
)


plt.ylabel(
    "Best Donor Cosine - Best Impostor Cosine"
)


plt.title(
    "Donor Identity Similarity Margin"
)


plt.grid(
    True,
    alpha=0.3
)


plt.tight_layout()


plt.savefig(
    MARGIN_PLOT_PATH,
    dpi=150
)


plt.close()


# ============================================================
# 25. INTERPRETATION HELPERS
# ============================================================

alpha_zero_result = results[
    0
]


alpha_one_result = results[
    -1
]


print()

print("=" * 92)

print(
    "DONOR-LEAKAGE CHANGE"
)

print("=" * 92)

print()


print(
    f"Alpha 0 donor Rank-1: "
    f"{alpha_zero_result['Rank-1']:.2f}%"
)


print(
    f"Alpha 1 donor Rank-1: "
    f"{alpha_one_result['Rank-1']:.2f}%"
)


print(
    f"Change: "
    f"{alpha_one_result['Rank-1'] - alpha_zero_result['Rank-1']:+.2f} "
    f"percentage points"
)


print()


print(
    f"Alpha 0 donor mAP: "
    f"{alpha_zero_result['mAP']:.2f}%"
)


print(
    f"Alpha 1 donor mAP: "
    f"{alpha_one_result['mAP']:.2f}%"
)


print(
    f"Change: "
    f"{alpha_one_result['mAP'] - alpha_zero_result['mAP']:+.2f} "
    f"percentage points"
)


# ============================================================
# 26. FINISH
# ============================================================

print()

print("=" * 92)

print(
    "Phase 3E donor-identity leakage evaluation completed."
)

print("=" * 92)

print()


print(
    "Saved:"
)


print(
    RESULTS_CSV_PATH
)


print(
    SAMPLE_RESULTS_CSV_PATH
)


print(
    RANK1_PLOT_PATH
)


print(
    MAP_PLOT_PATH
)


print(
    SIMILARITY_PLOT_PATH
)


print(
    MARGIN_PLOT_PATH
)