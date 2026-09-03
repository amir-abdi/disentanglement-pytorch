import os
import glob
import re
import csv
import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import (
    Dataset,
    DataLoader,
    TensorDataset
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


# ============================================================
# 3. EXPERIMENT CONFIG
# ============================================================

SHIFT_ALPHAS = [
    0.00,
    0.25,
    0.50,
    0.75,
    1.00
]

MODEL_NAME = "osnet_x1_0"

NUM_CLASSES = 751

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 128

FEATURE_BATCH_SIZE = 64
NUM_WORKERS = 4

QUERY_CHUNK_SIZE = 128


# ------------------------------------------------------------
# Adaptive attacker
# ------------------------------------------------------------

ATTACK_BATCH_SIZE = 256

ATTACK_EPOCHS = 30

ATTACK_LR = 1e-3

TRIPLET_MARGIN = 0.20

TRIPLET_WEIGHT = 0.50

MSE_WEIGHT = 0.10


RANKS = [
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
    "market1501_pis_adaptive_privacy"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


RESULTS_CSV = os.path.join(
    OUTPUT_DIR,
    "adaptive_privacy_attack_results.csv"
)

SPLIT_CSV = os.path.join(
    OUTPUT_DIR,
    "adaptive_identity_split.csv"
)

RANK1_PLOT = os.path.join(
    OUTPUT_DIR,
    "adaptive_attack_rank1.png"
)

MAP_PLOT = os.path.join(
    OUTPUT_DIR,
    "adaptive_attack_map.png"
)

SIMILARITY_PLOT = os.path.join(
    OUTPUT_DIR,
    "adaptive_feature_recovery.png"
)


# ============================================================
# 5. DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("=" * 96)
print("Phase 3F - Adaptive PIS Privacy Attack")
print("=" * 96)

print("Device:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

print()


# ============================================================
# 6. TRANSFORM
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
# 7. DATASET
# ============================================================

class Market1501Dataset(Dataset):

    def __init__(
        self,
        image_dir,
        split_name
    ):

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
                f"No images found:\n{image_dir}"
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
                    "path": path,
                    "filename": filename,
                    "pid": pid,
                    "camid": camid
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
# 8. LOAD OSNET
# ============================================================

print(
    "Loading OSNet..."
)


osnet = torchreid.models.build_model(
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
    isinstance(checkpoint, dict)
    and
    "state_dict" in checkpoint
):

    state_dict = checkpoint[
        "state_dict"
    ]

elif (
    isinstance(checkpoint, dict)
    and
    "model_state_dict" in checkpoint
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


osnet.load_state_dict(
    clean_state_dict,
    strict=False
)


osnet.eval()


for parameter in osnet.parameters():

    parameter.requires_grad = False


print(
    "OSNet loaded."
)

print()


# ============================================================
# 9. FEATURE EXTRACTION
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
        batch_size=FEATURE_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )


    features = []

    pids = []

    camids = []

    filenames = []


    with torch.inference_mode():

        for (
            images,
            batch_pids,
            batch_camids,
            batch_filenames
        ) in loader:

            images = images.to(
                device,
                non_blocking=True
            )


            embeddings = osnet(
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

            pids.append(
                batch_pids.cpu()
            )

            camids.append(
                batch_camids.cpu()
            )

            filenames.extend(
                list(
                    batch_filenames
                )
            )


    output = {
        "features":
            torch.cat(
                features,
                dim=0
            ).float(),

        "pids":
            torch.cat(
                pids,
                dim=0
            ).numpy(),

        "camids":
            torch.cat(
                camids,
                dim=0
            ).numpy(),

        "filenames":
            filenames
    }


    print(
        f"{split_name}: "
        f"{len(output['filenames'])} images"
    )


    return output


# ============================================================
# 10. ORIGINAL FEATURES
# ============================================================

print("=" * 96)
print("Extracting original features")
print("=" * 96)


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
# 11. AUXILIARY / TARGET IDENTITY SPLIT
# ============================================================

query_pid_set = set(
    original_query[
        "pids"
    ].tolist()
)


gallery_pid_set = set(
    original_gallery[
        "pids"
    ].tolist()
)


valid_test_pids = sorted(
    query_pid_set.intersection(
        gallery_pid_set
    )
)


print(
    "Valid test identities:",
    len(valid_test_pids)
)


split_rng = random.Random(
    SEED
)


shuffled_pids = (
    valid_test_pids.copy()
)


split_rng.shuffle(
    shuffled_pids
)


split_index = (
    len(shuffled_pids)
    //
    2
)


auxiliary_pids = set(
    shuffled_pids[
        :split_index
    ]
)


target_pids = set(
    shuffled_pids[
        split_index:
    ]
)


print(
    "Adaptive attacker auxiliary IDs:",
    len(auxiliary_pids)
)


print(
    "Held-out target IDs:",
    len(target_pids)
)


print()


# ============================================================
# 12. SAVE SPLIT
# ============================================================

with open(
    SPLIT_CSV,
    "w",
    newline=""
) as csv_file:

    writer = csv.writer(
        csv_file
    )

    writer.writerow(
        [
            "pid",
            "role"
        ]
    )

    for pid in sorted(
        auxiliary_pids
    ):

        writer.writerow(
            [
                pid,
                "auxiliary_attack_train"
            ]
        )

    for pid in sorted(
        target_pids
    ):

        writer.writerow(
            [
                pid,
                "held_out_attack_test"
            ]
        )


# ============================================================
# 13. HELPERS
# ============================================================

def make_mask(
    pids,
    pid_set
):

    return np.array(
        [
            pid in pid_set
            for pid in pids
        ],
        dtype=bool
    )


def subset_outputs(
    outputs,
    pid_set
):

    mask = make_mask(
        outputs[
            "pids"
        ],
        pid_set
    )


    indices = np.where(
        mask
    )[0]


    return {
        "features":
            outputs[
                "features"
            ][
                indices
            ],

        "pids":
            outputs[
                "pids"
            ][
                indices
            ],

        "camids":
            outputs[
                "camids"
            ][
                indices
            ],

        "filenames":
            [
                outputs[
                    "filenames"
                ][
                    index
                ]
                for index in indices
            ]
    }


# ============================================================
# 14. TARGET ORIGINAL SET
# ============================================================

target_original_query = subset_outputs(
    original_query,
    target_pids
)


target_original_gallery = subset_outputs(
    original_gallery,
    target_pids
)


print(
    "Held-out target query images:",
    len(
        target_original_query[
            "filenames"
        ]
    )
)


print(
    "Held-out target gallery images:",
    len(
        target_original_gallery[
            "filenames"
        ]
    )
)


print()


# ============================================================
# 15. MARKET-1501 METRICS
# ============================================================

def average_precision(
    matches
):

    relevant = np.where(
        matches == 1
    )[0]


    if len(relevant) == 0:

        return None


    cumulative = np.cumsum(
        matches
    )


    precisions = []


    for index in relevant:

        precisions.append(
            cumulative[index]
            /
            (
                index + 1
            )
        )


    return float(
        np.mean(
            precisions
        )
    )


def evaluate_reid(
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


    aps = []

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


        distance = (
            2.0
            -
            2.0
            *
            similarity
        ).cpu().numpy()


        for local_index in range(
            distance.shape[
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
                distance[
                    local_index
                ]
            )


            ordered_pids = gallery_pids[
                order
            ]


            ordered_camids = gallery_camids[
                order
            ]


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


            keep = ~remove


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


            ap = average_precision(
                matches
            )


            if ap is not None:

                aps.append(
                    ap
                )


    cmc /= valid_queries


    result = {
        "valid_queries":
            valid_queries,

        "mAP":
            float(
                np.mean(
                    aps
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
# 16. ORIGINAL -> ORIGINAL TARGET BASELINE
# ============================================================

target_original_baseline = evaluate_reid(
    target_original_query[
        "features"
    ],

    target_original_gallery[
        "features"
    ],

    target_original_query[
        "pids"
    ],

    target_original_gallery[
        "pids"
    ],

    target_original_query[
        "camids"
    ],

    target_original_gallery[
        "camids"
    ]
)


print("=" * 96)
print("HELD-OUT TARGET ORIGINAL BASELINE")
print("=" * 96)

print(
    f"Rank-1: "
    f"{target_original_baseline['Rank-1']:.2f}%"
)

print(
    f"mAP: "
    f"{target_original_baseline['mAP']:.2f}%"
)

print()


# ============================================================
# 17. ADAPTIVE ATTACK MAPPER
# ============================================================

class AdaptiveFeatureMapper(
    nn.Module
):

    def __init__(
        self,
        feature_dim
    ):

        super().__init__()


        self.network = nn.Sequential(
            nn.Linear(
                feature_dim,
                1024
            ),

            nn.ReLU(),

            nn.Linear(
                1024,
                feature_dim
            )
        )


    def forward(
        self,
        x
    ):

        residual = self.network(
            x
        )


        output = (
            x
            +
            residual
        )


        output = F.normalize(
            output,
            p=2,
            dim=1
        )


        return output


# ============================================================
# 18. ATTACK TRAINING LOSS
# ============================================================

def attack_loss(
    mapped_shifted,
    original_features,
    pids
):

    # --------------------------------------------------------
    # 1. Paired cosine reconstruction
    # --------------------------------------------------------

    paired_cosine = F.cosine_similarity(
        mapped_shifted,
        original_features,
        dim=1
    )


    cosine_loss = (
        1.0
        -
        paired_cosine.mean()
    )


    # --------------------------------------------------------
    # 2. Direct feature reconstruction
    # --------------------------------------------------------

    mse_loss = F.mse_loss(
        mapped_shifted,
        original_features
    )


    # --------------------------------------------------------
    # 3. Hard negative cross-domain metric loss
    #
    # Same-PID samples are masked out as negatives.
    # --------------------------------------------------------

    similarity_matrix = torch.mm(
        mapped_shifted,
        original_features.T
    )


    same_pid = (
        pids.unsqueeze(1)
        ==
        pids.unsqueeze(0)
    )


    negative_similarity = (
        similarity_matrix
        .masked_fill(
            same_pid,
            -1e9
        )
        .max(
            dim=1
        )
        .values
    )


    positive_distance = (
        1.0
        -
        paired_cosine
    )


    negative_distance = (
        1.0
        -
        negative_similarity
    )


    triplet_loss = F.relu(
        positive_distance
        -
        negative_distance
        +
        TRIPLET_MARGIN
    ).mean()


    total_loss = (
        cosine_loss

        +
        MSE_WEIGHT
        *
        mse_loss

        +
        TRIPLET_WEIGHT
        *
        triplet_loss
    )


    return (
        total_loss,
        cosine_loss,
        triplet_loss
    )


# ============================================================
# 19. TRAIN MAPPER
# ============================================================

def train_mapper(
    shifted_features,
    original_features,
    pids,
    alpha
):

    feature_dim = shifted_features.shape[
        1
    ]


    mapper = AdaptiveFeatureMapper(
        feature_dim
    ).to(
        device
    )


    optimizer = torch.optim.Adam(
        mapper.parameters(),
        lr=ATTACK_LR
    )


    training_dataset = TensorDataset(
        shifted_features,
        original_features,
        torch.tensor(
            pids,
            dtype=torch.long
        )
    )


    loader = DataLoader(
        training_dataset,
        batch_size=ATTACK_BATCH_SIZE,
        shuffle=True
    )


    print(
        f"Training adaptive mapper "
        f"for alpha={alpha:.2f}"
    )


    for epoch in range(
        1,
        ATTACK_EPOCHS + 1
    ):

        mapper.train()


        total_loss_sum = 0.0

        cosine_loss_sum = 0.0

        triplet_loss_sum = 0.0

        number_batches = 0


        for (
            shifted_batch,
            original_batch,
            pid_batch
        ) in loader:

            shifted_batch = shifted_batch.to(
                device
            )


            original_batch = original_batch.to(
                device
            )


            pid_batch = pid_batch.to(
                device
            )


            optimizer.zero_grad()


            mapped = mapper(
                shifted_batch
            )


            (
                loss,
                cosine_loss,
                triplet_loss
            ) = attack_loss(
                mapped,
                original_batch,
                pid_batch
            )


            loss.backward()


            optimizer.step()


            total_loss_sum += loss.item()

            cosine_loss_sum += (
                cosine_loss.item()
            )

            triplet_loss_sum += (
                triplet_loss.item()
            )

            number_batches += 1


        if (
            epoch == 1
            or
            epoch % 5 == 0
            or
            epoch == ATTACK_EPOCHS
        ):

            print(
                f"  Epoch {epoch:02d}/{ATTACK_EPOCHS}"
                f" | total={total_loss_sum / number_batches:.6f}"
                f" | cosine={cosine_loss_sum / number_batches:.6f}"
                f" | triplet={triplet_loss_sum / number_batches:.6f}"
            )


    mapper.eval()


    return mapper


# ============================================================
# 20. APPLY MAPPER
# ============================================================

def apply_mapper(
    mapper,
    features,
    batch_size=512
):

    mapped_batches = []


    with torch.inference_mode():

        for start in range(
            0,
            len(features),
            batch_size
        ):

            end = min(
                start
                +
                batch_size,
                len(features)
            )


            feature_batch = features[
                start:end
            ].to(
                device
            )


            mapped = mapper(
                feature_batch
            )


            mapped_batches.append(
                mapped.cpu()
            )


    return torch.cat(
        mapped_batches,
        dim=0
    )


# ============================================================
# 21. FEATURE RECOVERY SIMILARITY
# ============================================================

def mean_cosine_similarity(
    features_a,
    features_b
):

    return float(
        F.cosine_similarity(
            features_a,
            features_b,
            dim=1
        )
        .mean()
        .item()
    )


# ============================================================
# 22. RUN EACH ALPHA
# ============================================================

results = []


for alpha in SHIFT_ALPHAS:

    print()

    print("=" * 96)

    print(
        f"ADAPTIVE ATTACK: alpha={alpha:.2f}"
    )

    print("=" * 96)

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


    shifted_query = extract_features(
        shifted_query_dir,
        f"Shifted Query alpha={alpha:.2f}"
    )


    shifted_gallery = extract_features(
        shifted_gallery_dir,
        f"Shifted Gallery alpha={alpha:.2f}"
    )


    # ========================================================
    # VERIFY EXACT PAIRING
    # ========================================================

    if (
        shifted_query[
            "filenames"
        ]
        !=
        original_query[
            "filenames"
        ]
    ):

        raise RuntimeError(
            "Original/shifted query filenames do not match."
        )


    if (
        shifted_gallery[
            "filenames"
        ]
        !=
        original_gallery[
            "filenames"
        ]
    ):

        raise RuntimeError(
            "Original/shifted gallery filenames do not match."
        )


    # ========================================================
    # AUXILIARY ATTACK TRAINING DATA
    # ========================================================

    aux_query_mask = make_mask(
        original_query[
            "pids"
        ],
        auxiliary_pids
    )


    aux_gallery_mask = make_mask(
        original_gallery[
            "pids"
        ],
        auxiliary_pids
    )


    aux_shifted_features = torch.cat(
        [
            shifted_query[
                "features"
            ][
                aux_query_mask
            ],

            shifted_gallery[
                "features"
            ][
                aux_gallery_mask
            ]
        ],
        dim=0
    )


    aux_original_features = torch.cat(
        [
            original_query[
                "features"
            ][
                aux_query_mask
            ],

            original_gallery[
                "features"
            ][
                aux_gallery_mask
            ]
        ],
        dim=0
    )


    aux_pids = np.concatenate(
        [
            original_query[
                "pids"
            ][
                aux_query_mask
            ],

            original_gallery[
                "pids"
            ][
                aux_gallery_mask
            ]
        ]
    )


    print(
        "Auxiliary paired training samples:",
        len(
            aux_pids
        )
    )


    # ========================================================
    # TARGET SHIFTED DATA
    # ========================================================

    target_shifted_query = subset_outputs(
        shifted_query,
        target_pids
    )


    target_shifted_gallery = subset_outputs(
        shifted_gallery,
        target_pids
    )


    # ========================================================
    # RAW CROSS-DOMAIN ATTACK
    # ========================================================

    raw_oq_sg = evaluate_reid(
        target_original_query[
            "features"
        ],

        target_shifted_gallery[
            "features"
        ],

        target_original_query[
            "pids"
        ],

        target_shifted_gallery[
            "pids"
        ],

        target_original_query[
            "camids"
        ],

        target_shifted_gallery[
            "camids"
        ]
    )


    raw_sq_og = evaluate_reid(
        target_shifted_query[
            "features"
        ],

        target_original_gallery[
            "features"
        ],

        target_shifted_query[
            "pids"
        ],

        target_original_gallery[
            "pids"
        ],

        target_shifted_query[
            "camids"
        ],

        target_original_gallery[
            "camids"
        ]
    )


    raw_cosine = (
        (
            mean_cosine_similarity(
                target_shifted_query[
                    "features"
                ],
                target_original_query[
                    "features"
                ]
            )
            *
            len(
                target_shifted_query[
                    "features"
                ]
            )
        )
        +
        (
            mean_cosine_similarity(
                target_shifted_gallery[
                    "features"
                ],
                target_original_gallery[
                    "features"
                ]
            )
            *
            len(
                target_shifted_gallery[
                    "features"
                ]
            )
        )
    ) / (
        len(
            target_shifted_query[
                "features"
            ]
        )
        +
        len(
            target_shifted_gallery[
                "features"
            ]
        )
    )


    # ========================================================
    # TRAIN ADAPTIVE MAPPER
    # ========================================================

    mapper = train_mapper(
        aux_shifted_features,
        aux_original_features,
        aux_pids,
        alpha
    )


    # ========================================================
    # MAP HELD-OUT SHIFTED FEATURES
    # ========================================================

    mapped_shifted_query = apply_mapper(
        mapper,
        target_shifted_query[
            "features"
        ]
    )


    mapped_shifted_gallery = apply_mapper(
        mapper,
        target_shifted_gallery[
            "features"
        ]
    )


    # ========================================================
    # ADAPTIVE ATTACK:
    #
    # original query -> mapped shifted gallery
    # ========================================================

    adapted_oq_sg = evaluate_reid(
        target_original_query[
            "features"
        ],

        mapped_shifted_gallery,

        target_original_query[
            "pids"
        ],

        target_shifted_gallery[
            "pids"
        ],

        target_original_query[
            "camids"
        ],

        target_shifted_gallery[
            "camids"
        ]
    )


    # ========================================================
    # ADAPTIVE ATTACK:
    #
    # mapped shifted query -> original gallery
    # ========================================================

    adapted_sq_og = evaluate_reid(
        mapped_shifted_query,

        target_original_gallery[
            "features"
        ],

        target_shifted_query[
            "pids"
        ],

        target_original_gallery[
            "pids"
        ],

        target_shifted_query[
            "camids"
        ],

        target_original_gallery[
            "camids"
        ]
    )


    # ========================================================
    # FEATURE RECOVERY
    # ========================================================

    mapped_cosine = (
        (
            mean_cosine_similarity(
                mapped_shifted_query,
                target_original_query[
                    "features"
                ]
            )
            *
            len(
                mapped_shifted_query
            )
        )
        +
        (
            mean_cosine_similarity(
                mapped_shifted_gallery,
                target_original_gallery[
                    "features"
                ]
            )
            *
            len(
                mapped_shifted_gallery
            )
        )
    ) / (
        len(
            mapped_shifted_query
        )
        +
        len(
            mapped_shifted_gallery
        )
    )


    result = {
        "alpha":
            alpha,

        "original_target_rank1":
            target_original_baseline[
                "Rank-1"
            ],

        "original_target_mAP":
            target_original_baseline[
                "mAP"
            ],

        "raw_oq_sg_rank1":
            raw_oq_sg[
                "Rank-1"
            ],

        "raw_oq_sg_mAP":
            raw_oq_sg[
                "mAP"
            ],

        "adapted_oq_sg_rank1":
            adapted_oq_sg[
                "Rank-1"
            ],

        "adapted_oq_sg_mAP":
            adapted_oq_sg[
                "mAP"
            ],

        "raw_sq_og_rank1":
            raw_sq_og[
                "Rank-1"
            ],

        "raw_sq_og_mAP":
            raw_sq_og[
                "mAP"
            ],

        "adapted_sq_og_rank1":
            adapted_sq_og[
                "Rank-1"
            ],

        "adapted_sq_og_mAP":
            adapted_sq_og[
                "mAP"
            ],

        "raw_same_image_cosine":
            raw_cosine,

        "adapted_same_image_cosine":
            mapped_cosine
    }


    results.append(
        result
    )


    print()

    print(
        f"alpha={alpha:.2f} RESULTS"
    )

    print(
        "-" * 72
    )


    print(
        "Original Query -> Shifted Gallery"
    )

    print(
        f"  Before adaptation R1: "
        f"{raw_oq_sg['Rank-1']:.2f}%"
    )

    print(
        f"  After adaptation R1:  "
        f"{adapted_oq_sg['Rank-1']:.2f}%"
    )

    print(
        f"  Before adaptation mAP: "
        f"{raw_oq_sg['mAP']:.2f}%"
    )

    print(
        f"  After adaptation mAP:  "
        f"{adapted_oq_sg['mAP']:.2f}%"
    )


    print()


    print(
        "Shifted Query -> Original Gallery"
    )

    print(
        f"  Before adaptation R1: "
        f"{raw_sq_og['Rank-1']:.2f}%"
    )

    print(
        f"  After adaptation R1:  "
        f"{adapted_sq_og['Rank-1']:.2f}%"
    )


    print()


    print(
        f"Same-image cosine "
        f"before mapper: "
        f"{raw_cosine:.4f}"
    )

    print(
        f"Same-image cosine "
        f"after mapper:  "
        f"{mapped_cosine:.4f}"
    )


    del mapper

    del shifted_query
    del shifted_gallery

    del mapped_shifted_query
    del mapped_shifted_gallery


    if torch.cuda.is_available():

        torch.cuda.empty_cache()


# ============================================================
# 23. FINAL TABLE
# ============================================================

print()

print("=" * 116)

print(
    "FINAL ADAPTIVE ORIGINAL-IDENTITY ATTACK RESULTS"
)

print("=" * 116)

print()


print(
    f"{'Alpha':<8}"
    f"{'Raw OQ->SG R1':>16}"
    f"{'Adapt OQ->SG R1':>18}"
    f"{'Raw OQ->SG mAP':>18}"
    f"{'Adapt OQ->SG mAP':>20}"
    f"{'Raw Cos':>12}"
    f"{'Adapt Cos':>12}"
)


print(
    "-" * 108
)


for result in results:

    print(
        f"{result['alpha']:<8.2f}"
        f"{result['raw_oq_sg_rank1']:>15.2f}%"
        f"{result['adapted_oq_sg_rank1']:>17.2f}%"
        f"{result['raw_oq_sg_mAP']:>17.2f}%"
        f"{result['adapted_oq_sg_mAP']:>19.2f}%"
        f"{result['raw_same_image_cosine']:>12.4f}"
        f"{result['adapted_same_image_cosine']:>12.4f}"
    )


# ============================================================
# 24. SAVE RESULTS
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
# 25. PLOTS
# ============================================================

alphas = [
    result[
        "alpha"
    ]
    for result in results
]


raw_rank1 = [
    result[
        "raw_oq_sg_rank1"
    ]
    for result in results
]


adapted_rank1 = [
    result[
        "adapted_oq_sg_rank1"
    ]
    for result in results
]


plt.figure(
    figsize=(10, 6)
)


plt.plot(
    alphas,
    raw_rank1,
    marker="o",
    label="Fixed OSNet attack"
)


plt.plot(
    alphas,
    adapted_rank1,
    marker="o",
    label="Adaptive mapped attack"
)


plt.axhline(
    target_original_baseline[
        "Rank-1"
    ],
    linestyle="--",
    label="Original-domain baseline"
)


plt.xlabel(
    "PIS Shift Strength α"
)


plt.ylabel(
    "Original-Identity Attack Rank-1 (%)"
)


plt.title(
    "Fixed vs Adaptive Source-Identity Attack"
)


plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    RANK1_PLOT,
    dpi=150
)

plt.close()


# ============================================================

raw_map = [
    result[
        "raw_oq_sg_mAP"
    ]
    for result in results
]


adapted_map = [
    result[
        "adapted_oq_sg_mAP"
    ]
    for result in results
]


plt.figure(
    figsize=(10, 6)
)


plt.plot(
    alphas,
    raw_map,
    marker="o",
    label="Fixed OSNet attack"
)


plt.plot(
    alphas,
    adapted_map,
    marker="o",
    label="Adaptive mapped attack"
)


plt.axhline(
    target_original_baseline[
        "mAP"
    ],
    linestyle="--",
    label="Original-domain baseline"
)


plt.xlabel(
    "PIS Shift Strength α"
)


plt.ylabel(
    "Original-Identity Attack mAP (%)"
)


plt.title(
    "Adaptive Identity-Linkage Attack mAP"
)


plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    MAP_PLOT,
    dpi=150
)

plt.close()


# ============================================================

raw_cosines = [
    result[
        "raw_same_image_cosine"
    ]
    for result in results
]


mapped_cosines = [
    result[
        "adapted_same_image_cosine"
    ]
    for result in results
]


plt.figure(
    figsize=(10, 6)
)


plt.plot(
    alphas,
    raw_cosines,
    marker="o",
    label="Before adaptive mapper"
)


plt.plot(
    alphas,
    mapped_cosines,
    marker="o",
    label="After adaptive mapper"
)


plt.xlabel(
    "PIS Shift Strength α"
)


plt.ylabel(
    "Original-to-Shifted OSNet Cosine Similarity"
)


plt.title(
    "Adaptive Recovery of Original Feature Space"
)


plt.legend()

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
# 26. FINISH
# ============================================================

print()

print("=" * 96)

print(
    "Phase 3F adaptive privacy attack completed."
)

print("=" * 96)

print()

print(
    "Saved:"
)

print(
    RESULTS_CSV
)

print(
    SPLIT_CSV
)

print(
    RANK1_PLOT
)

print(
    MAP_PLOT
)

print(
    SIMILARITY_PLOT
)