import os
import glob
import csv

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

import matplotlib.pyplot as plt

# Reuse the encoder and decoder from the cloned repository
from architectures.encoders.person_conv128x64 import PersonGaussianConv128x64
from architectures.decoders.person_conv128x64 import PersonConv128x64

# Reuse the VAE mathematics from the repository
from common.ops import reparametrize, kl_divergence_mu0_var1


# ============================================================
# 1. CONFIGURATION
# ============================================================

MARKET1501_TRAIN_DIR = os.path.expanduser(
    "~/Desktop/ResearchResources/Phase1/deep-person-reid/"
    "reid-data/market1501/Market-1501-v15.09.15/bounding_box_train"
)

IMAGE_HEIGHT = 128
IMAGE_WIDTH = 64
NUM_CHANNELS = 3

# Size of the latent representation z
LATENT_DIM = 128

# RTX 2050-safe starting batch size
BATCH_SIZE = 8

# Proper Phase 2B experiment
NUM_EPOCHS = 20

LEARNING_RATE = 1e-4

NUM_WORKERS = 4

OUTPUT_DIR = "outputs/market1501_vae_128x64"

CHECKPOINT_DIR = os.path.join(
    OUTPUT_DIR,
    "checkpoints"
)

RECON_DIR = os.path.join(
    OUTPUT_DIR,
    "reconstructions"
)

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)

os.makedirs(
    RECON_DIR,
    exist_ok=True
)


# ============================================================
# 2. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("Phase 2B - Market-1501 Vanilla VAE")
print("=" * 70)

print("Device:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

print()


# ============================================================
# 3. MARKET-1501 DATASET
# ============================================================

class Market1501VAEDataset(Dataset):

    def __init__(
        self,
        image_dir,
        transform=None
    ):

        self.image_dir = image_dir
        self.transform = transform

        self.image_paths = sorted(
            glob.glob(
                os.path.join(
                    self.image_dir,
                    "*.jpg"
                )
            )
        )

        if len(self.image_paths) == 0:
            raise RuntimeError(
                f"No JPG images were found in:\n"
                f"{self.image_dir}"
            )

        print(
            f"Found {len(self.image_paths)} "
            f"Market-1501 training images"
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):

        image_path = self.image_paths[index]

        image = Image.open(
            image_path
        ).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image


# ============================================================
# 4. IMAGE PREPROCESSING
# ============================================================

# SimpleConv64 requires 64 x 64 input.
#
# ToTensor() maps pixel values:
#
# [0, 255] -> [0, 1]
#
# This matches the sigmoid decoder output and BCE loss.

transform = transforms.Compose(
    [
        transforms.Resize(
            (IMAGE_HEIGHT, IMAGE_WIDTH)
        ),
        transforms.ToTensor()
    ]
)


dataset = Market1501VAEDataset(
    MARKET1501_TRAIN_DIR,
    transform=transform
)


data_loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)


# ============================================================
# 5. FIXED VISUALISATION BATCH
# ============================================================

# We save the same people after every epoch.
# This makes it much easier to see whether reconstruction
# quality is improving over time.

fixed_images = torch.stack(
    [
        dataset[i]
        for i in range(
            min(8, len(dataset))
        )
    ]
)

fixed_images = fixed_images.to(device)


# ============================================================
# 6. BUILD VAE COMPONENTS
# ============================================================

encoder = PersonGaussianConv128x64(
    latent_dim=LATENT_DIM,
    num_channels=NUM_CHANNELS,
    image_size=None
).to(device)


decoder = PersonConv128x64(
    latent_dim=LATENT_DIM,
    num_channels=NUM_CHANNELS,
    image_size=None
).to(device)

# ============================================================
# 7. OPTIMIZER
# ============================================================

parameters = (
    list(encoder.parameters())
    +
    list(decoder.parameters())
)

optimizer = torch.optim.Adam(
    parameters,
    lr=LEARNING_RATE
)


# ============================================================
# 8. VAE FORWARD PASS
# ============================================================

def forward_vae(images):

    # --------------------------------------------------------
    # Encoder
    #
    # x -> mu, logvar
    # --------------------------------------------------------

    mu, logvar = encoder(images)

    # --------------------------------------------------------
    # Reparameterisation
    #
    # z = mu + sigma * epsilon
    #
    # epsilon ~ N(0, I)
    # --------------------------------------------------------

    z = reparametrize(
        mu,
        logvar
    )

    # --------------------------------------------------------
    # Decoder
    #
    # z -> reconstruction logits
    # --------------------------------------------------------

    reconstruction_logits = decoder(z)

    # Convert decoder output to [0, 1]

    reconstructed_images = torch.sigmoid(
        reconstruction_logits
    )

    return (
        reconstructed_images,
        mu,
        logvar,
        z
    )


# ============================================================
# 9. LOSS
# ============================================================

def vae_loss(
    reconstructed,
    original,
    mu,
    logvar
):

    # --------------------------------------------------------
    # Reconstruction Loss
    # --------------------------------------------------------

    reconstruction_loss = F.binary_cross_entropy(
        reconstructed,
        original,
        reduction="sum"
    ) / original.size(0)

    # --------------------------------------------------------
    # KL Divergence
    #
    # Encourages q(z|x) to remain close to N(0, I)
    # --------------------------------------------------------

    kl_loss = kl_divergence_mu0_var1(
        mu,
        logvar
    )

    # Vanilla VAE objective:
    #
    # L = L_reconstruction + L_KL

    total_loss = (
        reconstruction_loss
        +
        kl_loss
    )

    return (
        total_loss,
        reconstruction_loss,
        kl_loss
    )


# ============================================================
# 10. SAVE ORIGINAL VS RECONSTRUCTED IMAGES
# ============================================================

def save_reconstruction_grid(
    originals,
    reconstructions,
    epoch
):

    number_to_save = min(
        8,
        originals.size(0)
    )

    originals = originals[
        :number_to_save
    ]

    reconstructions = reconstructions[
        :number_to_save
    ]

    # Top row:
    # original Market-1501 images
    #
    # Bottom row:
    # VAE reconstructions

    comparison = torch.cat(
        [
            originals,
            reconstructions
        ],
        dim=0
    )

    save_path = os.path.join(
        RECON_DIR,
        f"epoch_{epoch:02d}.png"
    )

    save_image(
        comparison,
        save_path,
        nrow=number_to_save
    )

    print(
        f"Saved reconstruction grid to {save_path}"
    )


# ============================================================
# 11. LOSS HISTORY
# ============================================================

history_total = []
history_recon = []
history_kl = []


# ============================================================
# 12. TRAINING
# ============================================================

print()
print("Starting VAE training...")
print()

for epoch in range(
    1,
    NUM_EPOCHS + 1
):

    encoder.train()
    decoder.train()

    total_epoch_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0

    for batch_idx, images in enumerate(
        data_loader
    ):

        images = images.to(
            device,
            non_blocking=True
        )

        # ----------------------------------------------------
        # Clear previous gradients
        # ----------------------------------------------------

        optimizer.zero_grad()

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        reconstructed, mu, logvar, z = forward_vae(
            images
        )

        # ----------------------------------------------------
        # Calculate VAE loss
        # ----------------------------------------------------

        loss, recon_loss, kl_loss = vae_loss(
            reconstructed,
            images,
            mu,
            logvar
        )

        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        loss.backward()

        # ----------------------------------------------------
        # Update encoder and decoder parameters
        # ----------------------------------------------------

        optimizer.step()

        total_epoch_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()

        if batch_idx % 50 == 0:

            print(
                f"Epoch [{epoch:02d}/{NUM_EPOCHS}] "
                f"Batch [{batch_idx:03d}/{len(data_loader)}] "
                f"Total: {loss.item():.4f} "
                f"Recon: {recon_loss.item():.4f} "
                f"KL: {kl_loss.item():.4f}"
            )

    # ========================================================
    # 13. EPOCH AVERAGES
    # ========================================================

    average_total = (
        total_epoch_loss
        /
        len(data_loader)
    )

    average_recon = (
        total_recon_loss
        /
        len(data_loader)
    )

    average_kl = (
        total_kl_loss
        /
        len(data_loader)
    )

    history_total.append(
        average_total
    )

    history_recon.append(
        average_recon
    )

    history_kl.append(
        average_kl
    )

    print()
    print("=" * 70)
    print(
        f"Epoch {epoch}/{NUM_EPOCHS} finished"
    )
    print(
        f"Average Total Loss: "
        f"{average_total:.4f}"
    )
    print(
        f"Average Reconstruction Loss: "
        f"{average_recon:.4f}"
    )
    print(
        f"Average KL Loss: "
        f"{average_kl:.4f}"
    )
    print("=" * 70)


    # ========================================================
    # 14. FIXED RECONSTRUCTION EXAMPLE
    # ========================================================

    encoder.eval()
    decoder.eval()

    with torch.no_grad():

        fixed_reconstructions, _, _, _ = forward_vae(
            fixed_images
        )

    save_reconstruction_grid(
        fixed_images.detach().cpu(),
        fixed_reconstructions.detach().cpu(),
        epoch
    )


    # ========================================================
    # 15. SAVE CHECKPOINT
    # ========================================================

    checkpoint_path = os.path.join(
        CHECKPOINT_DIR,
        f"vae_epoch_{epoch:02d}.pth"
    )

    torch.save(
        {
            "epoch": epoch,
            "encoder_state_dict":
                encoder.state_dict(),
            "decoder_state_dict":
                decoder.state_dict(),
            "optimizer_state_dict":
                optimizer.state_dict(),

            "latent_dim":
                LATENT_DIM,

            "image_height":
                IMAGE_HEIGHT,

            "image_width":
                IMAGE_WIDTH,

            "average_total_loss":
                average_total,

            "average_reconstruction_loss":
                average_recon,

            "average_kl_loss":
                average_kl
        },
        checkpoint_path
    )

    print(
        f"Saved checkpoint to {checkpoint_path}"
    )

    print()


# ============================================================
# 16. SAVE LOSS HISTORY TO CSV
# ============================================================

csv_path = os.path.join(
    OUTPUT_DIR,
    "training_history.csv"
)

with open(
    csv_path,
    "w",
    newline=""
) as csv_file:

    writer = csv.writer(
        csv_file
    )

    writer.writerow(
        [
            "epoch",
            "total_loss",
            "reconstruction_loss",
            "kl_loss"
        ]
    )

    for i in range(NUM_EPOCHS):

        writer.writerow(
            [
                i + 1,
                history_total[i],
                history_recon[i],
                history_kl[i]
            ]
        )

print(
    f"Saved loss history to {csv_path}"
)


# ============================================================
# 17. PLOT LOSS CURVES
# ============================================================

epochs = range(
    1,
    NUM_EPOCHS + 1
)

plt.figure(
    figsize=(9, 6)
)

plt.plot(
    epochs,
    history_total,
    label="Total Loss"
)

plt.plot(
    epochs,
    history_recon,
    label="Reconstruction Loss"
)

plt.plot(
    epochs,
    history_kl,
    label="KL Loss"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.title(
    "Market-1501 Vanilla VAE Training Loss"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

loss_plot_path = os.path.join(
    OUTPUT_DIR,
    "training_loss.png"
)

plt.savefig(
    loss_plot_path,
    dpi=150
)

plt.close()

print(
    f"Saved loss plot to {loss_plot_path}"
)


# ============================================================
# 18. FINISH
# ============================================================

print()
print("=" * 70)
print(
    "Phase 2B VAE training experiment completed."
)
print("=" * 70)

print()
print(
    "Final Total Loss:",
    f"{history_total[-1]:.4f}"
)

print(
    "Final Reconstruction Loss:",
    f"{history_recon[-1]:.4f}"
)

print(
    "Final KL Loss:",
    f"{history_kl[-1]:.4f}"
)