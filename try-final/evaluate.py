import random
import numpy as np
import torch
import matplotlib.pyplot as plt

from skimage.color import lab2rgb

from dataset import ColorizationDataset
from model import ColorizationNet


VAL_FOLDER = "data/val_landscape"
MODEL_PATH = "checkpoints/stage4_landscape.pth"

IMAGE_SIZE = 96
NUMBER_OF_IMAGES = 6


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

dataset = ColorizationDataset(
    VAL_FOLDER,
    image_size=IMAGE_SIZE,
    augment=False
)


model = ColorizationNet()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)

model.eval()


indices = random.sample(
    range(len(dataset)),
    min(NUMBER_OF_IMAGES, len(dataset))
)


fig, axes = plt.subplots(
    len(indices),
    3,
    figsize=(10, 3 * len(indices))
)


for row, index in enumerate(indices):

    L, real_ab = dataset[index]


    L_input = L.unsqueeze(0).to(device)

    with torch.no_grad():
        predicted_ab = model(L_input)

    predicted_ab = (
        predicted_ab
        .squeeze(0)
        .cpu()
    )

    L_np = (
        L.squeeze(0)
        .numpy()
    )

    real_ab_np = (
        real_ab
        .permute(1, 2, 0)
        .numpy()
    )

    predicted_ab_np = (
        predicted_ab
        .permute(1, 2, 0)
        .numpy()
    )

    L_lab = L_np * 100.0

    real_ab_lab = real_ab_np * 128.0

    predicted_ab_lab = predicted_ab_np * 128.0


    original_lab = np.zeros(
        (IMAGE_SIZE, IMAGE_SIZE, 3),
        dtype=np.float32
    )

    original_lab[:, :, 0] = L_lab
    original_lab[:, :, 1:] = real_ab_lab

    original_rgb = lab2rgb(original_lab)

    original_rgb = np.clip(
        original_rgb,
        0,
        1
    )


    predicted_lab = np.zeros(
        (IMAGE_SIZE, IMAGE_SIZE, 3),
        dtype=np.float32
    )

    predicted_lab[:, :, 0] = L_lab
    predicted_lab[:, :, 1:] = predicted_ab_lab

    predicted_rgb = lab2rgb(predicted_lab)

    predicted_rgb = np.clip(
        predicted_rgb,
        0,
        1
    )


    axes[row, 0].imshow(original_rgb)
    axes[row, 0].set_title("Original Color")
    axes[row, 0].axis("off")

    axes[row, 1].imshow(
        L_np,
        cmap="gray",
        vmin=0,
        vmax=1
    )
    axes[row, 1].set_title("Grayscale Input")
    axes[row, 1].axis("off")

    axes[row, 2].imshow(predicted_rgb)
    axes[row, 2].set_title("Model Colorization")
    axes[row, 2].axis("off")


plt.tight_layout()
plt.show()