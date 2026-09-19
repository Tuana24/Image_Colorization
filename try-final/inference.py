import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from PIL import Image
from skimage.color import rgb2lab, lab2rgb
from skimage.transform import resize

from model import ColorizationNet


IMAGE_PATH = "inputs/input1.jpg"

MODEL_PATH = "checkpoints/stage4_landscape.pth"

OUTPUT_PATH = "outputs/output1.jpg"

MODEL_SIZE = 96

COLOR_STRENGTH = 1.0


device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Using device:", device)


model = ColorizationNet()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)

model.eval()

print("Model loaded:", MODEL_PATH)

image = Image.open(
    IMAGE_PATH
).convert("RGB")

original_width, original_height = image.size

print(
    "Original image size:",
    original_width,
    "x",
    original_height
)


original_rgb = np.array(
    image
).astype(np.float32) / 255.0


original_lab = rgb2lab(
    original_rgb
).astype(np.float32)


high_res_L = original_lab[:, :, 0]


small_image = image.resize(
    (MODEL_SIZE, MODEL_SIZE)
)

small_rgb = np.array(
    small_image
).astype(np.float32) / 255.0


small_lab = rgb2lab(
    small_rgb
).astype(np.float32)

small_L = small_lab[:, :, 0]


small_L_normalized = (
    small_L / 100.0
)


L_tensor = torch.from_numpy(
    small_L_normalized
)

L_tensor = (
    L_tensor
    .unsqueeze(0)
    .unsqueeze(0)
    .to(device)
)


with torch.no_grad():

    predicted_ab = model(
        L_tensor
    )
predicted_ab = (
    predicted_ab
    .squeeze(0)
    .cpu()
    .permute(1, 2, 0)
    .numpy()
)


predicted_ab = (
    predicted_ab
    * 128.0
    * COLOR_STRENGTH
)



high_res_ab = resize(
    predicted_ab,
    (
        original_height,
        original_width,
        2
    ),
    order=1,
    preserve_range=True,
    anti_aliasing=True
).astype(np.float32)



final_lab = np.zeros(
    (
        original_height,
        original_width,
        3
    ),
    dtype=np.float32
)


final_lab[:, :, 0] = high_res_L


final_lab[:, :, 1:] = high_res_ab


final_rgb = lab2rgb(
    final_lab
)

final_rgb = np.clip(
    final_rgb,
    0,
    1
)

final_uint8 = (
    final_rgb * 255
).astype(np.uint8)

final_image = Image.fromarray(
    final_uint8
)


os.makedirs(
    "outputs",
    exist_ok=True
)



final_image.save(
    OUTPUT_PATH,
    quality=95
)

print("Finished!")
print("Saved to:", OUTPUT_PATH)
print("Output resolution:", final_image.size)



input_grayscale = (
    high_res_L / 100.0
)



plt.figure(
    figsize=(14, 6)
)

plt.subplot(
    1,
    2,
    1
)

plt.imshow(
    input_grayscale,
    cmap="gray",
    vmin=0,
    vmax=1
)

plt.title(
    "Input Grayscale Image",
    fontsize=14
)

plt.axis(
    "off"
)


plt.subplot(
    1,
    2,
    2
)

plt.imshow(
    final_rgb
)

plt.title(
    "Colorized Result",
    fontsize=14
)

plt.axis(
    "off"
)


plt.tight_layout()

plt.show()