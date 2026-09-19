import os
import random
import shutil


SOURCE_FOLDER = "data/landscape"
TRAIN_FOLDER = "data/train_landscape"
VAL_FOLDER = "data/val_landscape"

VAL_RATIO = 0.15


os.makedirs(TRAIN_FOLDER, exist_ok=True)
os.makedirs(VAL_FOLDER, exist_ok=True)


images = []

for filename in os.listdir(SOURCE_FOLDER):

    if filename.lower().endswith(
        (".jpg", ".jpeg", ".png")
    ):
        images.append(filename)


random.shuffle(images)


val_count = int(
    len(images) * VAL_RATIO
)


val_images = images[:val_count]

train_images = images[val_count:]


for filename in train_images:

    shutil.copy2(
        os.path.join(
            SOURCE_FOLDER,
            filename
        ),
        os.path.join(
            TRAIN_FOLDER,
            filename
        )
    )


for filename in val_images:

    shutil.copy2(
        os.path.join(
            SOURCE_FOLDER,
            filename
        ),
        os.path.join(
            VAL_FOLDER,
            filename
        )
    )


print(
    "Training images:",
    len(train_images)
)

print(
    "Validation images:",
    len(val_images)
)