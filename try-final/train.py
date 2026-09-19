import os

import torch
from torch.utils.data import DataLoader

from dataset import ColorizationDataset
from model import ColorizationNet
from losses import ColorizationLoss

from tqdm import tqdm

TRAIN_FOLDER = "data/train_landscape"
VAL_FOLDER = "data/val_landscape"

IMAGE_SIZE = 96

BATCH_SIZE = 8
EPOCHS = 5

LEARNING_RATE = 0.00005

CHECKPOINT_FOLDER = "checkpoints"

LOAD_MODEL_PATH = None

SAVE_MODEL_NAME = "stage4_landscape.pth"


device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Using device:", device)


train_dataset = ColorizationDataset(
    TRAIN_FOLDER,
    image_size=IMAGE_SIZE,
    augment=True
)

val_dataset = ColorizationDataset(
    VAL_FOLDER,
    image_size=IMAGE_SIZE,
    augment=False
)

print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

model = ColorizationNet()
model = model.to(device)


if (
    LOAD_MODEL_PATH is not None
    and os.path.exists(LOAD_MODEL_PATH)
):

    model.load_state_dict(
        torch.load(
            LOAD_MODEL_PATH,
            map_location=device
        )
    )

    print(
        "Loaded previous model:",
        LOAD_MODEL_PATH
    )

else:

    print(
        "Training new model from scratch."
    )

criterion = ColorizationLoss()


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=1
)


os.makedirs(
    CHECKPOINT_FOLDER,
    exist_ok=True
)


best_val_loss = float("inf")


for epoch in range(EPOCHS):

    print()
    print("=" * 50)
    print(
        f"Epoch {epoch + 1}/{EPOCHS}"
    )
    print("=" * 50)


    model.train()

    total_train_loss = 0.0

    train_progress = tqdm(
        train_loader,
        desc=f"Training {epoch + 1}/{EPOCHS}"
    )


    for L, ab in train_progress:

        L = L.to(device)
        ab = ab.to(device)

        optimizer.zero_grad()

        predicted_ab = model(L)

        loss = criterion(
            predicted_ab,
            ab
        )

        loss.backward()

        optimizer.step()

        total_train_loss += loss.item()


        train_progress.set_postfix(
            loss=f"{loss.item():.6f}"
        )


    average_train_loss = (
        total_train_loss
        /
        len(train_loader)
    )

    model.eval()

    total_val_loss = 0.0


    with torch.no_grad():

        val_progress = tqdm(
            val_loader,
            desc="Validation"
        )


        for L, ab in val_progress:

            L = L.to(device)
            ab = ab.to(device)


            predicted_ab = model(L)


            loss = criterion(
                predicted_ab,
                ab
            )


            total_val_loss += loss.item()


            val_progress.set_postfix(
                loss=f"{loss.item():.6f}"
            )


    average_val_loss = (
        total_val_loss
        /
        len(val_loader)
    )



    scheduler.step(
        average_val_loss
    )


    current_lr = (
        optimizer
        .param_groups[0]["lr"]
    )



    print()

    print(
        f"Train loss: {average_train_loss:.6f}"
    )

    print(
        f"Validation loss: {average_val_loss:.6f}"
    )

    print(
        f"Learning rate: {current_lr}"
    )


    if average_val_loss < best_val_loss:

        best_val_loss = average_val_loss


        model_path = os.path.join(
            CHECKPOINT_FOLDER,
            SAVE_MODEL_NAME
        )


        torch.save(
            model.state_dict(),
            model_path
        )


        print()
        print(
            "Best model saved!"
        )

        print(
            "Saved to:",
            model_path
        )

print()
print("=" * 50)

print(
    "Training finished!"
)

print(
    "Best validation loss:",
    best_val_loss
)

print(
    "Best model:",
    os.path.join(
        CHECKPOINT_FOLDER,
        SAVE_MODEL_NAME
    )
)

print("=" * 50)