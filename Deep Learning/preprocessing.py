
import os
import random
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf


def count_images(data_dir: str) -> int:

    total = 0
    for class_name in sorted(os.listdir(data_dir)):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        num_imgs = len([
            f for f in os.listdir(class_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])
        print(f"{class_name:10s}: {num_imgs}")
        total += num_imgs

    print("Total images:", total)
    return total


def show_random_images_per_class(train_dir: str, n_cols: int = 4, class_names=None) -> None:
   
    if class_names is None:
        class_names = sorted([
            d for d in os.listdir(train_dir)
            if os.path.isdir(os.path.join(train_dir, d)) and not d.startswith(".")
        ])

    n_classes = len(class_names)
    n_rows = int(np.ceil(n_classes / n_cols))

    plt.figure(figsize=(4 * n_cols, 4 * n_rows))

    for idx, class_name in enumerate(class_names):
        class_path = os.path.join(train_dir, class_name)
        image_files = [
            f for f in os.listdir(class_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        if not image_files:
            continue

        img_name = random.choice(image_files)
        img_path = os.path.join(class_path, img_name)
        img = Image.open(img_path).convert("RGB")

        ax = plt.subplot(n_rows, n_cols, idx + 1)
        plt.imshow(img)
        plt.title(class_name)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def create_datasets(
    train_dir: str,
    test_dir: str,
    img_size=(224, 224),
    batch_size: int = 64,
    val_split: float = 0.2,
    seed: int = 42,
):
  
    train_ds_raw = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="int",
        validation_split=val_split,
        subset="training",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
    )

    val_ds_raw = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="int",
        validation_split=val_split,
        subset="validation",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
    )

    test_ds_raw = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="int",
        shuffle=False,
        image_size=img_size,
        batch_size=batch_size,
    )

    class_names = train_ds_raw.class_names
    print("Class names from dataset:", class_names)

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = (
        train_ds_raw
        .shuffle(1000)
        .cache()
        .prefetch(AUTOTUNE)
    )

    val_ds = (
        val_ds_raw
        .cache()
        .prefetch(AUTOTUNE)
    )

    test_ds = (
        test_ds_raw
        .cache()
        .prefetch(AUTOTUNE)
    )

    return train_ds, val_ds, test_ds, class_names


def compute_class_weights(train_ds, num_classes: int):

    label_counts = Counter()

    for _, labels in train_ds.unbatch():
        label_counts[int(labels.numpy())] += 1

    print("Label counts per index:")
    for idx, count in label_counts.items():
        print(idx, ":", count)

    total = sum(label_counts.values())
    class_weight = {
        idx: total / (num_classes * count)
        for idx, count in label_counts.items()
    }

    print("\nClass weights:")
    for idx, w in class_weight.items():
        print(idx, "->", round(w, 3))

    return class_weight, label_counts


def preprocess_pil_image(pil_img: Image.Image, img_size=(224, 224)) -> np.ndarray:

    img = pil_img.convert("RGB")
    img = img.resize(img_size)
    img_array = np.array(img).astype("float32")  
    img_array = np.expand_dims(img_array, axis=0)  
    return img_array
