from pathlib import Path

import numpy as np
import PIL.Image as Image
import torch

IMG_SIZE = 448
IMG_EXTS = {".jpg", ".jpeg", ".png"}

DATA_ROOT = Path(__file__).resolve().parents[1] / "coco128"
IMAGE_DIR = DATA_ROOT / "images" / "train2017"
LABEL_DIR = DATA_ROOT / "labels" / "train2017"


def load_labels(label_path: Path) -> np.ndarray:
    """One label file -> (N, 5) array of [class, x_center, y_center, w, h].

    Box coords are normalised to [0, 1] by the COCO128 annotations. A missing or
    empty file means "no objects" and yields an empty (0, 5) array so the row
    still lines up with its image.
    """
    if not label_path.is_file():
        return np.zeros((0, 5), dtype=np.float32)

    rows = [line.split() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        return np.zeros((0, 5), dtype=np.float32)

    return np.array(rows, dtype=np.float32)


def process_dataset(image_dir: Path = IMAGE_DIR, label_dir: Path = LABEL_DIR):
    image_paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not image_paths:
        raise FileNotFoundError(f"no images found in {image_dir}")

    image_arr = []
    label_arr = []
    unlabelled = 0

    for image_path in image_paths:
        with Image.open(image_path) as img_file:
            img_file = img_file.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
            image_arr.append(np.asarray(img_file))

        labels = load_labels(label_dir / f"{image_path.stem}.txt")
        unlabelled += len(labels) == 0
        label_arr.append(labels)

    if unlabelled:
        print(f"{unlabelled}/{len(image_paths)} images have no labels (kept with empty boxes)")

    # scale to [0, 1] -- raw 0-255 pixels blow up the first conv's gradients
    images = torch.from_numpy(np.stack(image_arr)).float() / 255.0
    return images, label_arr


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    data, labels = process_dataset()
    print(data.shape, len(labels))

    plt.imshow(Image.fromarray((data[0].numpy() * 255).astype("uint8")))
    plt.show()
