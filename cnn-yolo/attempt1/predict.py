
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Rectangle
from PIL import Image
from torchvision.transforms import functional as TF

from yolov1 import B, C, IMG_SIZE, S, YOLOv1, decode

HERE = Path(__file__).parent
DEFAULT_WEIGHTS = HERE / "yolov1.pt"
SAMPLE_DIR = HERE / "coco128" / "images" / "train2017"
LABEL_DIR = HERE / "coco128" / "labels" / "train2017"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# COCO class ids as they appear in the YOLO-format labels (0-79, no background class).
CLASS_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]
assert len(CLASS_NAMES) == C

PALETTE = plt.get_cmap("tab20").colors


def class_color(class_id: int) -> tuple[float, float, float]:
    return PALETTE[class_id % len(PALETTE)]


# --------------------------------------------------------------------------------------
# Input handling
# --------------------------------------------------------------------------------------


def collect_images(sources: list[str], limit: int) -> list[Path]:
    """Expand file/directory arguments into a list of image paths."""
    if not sources:
        paths = sorted(SAMPLE_DIR.glob("*.jpg"))
        if not paths:
            raise FileNotFoundError(f"no sample images under {SAMPLE_DIR}")
        return paths[:limit]

    paths: list[Path] = []
    for source in sources:
        path = Path(source)
        if path.is_dir():
            paths += sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        elif path.is_file():
            paths.append(path)
        else:
            raise FileNotFoundError(source)

    if not paths:
        raise FileNotFoundError(f"no images found in {sources}")
    return paths[:limit] if limit > 0 else paths


def load_model(weights: Path, device: torch.device) -> YOLOv1:
    if not weights.exists():
        raise FileNotFoundError(f"{weights} not found — train first with `python yolov1.py`")

    model = YOLOv1()
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    return model.to(device).eval()


def preprocess(path: Path) -> tuple[Image.Image, torch.Tensor]:
    """Return the original image and the (3, 448, 448) tensor the network expects.

    The resize does not preserve aspect ratio, matching training; predicted boxes are
    normalised to [0, 1] so they map straight back onto the original image.
    """
    image = Image.open(path).convert("RGB")
    tensor = TF.to_tensor(image.resize((IMG_SIZE, IMG_SIZE)))
    return image, tensor


def read_ground_truth(image_path: Path) -> list[tuple[int, float, float, float, float]]:
    """YOLO-format labels for a COCO128 image, as (class, cx, cy, w, h) in [0, 1]."""
    label_path = LABEL_DIR / f"{image_path.stem}.txt"
    if not label_path.exists():
        return []

    boxes = []
    for line in label_path.read_text().splitlines():
        if line.strip():
            class_id, cx, cy, w, h = line.split()
            boxes.append((int(class_id), float(cx), float(cy), float(w), float(h)))
    return boxes


# --------------------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------------------


def draw(axis, image: Image.Image, detection: dict[str, torch.Tensor],
         truth: list, show_grid: bool) -> None:
    width, height = image.size
    axis.imshow(image)
    axis.set_axis_off()

    if show_grid:  # the S x S cells each responsible for one object
        for i in range(1, S):
            axis.axhline(i * height / S, color="white", lw=0.4, alpha=0.35)
            axis.axvline(i * width / S, color="white", lw=0.4, alpha=0.35)

    for _, cx, cy, w, h in truth:  # dashed white = ground truth
        axis.add_patch(Rectangle(((cx - w / 2) * width, (cy - h / 2) * height),
                                 w * width, h * height, fill=False,
                                 edgecolor="white", lw=1.4, ls="--", alpha=0.8))

    boxes, scores, labels = detection["boxes"], detection["scores"], detection["labels"]
    for (x1, y1, x2, y2), score, label in zip(boxes.tolist(), scores.tolist(), labels.tolist()):
        x1, x2 = x1 * width, x2 * width
        y1, y2 = y1 * height, y2 * height
        color = class_color(label)

        axis.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                 edgecolor=color, lw=2.0))
        axis.text(x1, max(y1 - 4, 8), f"{CLASS_NAMES[label]} {score:.2f}",
                  color="black", fontsize=8, va="bottom",
                  bbox=dict(facecolor=color, edgecolor="none", pad=1.2, alpha=0.9))


def grid_shape(count: int) -> tuple[int, int]:
    columns = min(3, count)
    return (count + columns - 1) // columns, columns


def peak_score(predictions: torch.Tensor) -> float:
    """Highest class-confidence product anywhere in the batch, ignoring the threshold.

    Useful for telling "the model saw nothing here" apart from "the threshold is too high
    for this checkpoint" — an undertrained net keeps every score near zero.
    """
    best_class_score = predictions[..., :C].clamp(0, 1).max(dim=-1).values
    confidence = predictions[..., C:].reshape(-1, S, S, B, 5)[..., 0].clamp(0, 1)
    return (confidence * best_class_score.unsqueeze(-1)).max().item()


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", nargs="*",
                        help="image files or directories (default: COCO128 samples)")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    parser.add_argument("--limit", type=int, default=6,
                        help="max images to run (0 = no limit)")
    parser.add_argument("--device", default=None, help="cpu, cuda, mps (default: auto)")
    parser.add_argument("--truth", action="store_true",
                        help="overlay COCO128 ground-truth boxes (dashed white)")
    parser.add_argument("--grid", action="store_true", help="overlay the S x S cell grid")
    parser.add_argument("--save", type=Path, default=HERE / "predictions.png",
                        help="where to write the figure")
    parser.add_argument("--no-show", action="store_true", help="save without opening a window")
    return parser.parse_args()


def pick_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    if args.no_show:
        matplotlib.use("Agg")

    device = pick_device(args.device)
    model = load_model(args.weights, device)
    paths = collect_images(args.images, args.limit)
    print(f"{len(paths)} image(s) on {device} from {args.weights.name}")

    images, tensors = zip(*(preprocess(p) for p in paths))
    with torch.no_grad():
        predictions = model(torch.stack(tensors).to(device))
    # NMS runs on CPU: torchvision's MPS kernel is missing on some builds.
    detections = decode(predictions.cpu(), conf_threshold=args.conf, iou_threshold=args.iou)

    rows, columns = grid_shape(len(paths))
    figure, axes = plt.subplots(rows, columns, figsize=(5 * columns, 5 * rows), squeeze=False)
    for axis in axes.flat:
        axis.set_axis_off()

    for axis, path, image, detection in zip(axes.flat, paths, images, detections):
        truth = read_ground_truth(path) if args.truth else []
        draw(axis, image, detection, truth, args.grid)
        axis.set_title(f"{path.name} — {len(detection['scores'])} detection(s)", fontsize=9)

        names = [f"{CLASS_NAMES[int(l)]} {s:.2f}"
                 for l, s in zip(detection["labels"], detection["scores"])]
        print(f"  {path.name}: {', '.join(names) if names else 'nothing above threshold'}")

    if not any(len(d["scores"]) for d in detections):
        best = peak_score(predictions.cpu())
        print(f"\nno detections — the best score in this batch was {best:.3f}. "
              f"Retry with --conf {max(best * 0.6, 0.01):.2f} to see what the model is "
              f"leaning towards; scores this low mean the checkpoint is undertrained.")

    figure.tight_layout()
    figure.savefig(args.save, dpi=130)
    print(f"saved {args.save}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
