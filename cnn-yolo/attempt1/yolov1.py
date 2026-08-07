"""Minimal YOLOv1 (Redmon et al., 2016) in PyTorch, trained on the COCO128 sample set.

The image is split into an S x S grid. Each cell predicts B boxes (confidence + x, y, w, h)
and one shared set of C class probabilities, so the network output is S*S*(C + B*5) numbers.

Run:  python yolov1.py
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import nms
from torchvision.transforms import functional as TF

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

S = 7  # grid size
B = 2  # boxes predicted per cell
C = 80 # COCO classes
IMG_SIZE = 448
CELL_VECTOR = C + B * 5  # per-cell prediction length: [classes..., (conf, x, y, w, h) * B]

DATA_ROOT = Path(__file__).resolve().parents[1] / "coco128"  # cnn-yolo/coco128, next to attempt1/


def pick_device(requested: str | None = None) -> torch.device:
    """cuda > mps (Apple silicon) > cpu, unless a device is named explicitly."""
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class TrainConfig:
    # 128 images / batch 8 = 16 steps per epoch, so "50 epochs" is only 800 updates --
    # nowhere near enough to train 284M parameters from scratch. See the note in train().
    epochs: int = 150
    batch_size: int = 8
    lr: float = 1e-4
    weight_decay: float = 5e-4
    grad_clip: float = 10.0
    num_workers: int = 2
    device: torch.device = field(default_factory=pick_device)


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------


class Coco128Dataset(Dataset):
    """Images plus YOLO-format labels (`class cx cy w h`, all normalised to [0, 1]).

    Returns (image, target) where target is (S, S, C + 5): the ground truth only ever holds
    one box per cell, since every box in a cell shares the same class prediction anyway.
    """

    def __init__(self, root: Path, split: str = "train2017"):
        self.image_paths = sorted((root / "images" / split).glob("*.jpg"))
        self.label_dir = root / "labels" / split
        if not self.image_paths:
            raise FileNotFoundError(f"no images under {root / 'images' / split}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path = self.image_paths[index]

        # Labels are normalised, so a plain (non-aspect-preserving) resize leaves them valid.
        image = Image.open(image_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        image = TF.to_tensor(image)

        return image, self._build_target(self._read_boxes(image_path))

    def _read_boxes(self, image_path: Path) -> list[tuple[int, float, float, float, float]]:
        label_path = self.label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():  # background image
            return []

        boxes = []
        for line in label_path.read_text().splitlines():
            if not line.strip():
                continue
            class_id, cx, cy, w, h = line.split()
            boxes.append((int(class_id), float(cx), float(cy), float(w), float(h)))
        return boxes

    @staticmethod
    def _build_target(boxes) -> torch.Tensor:
        target = torch.zeros(S, S, C + 5)

        for class_id, cx, cy, w, h in boxes:
            # min() guards a centre sitting exactly on the right/bottom edge (cx == 1.0),
            # which would otherwise index cell S and raise.
            col, row = min(int(cx * S), S - 1), min(int(cy * S), S - 1)  # cell that owns this box
            if target[row, col, C] == 1:
                continue  # cell already taken: YOLOv1 simply cannot predict both

            target[row, col, class_id] = 1.0                   # one-hot class
            target[row, col, C] = 1.0                          # objectness
            # x, y become offsets within the cell; w, h stay relative to the whole image.
            target[row, col, C + 1:] = torch.tensor([cx * S - col, cy * S - row, w, h])

        return target


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------

# Darknet-24 backbone. Tuples are (kernel, out_channels, stride, padding), "M" is a
# 2x2 max pool, and a list is [conv, conv, repeats].
ARCHITECTURE = [
    (7, 64, 2, 3), "M",
    (3, 192, 1, 1), "M",
    (1, 128, 1, 0), (3, 256, 1, 1), (1, 256, 1, 0), (3, 512, 1, 1), "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4], (1, 512, 1, 0), (3, 1024, 1, 1), "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2], (3, 1024, 1, 1), (3, 1024, 2, 1),
    (3, 1024, 1, 1), (3, 1024, 1, 1),
]


class ConvBlock(nn.Module):
    """Conv -> BatchNorm -> LeakyReLU. BatchNorm is a YOLOv2 addition, but without it
    training this depth from scratch on a small set diverges almost immediately."""

    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(self.norm(self.conv(x)), 0.1)


class YOLOv1(nn.Module):
    """The paper's head is Linear(50176, 4096), i.e. 224M of the model's 284M weights --
    80% of the network is a fully connected layer being fitted to 128 images. Shrinking it
    to 1024 and dropping the 0.5 dropout costs nothing here and makes the thing actually
    converge; restore hidden=4096, dropout=0.5 if you ever train on full COCO."""

    def __init__(self, hidden: int = 1024, dropout: float = 0.0):
        super().__init__()
        self.backbone = self._build_backbone()
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, hidden),
            nn.Dropout(dropout),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden, S * S * CELL_VECTOR),
        )

    @staticmethod
    def _build_backbone() -> nn.Sequential:
        layers, in_channels = [], 3

        for spec in ARCHITECTURE:
            if spec == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            elif isinstance(spec, tuple):
                kernel, out_channels, stride, padding = spec
                layers.append(ConvBlock(in_channels, out_channels, kernel_size=kernel,
                                        stride=stride, padding=padding))
                in_channels = out_channels
            else:  # repeated conv pair
                first, second, repeats = spec
                for _ in range(repeats):
                    for kernel, out_channels, stride, padding in (first, second):
                        layers.append(ConvBlock(in_channels, out_channels, kernel_size=kernel,
                                                stride=stride, padding=padding))
                        in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(N, 3, 448, 448) -> (N, S, S, C + B * 5), raw (unactivated) predictions."""
        return self.head(self.backbone(x)).reshape(-1, S, S, CELL_VECTOR)


# --------------------------------------------------------------------------------------
# Loss
# --------------------------------------------------------------------------------------


def to_corners(boxes: torch.Tensor) -> torch.Tensor:
    """(..., 4) midpoint boxes (x, y, w, h) -> corner boxes (x1, y1, x2, y2).

    w/h are clamped at 0 first: the head is unactivated, so an untrained net happily emits
    a negative width, which would give x2 < x1 -- a box torchvision's NMS treats as having
    negative area and IoU never suppresses. Clamping collapses it to an empty box instead.
    """
    xy, wh = boxes[..., :2], boxes[..., 2:4].clamp(min=0)
    return torch.cat([xy - wh / 2, xy + wh / 2], dim=-1)


def iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Element-wise IoU between two broadcastable sets of corner boxes."""
    x1 = torch.max(a[..., 0], b[..., 0])
    y1 = torch.max(a[..., 1], b[..., 1])
    x2 = torch.min(a[..., 2], b[..., 2])
    y2 = torch.min(a[..., 3], b[..., 3])

    overlap = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return overlap / (area_a + area_b - overlap + 1e-6)


def to_image_frame(boxes: torch.Tensor) -> torch.Tensor:
    """(N, S, S, ..., 4) boxes with cell-relative x, y -> x, y relative to the whole image.

    Needed before computing IoU: mixing cell-relative centres with image-relative widths
    would distort the overlap.
    """
    grid = torch.arange(S, device=boxes.device, dtype=boxes.dtype)
    trailing = [1] * (boxes.dim() - 4)  # keeps any per-cell box axis intact
    cols = grid.view(1, 1, S, *trailing)  # varies along the width axis
    rows = grid.view(1, S, 1, *trailing)

    xy = (boxes[..., :2] + torch.stack(torch.broadcast_tensors(cols, rows), dim=-1)) / S
    return torch.cat([xy, boxes[..., 2:4]], dim=-1)


class YOLOLoss(nn.Module):
    """Sum-of-squared-errors loss from the paper, averaged over the batch.

    Only the box with the highest IoU against the ground truth is held "responsible" for a
    cell's object; every other box contributes to the no-object confidence term only.
    """

    def __init__(self, lambda_coord: float = 5.0, lambda_noobj: float = 0.5):
        super().__init__()
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, predictions: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        batch_size = predictions.shape[0]

        pred_boxes = predictions[..., C:].reshape(-1, S, S, B, 5)
        pred_conf, pred_xywh = pred_boxes[..., 0], pred_boxes[..., 1:]
        true_xywh = target[..., C + 1:].unsqueeze(3)  # (N, S, S, 1, 4), broadcasts over B
        has_object = target[..., C:C + 1]             # (N, S, S, 1)

        # Pick the responsible box per cell, then mask out cells with no ground truth.
        ious = iou(to_corners(to_image_frame(pred_xywh)),
                   to_corners(to_image_frame(true_xywh)))
        best_box = ious.argmax(dim=-1, keepdim=True)                      # (N, S, S, 1)
        responsible = F.one_hot(best_box.squeeze(-1), B) * has_object     # (N, S, S, B)

        # Localisation: sqrt on w/h so a fixed error matters more on small boxes. The
        # prediction is unconstrained, so keep the sign; the epsilon matters because
        # d/dx sqrt(|x|) blows up as x -> 0, and a width drifting through zero would
        # otherwise fire an unbounded gradient straight into the head.
        pred_wh = pred_xywh[..., 2:].sign() * (pred_xywh[..., 2:].abs() + 1e-6).sqrt()
        box_error = (pred_xywh[..., :2] - true_xywh[..., :2]).pow(2).sum(-1) \
                    + (pred_wh - true_xywh[..., 2:].sqrt()).pow(2).sum(-1)
        box_loss = (responsible * box_error).sum()

        # Confidence: 1 for the responsible box, 0 everywhere else (paper uses the IoU).
        object_loss = (responsible * (pred_conf - 1).pow(2)).sum()
        no_object_loss = ((1 - responsible) * pred_conf.pow(2)).sum()

        class_loss = (has_object * (predictions[..., :C] - target[..., :C]).pow(2)).sum()

        total = (self.lambda_coord * box_loss + object_loss
                 + self.lambda_noobj * no_object_loss + class_loss)
        return total / batch_size


# --------------------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------------------


@torch.no_grad()
def decode(predictions: torch.Tensor, conf_threshold: float = 0.25,
           iou_threshold: float = 0.5) -> list[dict[str, torch.Tensor]]:
    """Turn raw grid predictions into per-image detections in (x1, y1, x2, y2) form.

    Coordinates are normalised to [0, 1]; multiply by the image size to draw them.
    """
    # Both heads are regressed with plain SSE against one-hot / 1.0 targets, so their raw
    # outputs already are the probabilities -- only clamp them. Re-applying softmax here
    # would rescale a perfect one-hot row to 1/(1 + 79/e) = 0.03 and cap every score.
    class_scores = predictions[..., :C].clamp(0, 1)
    best_class_score, best_class = class_scores.max(dim=-1)             # (N, S, S)

    boxes = predictions[..., C:].reshape(-1, S, S, B, 5)
    scores = boxes[..., 0].clamp(0, 1) * best_class_score.unsqueeze(-1) # (N, S, S, B)
    corners = to_corners(to_image_frame(boxes[..., 1:])).clamp(0, 1)

    detections = []
    for image_boxes, image_scores, image_classes in zip(
        corners.flatten(1, 3), scores.flatten(1, 3),
        best_class.unsqueeze(-1).expand(-1, -1, -1, B).flatten(1, 3),
    ):
        # Empty boxes (clamped-away negative w/h, or a box entirely outside the frame)
        # score just like real ones but cover nothing, and NMS cannot suppress them.
        areas = (image_boxes[:, 2] - image_boxes[:, 0]) * (image_boxes[:, 3] - image_boxes[:, 1])
        keep = (image_scores > conf_threshold) & (areas > 0)
        image_boxes, image_scores, image_classes = (
            image_boxes[keep], image_scores[keep], image_classes[keep])

        # Offset boxes by class so NMS never suppresses across different classes.
        keep = nms(image_boxes + image_classes.unsqueeze(-1), image_scores, iou_threshold)
        detections.append({
            "boxes": image_boxes[keep],
            "scores": image_scores[keep],
            "labels": image_classes[keep],
        })

    return detections


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------


@torch.no_grad()
def recalibrate_batchnorm(model: nn.Module, loader: DataLoader, device: torch.device) -> None:
    """Recompute every BatchNorm's running mean/var as an exact average over the dataset.

    The running stats are an EMA of batch statistics, and at batch_size 8 they trail the
    weights badly: a freshly trained checkpoint here scored 15.9 in train() mode but 24.2
    in eval() mode, which is exactly the "training looked fine, predictions are garbage"
    symptom. momentum=None switches BatchNorm to a cumulative average, so one clean pass
    over the data replaces the EMA with the real thing.
    """
    norms = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    for norm in norms:
        norm.reset_running_stats()
        norm.momentum = None

    model.train()
    for images, _ in loader:
        model(images.to(device, non_blocking=True))

    for norm in norms:
        norm.momentum = 0.1
    model.eval()


def train(config: TrainConfig = TrainConfig()) -> YOLOv1:
    """Fits the 128-image sample set. This is a pipeline sanity check, not a detector:
    with no pretrained backbone and 128 images covering 71 of the 80 classes, the best
    honest outcome is memorising the training set. If the loss plateaus well above ~2 the
    bug is in the setup; if it drops near 0 and predictions still look wrong, the bug is
    in decode()."""
    dataset = Coco128Dataset(DATA_ROOT)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        # pin_memory only helps CUDA's host->device copies; on MPS it just warns.
        pin_memory=config.device.type == "cuda",
        persistent_workers=config.num_workers > 0,
    )

    model = YOLOv1().to(config.device)
    criterion = YOLOLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr,
                                 weight_decay=config.weight_decay)

    steps = config.epochs * len(loader)
    print(f"{len(dataset)} images on {config.device} — {len(loader)} steps/epoch, "
          f"{steps} updates total")

    model.train()
    for epoch in range(1, config.epochs + 1):
        running_loss, started = 0.0, time.time()

        for images, targets in loader:
            images = images.to(config.device, non_blocking=True)
            targets = targets.to(config.device, non_blocking=True)

            loss = criterion(model(images), targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # The loss sums squared errors over 49 cells, so early batches can produce
            # gradients big enough to knock the net into a constant-output collapse.
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            running_loss += loss.item() * images.shape[0]

        print(f"epoch {epoch:3d}/{config.epochs}  loss {running_loss / len(dataset):.4f}"
              f"  ({time.time() - started:.1f}s)")

    recalibrate_batchnorm(model, loader, config.device)
    return model


def parse_args() -> TrainConfig:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers)
    parser.add_argument("--device", default=None, help="cpu, cuda, mps (default: auto)")
    args = parser.parse_args()

    return TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                       num_workers=args.num_workers, device=pick_device(args.device))


if __name__ == "__main__":
    model = train(parse_args())
    torch.save(model.state_dict(), Path(__file__).parent / "yolov1.pt")
