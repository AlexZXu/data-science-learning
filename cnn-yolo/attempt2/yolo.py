import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from dset import process_dataset
import PIL.Image as Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math

S = 7  # grid size
B = 2  # boxes predicted per cell
C = 80 # COCO classes
IMG_SIZE = 448
CELL_VECTOR = C + B * 5 
GAMMA_COORD = 5.0
GAMMA_NOOBJ = 0.5

architecture = [
    (7, 64, 2, 3), "M",
    (3, 192, 1, 1), "M",
    (1, 128, 1, 0), (3, 256, 1, 1), (1, 256, 1, 0), (3, 512, 1, 1), "M",
    [
        (1, 256, 1, 0),
        (3, 512, 1, 1),
        4
    ], (1, 512, 1, 0), (3, 1024, 1, 1), "M",
    [
        (1, 512, 1, 0),
        (3, 1024, 1, 1),
        2
    ], (3, 1024, 1, 1), (3, 1024, 2, 1),
    (3, 1024, 1, 1), (3, 1024, 1, 1)
]


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(self.norm(self.conv(x)), 0.1)


class YoloModel(nn.Module):
    def __init__(self):
        super(YoloModel, self).__init__()
        self.backbone = self._create_backbone()

        dropout = 0.1
        hidden = 1024

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, hidden),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden, S * S * CELL_VECTOR),
        )

    def _create_backbone(self):
        arch_list = []
        in_channels = 3

        for arc in architecture:
            if isinstance(arc, tuple):
                convblock = ConvBlock(in_channels, arc[1], kernel_size=arc[0], stride=arc[2], padding=arc[3])
                arch_list.append(convblock)
                in_channels = arc[1]
            elif isinstance(arc, str): #"M"
                maxpool = nn.MaxPool2d(2, 2)
                arch_list.append(maxpool)
            else:
                reps = arc[2]
                copy_1 = arc[0]
                copy_2 = arc[1]

                for _ in range(reps):
                    convblock1 = ConvBlock(in_channels, copy_1[1], kernel_size=copy_1[0], stride=copy_1[2], padding=copy_1[3])
                    in_channels = copy_1[1]
                    convblock2 = ConvBlock(in_channels, copy_2[1], kernel_size=copy_2[0], stride=copy_2[2], padding=copy_2[3])
                    in_channels = copy_2[1]

                    arch_list.extend([convblock1, convblock2])

        return nn.Sequential(*arch_list)

    def forward(self, x):
        logits = self.head(self.backbone(x))
        # every target lives in [0, 1] (one-hot classes, IoU confidence, cell-relative
        # centres, normalised w/h), so squash the head. Unbounded outputs let the
        # sqrt(w) term produce huge gradients and the loss diverges.
        return logits.reshape(-1, S, S, CELL_VECTOR) # shape is (N, S, S, CELL_VECTOR)


class YoloLoss(nn.Module):
    def __init__(self):
        super(YoloLoss, self).__init__()

    def forward(self, predictions : torch.Tensor, targets: torch.Tensor):
        """
        Predictions has shape (B, S, S, C + 5 * B)
        Targets has shape (B, S, S, C + 5)
        """

        loss = 0.0
        for batch_idx in range(targets.shape[0]):
            for row in range(S):
                for col in range(S):
                    cell_target = targets[batch_idx, row, col]
                    cell_preds = predictions[batch_idx, row, col]

                    curr_loss = 0.0

                    if cell_target[-5] > 0:
                        # targets are stored [row, col] == [y cell, x cell]
                        target_box_corners = convert_to_box_corners(col, row, cell_target[-4:])

                        ious = []
                        for box in range(B):
                            pred_box_corners = convert_to_box_corners(col, row, cell_preds[C + box * 5 + 1 : C + (box + 1) * 5])
                            ious.append(compute_iou(target_box_corners, pred_box_corners))

                        # the box with the highest IoU is "responsible" for this object
                        responsible = int(torch.stack(ious).argmax())

                        for box in range(B):
                            pred_confidence = cell_preds[C + box * 5]

                            if box != responsible:
                                curr_loss += GAMMA_NOOBJ * (0 - pred_confidence)**2
                                continue

                            curr_loss += GAMMA_COORD * (
                                (cell_target[C + 1] - cell_preds[C + box * 5 + 1])**2
                                +
                                (cell_target[C + 2] - cell_preds[C + box * 5 + 2])**2
                            )

                            pred_w = cell_preds[C + box * 5 + 3]
                            pred_h = cell_preds[C + box * 5 + 4]

                            target_w = cell_target[C + 3]
                            target_h = cell_target[C + 4]

                            # predictions are sigmoid-bounded so they are already >= 0;
                            # the clamp only keeps d(sqrt)/dx finite near zero
                            curr_loss += GAMMA_COORD * (
                                (torch.sqrt(target_w.clamp(min=1e-4)) - torch.sqrt(pred_w.clamp(min=1e-4))) ** 2
                                +
                                (torch.sqrt(target_h.clamp(min=1e-4)) - torch.sqrt(pred_h.clamp(min=1e-4))) ** 2
                            )

                            curr_loss += (ious[box].detach() - pred_confidence)**2

                        for cl in range(C):
                            curr_loss += (cell_target[cl] - cell_preds[cl])**2

                    else:
                        for box in range(B):
                            pred_confidence = cell_preds[C + box * 5]
                            curr_loss += GAMMA_NOOBJ * (0 - pred_confidence)**2

                    loss += curr_loss

        return loss / targets.shape[0]


def convert_label_set(labels) -> torch.Tensor:
    new_labels = []
    for label_sample in labels:
        label_sample = label_sample[np.lexsort((label_sample[:, 2], label_sample[:, 1]))]
        delete_indexes = []
        for i in range(label_sample.shape[0] - 1):
            if label_sample[i, 1] == label_sample[i + 1, 1] and label_sample[i, 2] == label_sample[i + 1, 2]:
                delete_indexes.append(i)

        label_sample = np.delete(label_sample, delete_indexes, axis=0)

        reshaped_label_sample = np.zeros((S, S, C + 5), dtype=np.float32)

        for l in label_sample:
            # a centre of exactly 1.0 would floor to S, so clamp into the grid
            x_grid = min(math.floor(l[1] * S), S - 1)
            y_grid = min(math.floor(l[2] * S), S - 1)

            class_arr = np.zeros(C, dtype=np.float32)
            class_arr[int(l[0])] = 1.0

            box_vec = np.array([1.0, l[1] * S - x_grid, l[2] * S - y_grid, l[3], l[4]], dtype=np.float32)

            # stored [row, col] == [y cell, x cell] to match the image layout
            reshaped_label_sample[y_grid, x_grid] = np.hstack((class_arr, box_vec))

        new_labels.append(reshaped_label_sample)

    return torch.from_numpy(np.array(new_labels))



def convert_to_box_corners(x_index, y_index, box_data: torch.Tensor) -> torch.Tensor:
    """
    Box data is: (Relative X Index, Relative Y Index, W, H)
    Dim: (..., 4)

    Outputs: 
    [
    (Top Left X, Top Left Y),
    (Bottom Right X, Bottom Right Y)
    ]
    Dim: Tensor(2, 2)
    """

    x_center = (x_index + box_data[..., 0]) * IMG_SIZE / S
    y_center = (y_index + box_data[..., 1]) * IMG_SIZE / S

    width = box_data[..., 2] * IMG_SIZE
    height = box_data[..., 3] * IMG_SIZE

    top_left = torch.stack(
        [
            x_center - width / 2,
            y_center - height / 2,
        ],
        dim=-1,
    )

    bottom_right = torch.stack(
        [
            x_center + width / 2,
            y_center + height / 2,
        ],
        dim=-1,
    )

    return torch.stack([top_left, bottom_right], dim=-2)


def compute_iou(corners_1: torch.Tensor, corners_2: torch.Tensor, eps: float = 1e-6,) -> float:
    left = torch.maximum(corners_1[..., 0, 0], corners_2[..., 0, 0])
    top = torch.maximum(corners_1[..., 0, 1], corners_2[..., 0, 1])

    right = torch.minimum(corners_1[..., 1, 0], corners_2[..., 1, 0])
    bottom = torch.minimum(corners_1[..., 1, 1], corners_2[..., 1, 1])

    intersection_width = (right - left).clamp(min=0)
    intersection_height = (bottom - top).clamp(min=0)
    intersection = intersection_width * intersection_height

    width_1 = (corners_1[..., 1, 0] - corners_1[..., 0, 0]).clamp(min=0)
    height_1 = (corners_1[..., 1, 1] - corners_1[..., 0, 1]).clamp(min=0)

    width_2 = (corners_2[..., 1, 0] - corners_2[..., 0, 0]).clamp(min=0)
    height_2 = (corners_2[..., 1, 1] - corners_2[..., 0, 1]).clamp(min=0)

    area_1 = width_1 * height_1
    area_2 = width_2 * height_2

    union = area_1 + area_2 - intersection

    return intersection / union.clamp(min=eps)

"""
TEST DATASET LOAD
"""

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train(epochs: int = 0, batch_size: int = 16, lr: float = 1e-4):
    device = get_device()
    print(f"training on {device}")

    images, labels = process_dataset()
    images_ds = images.permute(0, 3, 1, 2)
    new_labels = convert_label_set(labels)

    dataset = TensorDataset(
        images_ds,
        new_labels
    )

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    model = YoloModel().to(device)
    criterion = YoloLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        running_loss = 0.0
        for data, label in data_loader:
            data, label = data.to(device), label.to(device)

            optimizer.zero_grad()

            output : torch.Tensor = model(data)

            loss = criterion(output, label)

            loss.backward()
            # the summed SSE loss produces large gradients early on
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()

            running_loss += loss.item()

        running_loss /= len(data_loader)
        print(f"epoch {epoch + 1}/{epochs}  loss {running_loss:.4f}")

    return model

def predict(model: YoloModel):
    device = get_device()
    images, labels = process_dataset()

    print(images[0].shape)

    convert_to_image = Image.fromarray(
        (images[0] * 255)
        .clamp(0, 255)
        .to(torch.uint8)
        .cpu()
        .numpy()
    )

    # plt.imshow(convert_to_image)
    # plt.show()

    converted = images[0].unsqueeze(0).permute(0, 3, 1, 2).to(device)
    print(converted.shape)
    # print(images[0].unsqueeze(0).permute(0, 3, 1, 2).shape)
    with torch.no_grad():
        preds = model(converted)

        conf_scores = preds[..., torch.arange(80, CELL_VECTOR, 5)]
        

        print(preds.shape)

if __name__ == "__main__":
    model = train()
    predict(model)


# fig, ax = plt.subplots()

# ax.imshow(Image.fromarray(images_ds[sample_index][0].detach().numpy().astype("uint8")))

# modified_labels = modify_test_labels(test_label)
# label_i = 0

# for l in modified_labels:
#     corner_ls = convert_to_box_corners(l)

    # Computing using original data
    # c_x = (l[1] + l[3]) * IMG_SIZE / S
    # c_y = (l[2] + l[4]) * IMG_SIZE / S

    # w = l[5] * IMG_SIZE
    # h = l[6] * IMG_SIZE

    # print(c_x - w / 2, c_y - h / 2)

    # rect = patches.Rectangle((corner_ls[0, 0], corner_ls[0, 1]), corner_ls[1, 0] - corner_ls[0, 0], corner_ls[1, 1] - corner_ls[0, 1], linewidth=2, edgecolor='r', facecolor='none')
    # ax.add_patch(rect)    

# plt.show()






