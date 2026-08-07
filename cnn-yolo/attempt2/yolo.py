import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from dset import process_dataset
import PIL.Image as Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

S = 7  # grid size
B = 2  # boxes predicted per cell
C = 80 # COCO classes
IMG_SIZE = 448
CELL_VECTOR = C + B * 5 

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
            nn.Dropout(dropout),
            nn.LeakyReLU(0.1),
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
        return self.head(self.backbone(x)).reshape(-1, S, S, CELL_VECTOR) # shape is (N, S, S, CELL_VECTOR)


class YoloLoss(nn.Module):
    def __init__(self):
        super(YoloLoss, self).__init__()

    def forward(self, predictions : torch.Tensor, targets: torch.Tensor):
        """
        Predictions has shape (B, S, S, CELL_VECTOR)
        Targets has shape (..., 5)
        """


def modify_test_labels(test_labels: np.ndarray) -> torch.Tensor:
    """
    Converts test labels of shape (..., 5) that have:
    - Class
    - Center_x (full image)
    - Center_y (full image)
    - Width (full image)
    - Height (full image)

    into test labels of shape (..., 7) that have:
    - Class
    - Grid X Index (0...S-1)
    - Grid Y Index (0...S-1)
    - Relative Grid Center_x (relative to grid X, Y)
    - Relative Grid Center_y (relative to grid X, Y)
    - Width (full image)
    - Height (full image)
    """

    test_labels[..., 1:3] = S * test_labels[..., 1:3]
    x = test_labels[..., 1].copy()
    y = test_labels[..., 2].copy()

    x_floor = np.floor(x)
    y_floor = np.floor(y)

    test_labels[..., 1] = x_floor
    test_labels[..., 2] = y_floor
    
    test_labels = np.insert(test_labels, 3, x - x_floor, axis=1)
    test_labels = np.insert(test_labels, 4, y - y_floor, axis=1)

    test_labels = test_labels[np.lexsort((test_labels[:, 2], test_labels[:, 1]))]

    delete_indexes = []
    for i in range(test_labels.shape[0] - 1):
        if test_labels[i, 1] == test_labels[i + 1, 1] and test_labels[i, 2] == test_labels[i + 1, 2]:
            delete_indexes.append(i)

    test_labels = np.delete(test_labels, delete_indexes, axis=0)
    return torch.tensor(test_labels)

def convert_to_box_corners(box_data: torch.Tensor) -> torch.Tensor:
    """
    Box data is: (Class, X Grid Index, Y Grid Index, Relative X Index, Relative Y Index, W, H)
    Dim: (..., 7)

    Outputs: 
    [
    (Top Left X, Top Left Y),
    (Bottom Right X, Bottom Right Y)
    ]
    Dim: Tensor(2, 2)
    """

    w = box_data[5] * IMG_SIZE
    h = box_data[6] * IMG_SIZE

    top_left_X = (box_data[1] + box_data[3]) * IMG_SIZE / S - w / 2
    top_left_Y = (box_data[2] + box_data[4]) * IMG_SIZE / S - h / 2

    bottom_right_X = (box_data[1] + box_data[3]) * IMG_SIZE / S + w / 2
    bottom_right_Y = (box_data[2] + box_data[4]) * IMG_SIZE / S + h / 2

    corners_tensor = torch.tensor(
        [[top_left_X, top_left_Y],
         [bottom_right_X, bottom_right_Y]],
        dtype=torch.float32
    )

    return corners_tensor


def compute_iou(corner_tensor_1: torch.Tensor, corner_tensor_2: torch.Tensor) -> float:
    ...

"""
TEST DATASET LOAD
"""

sample_index = 40

images, labels = process_dataset()
images_ds = TensorDataset(images)
test_img = images_ds[sample_index][0]
test_img = test_img.permute(2, 0, 1).unsqueeze(dim=0)
test_label = labels[sample_index]

fig, ax = plt.subplots()

ax.imshow(Image.fromarray(images_ds[sample_index][0].detach().numpy().astype("uint8")))

modified_labels = modify_test_labels(test_label)
label_i = 0

for l in modified_labels:
    corner_ls = convert_to_box_corners(l)

    # Computing using original data
    # c_x = (l[1] + l[3]) * IMG_SIZE / S
    # c_y = (l[2] + l[4]) * IMG_SIZE / S

    # w = l[5] * IMG_SIZE
    # h = l[6] * IMG_SIZE

    # print(c_x - w / 2, c_y - h / 2)

    rect = patches.Rectangle((corner_ls[0, 0], corner_ls[0, 1]), corner_ls[1, 0] - corner_ls[0, 0], corner_ls[1, 1] - corner_ls[0, 1], linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)    

plt.show()









model = YoloModel()

output : torch.Tensor = model(test_img)

print(output.shape)