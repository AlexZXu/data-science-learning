"""Recover the ground-truth labels for the Kaggle CIFAR-10 test images.

The Kaggle competition ships `test/` as 300,000 unlabeled PNGs. Only 10,000 of
them are the real CIFAR-10 test set; the other 290,000 are junk images added so
that nobody can hand-label their way to a good score. Kaggle also states that
the real 10,000 were "trivially modified" to stop people from looking them up by
file hash.

The original test set is public though (cs.toronto.edu), labels and all. So the
test set can be reconstructed by matching: for each official test image, find
the Kaggle image it corresponds to. Two passes:

  1. exact match on decoded pixels (md5) -- free, and catches everything if the
     images were left untouched;
  2. nearest neighbour in raw pixel space for whatever pass 1 missed. 3072-dim
     L2 against all 300,000 candidates, done as one matrix product per chunk on
     the GPU. A "trivial modification" leaves an image far closer to its own
     original than to any other picture, so the nearest neighbour is the match,
     and the printed distance gap is the evidence that it is.

Output: cifar-10/testLabels.csv, `id,label`, same format as trainLabels.csv,
holding only the 10,000 real test images.

Run from this directory:  python recover_test_labels.py
The first run downloads the ~170 MB official archive into cifar-10/official/.
"""
import csv
import hashlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.datasets import CIFAR10

HERE = Path(__file__).parent
KAGGLE_TEST = HERE / "cifar-10" / "test"
OFFICIAL_ROOT = HERE / "cifar-10" / "official"
OUT_CSV = HERE / "cifar-10" / "testLabels.csv"

N_KAGGLE = 300_000
CHUNK = 4096  # Kaggle images per distance-matrix chunk


def load_png(id):
    """Kaggle test image `id` as [32, 32, 3] uint8."""
    return np.asarray(Image.open(KAGGLE_TEST / f"{id}.png").convert("RGB"), dtype=np.uint8)


def load_kaggle():
    """All 300,000 test images, row i holding id i + 1.

    ~900 MB as uint8, and eight processes decode it in well under a minute --
    cheap enough not to bother caching.
    """
    with Pool(8) as pool:
        images = pool.map(load_png, range(1, N_KAGGLE + 1), chunksize=500)
    return np.stack(images)


def exact_matches(official, kaggle):
    """{official index: kaggle id} for images that survived byte-identical."""
    by_hash = {
        hashlib.md5(official[i].tobytes()).digest(): i for i in range(len(official))
    }
    found = {}
    for j in range(len(kaggle)):
        i = by_hash.get(hashlib.md5(kaggle[j].tobytes()).digest())
        if i is not None:
            found[i] = j + 1
    return found


def nearest_matches(official, kaggle, device):
    """Nearest Kaggle image for every official image, by L2 over raw pixels.

    Returns (kaggle ids, best distances, runner-up distances). The two distance
    arrays are the sanity check: a real match sits near zero while the runner-up
    sits far away, so any overlap between them means the matching is guessing.
    """
    queries = torch.from_numpy(official.reshape(len(official), -1)).to(device, torch.float16)
    q_sq = (queries.float() ** 2).sum(1)[:, None]

    n = len(official)
    best = torch.full((n,), float("inf"), device=device)
    second = torch.full((n,), float("inf"), device=device)
    best_j = torch.zeros(n, dtype=torch.long, device=device)

    for start in range(0, len(kaggle), CHUNK):
        block = torch.from_numpy(
            kaggle[start:start + CHUNK].reshape(-1, queries.shape[1])
        ).to(device, torch.float16)

        # |q - k|^2 = |q|^2 - 2 q.k + |k|^2, so the whole chunk is one matmul.
        # fp16 for the product (it is what makes this fast on a GPU) and fp32
        # for the sums, which are the terms large enough to lose precision.
        dist = q_sq - 2.0 * (queries @ block.T).float() + (block.float() ** 2).sum(1)[None, :]

        # Two per chunk, because the runner-up may live in the same chunk as the
        # winner. Merge into the running best/second.
        top = dist.topk(2, dim=1, largest=False)
        cand_v = torch.cat([torch.stack([best, second], 1), top.values], 1)
        cand_j = torch.cat(
            [torch.stack([best_j, best_j], 1), top.indices + start], 1
        )
        order = cand_v.argsort(dim=1)
        best = cand_v.gather(1, order[:, 0:1]).squeeze(1)
        second = cand_v.gather(1, order[:, 1:2]).squeeze(1)
        best_j = cand_j.gather(1, order[:, 0:1]).squeeze(1)

        print(f"  scanned {min(start + CHUNK, len(kaggle)):>6}/{len(kaggle)}", end="\r")

    print()
    return (
        best_j.cpu().numpy() + 1,
        best.clamp(min=0).sqrt().cpu().numpy(),
        second.clamp(min=0).sqrt().cpu().numpy(),
    )


def main():
    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    official_ds = CIFAR10(root=OFFICIAL_ROOT, train=False, download=True)
    official = np.asarray(official_ds.data, dtype=np.uint8)  # [10000, 32, 32, 3]
    official_labels = np.asarray(official_ds.targets)
    class_names = official_ds.classes
    print(f"official test set: {official.shape}, classes {class_names}")

    print("decoding Kaggle test images...")
    kaggle = load_kaggle()
    print(f"kaggle test set: {kaggle.shape}")

    exact = exact_matches(official, kaggle)
    print(f"exact pixel matches: {len(exact)}/{len(official)}")

    if len(exact) == len(official):
        match_id = np.array([exact[i] for i in range(len(official))])
    else:
        print(f"nearest-neighbour matching the remaining {len(official) - len(exact)}...")
        match_id, best, second = nearest_matches(official, kaggle, device)
        for i, j in exact.items():  # trust an exact hit over a distance
            match_id[i] = j
        print(
            f"match distance:     max {best.max():8.2f}  mean {best.mean():8.2f}\n"
            f"runner-up distance: min {second.min():8.2f}  mean {second.mean():8.2f}"
        )

    unique = len(np.unique(match_id))
    print(f"distinct Kaggle ids matched: {unique}/{len(official)}")
    if unique != len(official):
        # Two official images claiming the same Kaggle image means at least one
        # is wrong, and a wrong label is worse than a missing one.
        raise SystemExit("collision in the matching -- refusing to write labels")

    order = np.argsort(match_id)
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "label"])
        for i in order:
            writer.writerow([int(match_id[i]), class_names[official_labels[i]]])
    print(f"wrote {OUT_CSV} ({len(order)} rows)")


if __name__ == "__main__":
    main()
