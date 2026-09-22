"""graph_ae 32 csatornaval. Minden ugyanaz, mint a notebookban - csak a
z_channels 16 helyett 32, es lidar/graph_ae_32.ckpt-be ment.
"""
import glob
import os
import time

import numpy as np
import torch
from tqdm.auto import tqdm

from lidar.graph_ae import LidarAE, DDCONFIG, points_to_range_image

DATA_DIR = "dataset/lidar"
EPOCHS = 40
BATCH = 64
LR = 4.5e-6
CKPT = "lidar/graph_ae_32.ckpt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(0)
np.random.seed(0)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# --- Adat: UGYANAZ a vagas, mint a notebookban ----------------------------
PATHS = sorted(glob.glob(os.path.join(DATA_DIR, "*.npy")))
PERM = np.random.default_rng(0).permutation(len(PATHS))
N_TEST = int(len(PATHS) * 0.15)
N_VAL = int(len(PATHS) * 0.15)
SPANS = {"test": (0, N_TEST), "val": (N_TEST, N_TEST + N_VAL),
         "train": (N_TEST + N_VAL, len(PATHS))}

SHAPE = points_to_range_image(np.load(PATHS[0])).shape
print(f"{len(PATHS)} frame   range image {SHAPE}")


def load_ranges(name, workers=16):
    from concurrent.futures import ThreadPoolExecutor
    lo, hi = SPANS[name]
    idx = PERM[lo:hi]
    arr = np.empty((len(idx), *SHAPE), dtype=np.float32)

    def read_one(k):
        arr[k] = points_to_range_image(np.load(PATHS[idx[k]]))

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(tqdm(ex.map(read_one, range(len(idx))), total=len(idx),
                  desc=f"{name:5s}", unit="frame"))
    return torch.from_numpy(arr)


def evaluate(model, data, batch):
    model.eval()
    total, n, acc = 0.0, 0, {}
    with torch.no_grad():
        for i in range(0, len(data), batch):
            xb = data[i:i + batch].to(DEVICE, non_blocking=True).float()
            loss, p = model.loss(xb, parts=True)
            for k, v in p.items():
                acc[k] = acc.get(k, 0.0) + v * len(xb)
            total += float(loss) * len(xb)
            n += len(xb)
    return total / n, {k: v / n for k, v in acc.items()}


r_train, r_val = load_ranges("train"), load_ranges("val")
print(f"train {tuple(r_train.shape)}  val {tuple(r_val.shape)}  "
      f"{(r_train.numel() + r_val.numel()) * 4 / 1e9:.1f} GB")

# --- A modell: csak a z_channels mas --------------------------------------
cfg = dict(DDCONFIG)
cfg["z_channels"] = 32
model = LidarAE(cfg).to(DEVICE)

_base = LidarAE(DDCONFIG)
print(f"\nz_channels 16 -> 32   "
      f"parameter {sum(p.numel() for p in _base.parameters()) / 1e6:.2f}M -> "
      f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M   "
      f"latens {_base.latent_shape} -> {model.latent_shape}")
del _base

opt = torch.optim.Adam(model.parameters(), lr=LR)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt, mode="min", factor=0.5, patience=2, min_lr=LR / 100)

best, bad, start_ep, best_state = float("inf"), 0, 1, None
hist = {"train": [], "val": [], "diag": [], "lr": []}

# --- Folytatas, ha van checkpoint -----------------------------------------
if os.path.exists(CKPT):
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["state_dict"])
    opt.load_state_dict(ck["optimizer"])
    sched.load_state_dict(ck["sched"])
    hist = {k: list(ck[k]) for k in ("train", "val", "diag", "lr")}
    best, bad = ck["best_val"], ck["bad"]
    best_state = ck["best_state"]
    start_ep = ck["epoch"] + 1
    print(f"FOLYTATAS a {ck['epoch']}. epoch utan (best {best:.5f})")

print(f"\ngraph_ae_32  (lr={LR}, batch={BATCH}, {len(r_train)} train frame, "
      f"epoch {start_ep}-{EPOCHS})")
t0 = time.time()

for ep in range(start_ep, EPOCHS + 1):
    model.train()
    run, n = 0.0, 0
    perm = torch.randperm(len(r_train))
    bar = tqdm(range(0, len(r_train), BATCH), desc=f"epoch {ep}/{EPOCHS}",
               leave=False)
    for i in bar:
        xb = r_train[perm[i:i + BATCH]].to(DEVICE, non_blocking=True).float()
        loss = model.loss(xb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        opt.step()
        run += float(loss.detach()) * len(xb)
        n += len(xb)
        bar.set_postfix(loss=f"{float(loss.detach()):.5f}")

    tr = run / n
    va, diag = evaluate(model, r_val, BATCH)

    # A graph_ae-t az URES teruletek dominaljak, ezert a foglalt teruleten
    # mert L1 a kovetett metrika - nem a nyers loss.
    score = diag.get("l1_occupied", va)
    sched.step(score)

    hist["train"].append(tr)
    hist["val"].append(va)
    hist["diag"].append(diag)
    hist["lr"].append(opt.param_groups[0]["lr"])

    improved = score < best
    if improved:
        best, bad = score, 0
        best_state = {k: v.detach().cpu().clone()
                      for k, v in model.state_dict().items()}
    else:
        bad += 1

    # Atomikus mentes minden epoch vegen: .tmp, aztan atnevezes.
    tmp = CKPT + ".tmp"
    torch.save({"state_dict": {k: v.detach().cpu()
                               for k, v in model.state_dict().items()},
                "best_state": best_state,
                "optimizer": opt.state_dict(),
                "sched": sched.state_dict(),
                "epoch": ep, "bad": bad,
                "hparams": cfg,
                "arch": "LidarAE",
                "best_val": best,
                **hist}, tmp)
    os.replace(tmp, CKPT)

    extra = "  ".join(f"{k} {v:.4f}" for k, v in diag.items()
                      if k in ("l1_occupied", "l1_empty"))
    print(f"  epoch {ep:2d}  train {tr:.5f}  val {va:.5f}   {extra}"
          f"  lr {opt.param_groups[0]['lr']:.2e}"
          f"  [{(time.time() - t0) / 60:.0f} perc]"
          f"{'  *' if improved else ''}", flush=True)

    if bad >= 5:
        print("  early stop (5 epoch javulas nelkul)")
        break

model.load_state_dict(best_state)
print(f"\nkesz: {CKPT}  (best {best:.5f}, {(time.time() - t0) / 60:.1f} perc)")
