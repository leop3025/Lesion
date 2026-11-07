from __future__ import annotations

from pathlib import Path
import csv
import importlib.util
import json
import math
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import binary_dilation, distance_transform_edt, generate_binary_structure, map_coordinates
from scipy.stats import wilcoxon


# ============================================================
# STAGE 19 — EXPLAINABILITY (baseline U-Net v2 vs distance U-Net v2)
# ============================================================
#
# Models (final 70-patient models, already evaluated on the test set):
#   baseline_unet_v2  <- 14_train_baseline_unet_v2.py   (9 channels)
#   distance_unet_v2  <- 17_distance_map_unet_v2.py     (9 channels + DIST)
# Swin UNETR (file 18) is intentionally NOT analysed.
#
# Nothing is trained or tuned here; all settings below are fixed in
# advance. Inputs, TTA and post-processing are imported from scripts 14/17,
# so every prediction is made exactly as in the reported test evaluation.
#
# Analyses (30 test patients):
#   1. Channel ablation   — re-predict with one input removed / replaced:
#        no_FLAIR, no_T2, no_T1 (all context slices set to 0),
#        no_neighbour_slices (z-1, z+1 replaced by slice z = plain 2D),
#        distance model only: dist_zero (DIST = 0, unusual-input stress test) and
#        dist_counterfactual: anatomically controlled counterfactual distance-map
#          perturbation (not a clinically validated anatomical replacement) —
#          (a) donor = development patient with the most similar ventricles
#              (volume, position relative to the brain, spread per axis),
#          (b) patients are NOT in a common space (script 10 uses each
#              patient's own FLAIR grid), so the donor brain is registered to
#              the target brain (world mm: centroid + per-axis scaling) and the
#              donor ventricle mask resampled with nearest neighbour,
#          (c) plausibility / registration QC; failures try the next donor,
#              no suitable donor -> skipped with the reason saved,
#          (d) distance map recomputed from the transformed mask exactly as
#              in 15a (3D EDT in mm, clip at D_max, / D_max),
#          (e) affected voxels (PV/deep assignment changes) counted, and
#              predictions compared inside them and overall, including the
#              fraction of labels that actually SWITCH in the direction of the
#              new geometry (cf_switch_consistent), not only agreement.
#      Dice/sensitivity/precision per target, PV/deep consistency with the TRUE
#      and the COUNTERFACTUAL geometry (10 mm rule), and voxel-wise changes vs
#      the original prediction.
#   2. Performance by distance to the ventricles — saved test predictions,
#      sensitivity / precision / Dice / FP volume in distance bands, and
#      PV/deep label accuracy of correctly detected abnormal voxels per band.
#   3. Integrated Gradients (Sundararajan et al., ICML 2017) — attribution of
#      the predicted PV / deep abnormal-WMH region of one pre-specified slice
#      per patient (slice with most GT abnormal WMH) to every input channel.
#      The explained region comes from a single forward pass (no TTA).
#   4. Uncertainty — two separate quantities from the 4 TTA predictions:
#        predictive entropy  H(mean softmax)            (total uncertainty)
#        TTA disagreement    mutual information = H(mean) - mean H(flip)
#                            + fraction of flips whose label differs
#      (Wang et al., Neurocomputing 2019) and their relation to abnormal-WMH
#      DETECTION errors (PV<->deep confusions count as correct), whole brain
#      and inside a lesion region of interest (dilated GT | prediction).
#   5. Rule-based comparison — each model's saved prediction with abnormal
#      voxels relabelled PV/deep by the 10 mm rule; paired comparison of the
#      distance model against "baseline + 10 mm rule" (Holm-corrected).
#
# Ground-truth dependencies (stated explicitly):
#   - The distance channel of the distance model is computed from the EXPERT
#     ventricle mask, also at test time (inherited from script 17); results
#     therefore assume expert-quality ventricle masks. The same holds for the
#     10 mm rule in analysis 5.
#   - Ground-truth labels are used only to choose the IG slice / example
#     patients (fixed, label-based rules) and for evaluation — never as input.
#
# Outputs: logs/explainability/   (refuses to overwrite unless --overwrite)
#
# Usage:
#   python 19_explainability.py --check            # preflight: 1 patient, writes nothing
#   python 19_explainability.py                    # full analysis
#   python 19_explainability.py --skip-ig          # skip slower analyses
#   flags: --skip-ablation --skip-bands --skip-ig --skip-uncertainty --skip-rule --overwrite
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
OUT_ROOT = PROJECT_DIR / "logs" / "explainability"
DISTANCE_MM_ROOT = PROJECT_DIR / "distance_map"
PREPROCESSED_ROOT = PROJECT_DIR / "preprocessed"

MODELS = {
    "baseline_unet_v2": "14_train_baseline_unet_v2.py",
    "distance_unet_v2": "17_distance_map_unet_v2.py",
}

SKIP_ABLATION = "--skip-ablation" in sys.argv
SKIP_BANDS = "--skip-bands" in sys.argv
SKIP_IG = "--skip-ig" in sys.argv
SKIP_UNCERTAINTY = "--skip-uncertainty" in sys.argv
SKIP_RULE = "--skip-rule" in sys.argv
CHECK_ONLY = "--check" in sys.argv
OVERWRITE = "--overwrite" in sys.argv

PV_THRESHOLD_MM = 10.0
ABNORMAL = (3, 4)
PV_LABEL, DEEP_LABEL = 3, 4

# fixed in advance (not tuned on results)
DISTANCE_BANDS_MM = [(0.0, 5.0), (5.0, 10.0), (10.0, 20.0), (20.0, 35.0), (35.0, math.inf)]
BAND_TARGETS = {"abnormal_wmh": (3, 4), "normal_wmh": (2,)}
ABLATION_TARGETS = ["abnormal_wmh", "pv_abnormal_wmh", "deep_abnormal_wmh", "normal_wmh", "all_wmh"]
IG_STEPS = 64                  # Riemann (midpoint) steps from the all-zero baseline (same for every case)
IG_CHUNK = 4                   # interpolation steps per forward/backward pass (GPU memory)
IG_CLASSES = {"pv_abnormal_wmh": PV_LABEL, "deep_abnormal_wmh": DEEP_LABEL}
IG_MIN_REGION_VOXELS = 5       # skip a class on a slice if the predicted region is smaller
IG_COMPLETENESS_TOL = 0.05     # |sum(attr) - (F(x) - F(0))| / |F(x) - F(0)| above this is flagged
IG_MIN_SCORE_DELTA = 1e-3      # |F(x) - F(0)| below this: zero baseline uninformative, flagged
N_EXAMPLE_PATIENTS = 4         # chosen by GT abnormal-WMH burden quantiles (label-based)
N_BOOTSTRAP = 5000
BOOTSTRAP_SEED = 20260911
ENTROPY_BINS = 1000
UNC_ROI_DILATION = 3           # in-plane dilation (voxels, ~1 mm each) of GT | predicted abnormal WMH
FIG_DPI = 300                  # journal resolution
MAX_INTACT_MISMATCH = 1e-4     # max voxel fraction where the intact re-prediction may differ
                               # from the saved test prediction (reproducibility check)


# ============================================================
# UTILITIES
# ============================================================

def load_script_module(name, filename):
    """Import a training script without running its main(); hide our CLI flags from it."""
    path = PROJECT_DIR / filename
    saved_argv = sys.argv
    sys.argv = [str(path)]
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.argv = saved_argv
    return module


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")


def finite(values):
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def bootstrap_mean_ci(values, n=N_BOOTSTRAP, seed=BOOTSTRAP_SEED):
    arr = finite(values)
    if arr.size == 0:
        return math.nan, math.nan, math.nan, 0
    rng = np.random.default_rng(seed)
    means = arr[rng.integers(0, arr.size, size=(n, arr.size))].mean(axis=1)
    return float(arr.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), int(arr.size)


def holm(pvals):
    """Holm step-down adjusted p-values; NaN entries stay NaN and are not counted."""
    p = np.asarray(pvals, dtype=float)
    out = np.full(p.shape, np.nan)
    idx = np.where(np.isfinite(p))[0]
    m = len(idx)
    running = 0.0
    for rank, i in enumerate(idx[np.argsort(p[idx])]):
        running = max(running, min(1.0, (m - rank) * p[i]))
        out[i] = running
    return out.tolist()


def paired_wilcoxon(diffs):
    d = finite(diffs)
    if d.size < 5 or not np.any(d != 0):
        return math.nan
    return float(wilcoxon(d).pvalue)


def rankdata(x):
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(1, len(x) + 1)
    for v in np.unique(x):
        m = x == v
        ranks[m] = ranks[m].mean()
    return ranks


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return math.nan
    ra, rb = rankdata(a[ok]), rankdata(b[ok])
    return float(np.corrcoef(ra, rb)[0, 1])


def dice(pred_mask, gt_mask):
    gt_n = int(gt_mask.sum())
    if gt_n == 0:
        return math.nan
    tp = int((pred_mask & gt_mask).sum())
    return 2.0 * tp / (int(pred_mask.sum()) + gt_n)


def sensitivity(pred_mask, gt_mask):
    gt_n = int(gt_mask.sum())
    return int((pred_mask & gt_mask).sum()) / gt_n if gt_n else math.nan


def precision(pred_mask, gt_mask):
    pr_n = int(pred_mask.sum())
    return int((pred_mask & gt_mask).sum()) / pr_n if pr_n else math.nan


def contour_if_any(ax, mask, color):
    if mask.any():
        ax.contour(np.rot90(mask), levels=[0.5], colors=color, linewidths=0.8)


_MM_CACHE = {}


def distance_mm(pid):
    if pid not in _MM_CACHE:
        _MM_CACHE[pid] = np.asanyarray(
            nib.load(str(DISTANCE_MM_ROOT / pid / f"{pid}_VentDist_mm.nii.gz")).dataobj).astype(np.float32)
    return _MM_CACHE[pid]


def geometry_rule(d_mm):
    """Label the 10 mm rule would give an abnormal voxel at distance d."""
    return np.where(d_mm <= PV_THRESHOLD_MM, PV_LABEL, DEEP_LABEL).astype(np.uint8)


def example_patients(mod, ids, n):
    """Pre-specified, label-based choice: patients at GT abnormal-burden quantiles."""
    burden = sorted(ids, key=lambda p: int(np.isin(mod.get_patient_volumes(p)["label"], ABNORMAL).sum()))
    qs = [(2 * k + 1) / (2 * n) for k in range(n)]
    picks = []
    for q in qs:
        p = burden[min(len(burden) - 1, int(q * len(burden)))]
        if p not in picks:
            picks.append(p)
    return picks


def load_saved_prediction(info, pid):
    path = info["pred_dir"] / f"{pid}_pred.nii.gz"
    if not path.exists():
        raise FileNotFoundError(f"missing saved test prediction {path}")
    return np.asanyarray(nib.load(str(path)).dataobj).astype(np.uint8)


# ============================================================
# MODEL / DATA
# ============================================================

class ModelBundle:
    def __init__(self, name, filename):
        self.name = name
        self.mod = load_script_module(f"m_{name}", filename)
        mod = self.mod
        if not mod.TEST_METRICS_JSON.exists():
            raise RuntimeError(f"{name}: final test results not found ({mod.TEST_METRICS_JSON}). "
                               "Explainability is run only on final, already evaluated models.")
        mod.seed_everything(mod.RANDOM_SEED)
        self.device = mod.DEVICE
        self.use_amp = mod.USE_AMP
        self.use_tta = bool(getattr(mod, "USE_TTA", True))
        self.min_voxels = int(json.loads(mod.POSTPROC_JSON.read_text(encoding="utf-8"))["min_lesion_voxels"])

        self.model = mod.build_model()
        ckpt = torch.load(mod.FINAL_CKPT_DIR / "final70_model.pth", map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.test_ids = mod.read_patient_ids(mod.SPLIT_ROOT / "test_30_patients.csv")
        dataset_cls = getattr(mod, "DistanceSliceDataset", None) or getattr(mod, "BaselineSliceDataset")
        self.dataset = dataset_cls([mod.SPLIT_ROOT / "test"], self.test_ids, augment=False)
        self.loader = mod.make_loader(self.dataset, shuffle=False, seed=mod.RANDOM_SEED)
        self.sample_index = {(pid, z): i for i, (_, pid, z) in enumerate(self.dataset.samples)}

        ch = list(mod.CHANNEL_ORDER)
        self.channels = ch
        self.has_dist = "DIST" in ch
        self.dist_idx = ch.index("DIST") if self.has_dist else None
        self.modality_idx = {m: [i for i, n in enumerate(ch) if n.startswith(f"{m}_z")] for m in mod.MRI_MODALITIES}
        self.context_map = []   # (context channel index, matching centre channel index)
        for m in mod.MRI_MODALITIES:
            centre = ch.index(f"{m}_z+0")
            for i, n in enumerate(ch):
                if n.startswith(f"{m}_z") and i != centre:
                    self.context_map.append((i, centre))
        self.groups = {}
        for m in mod.MRI_MODALITIES:
            centre = ch.index(f"{m}_z+0")
            self.groups[f"{m}_centre"] = [centre]
            self.groups[f"{m}_context"] = [i for i in self.modality_idx[m] if i != centre]
        if self.has_dist:
            self.groups["DIST"] = [self.dist_idx]
        print(f"[{name}] {len(self.test_ids)} test patients, {len(self.dataset)} slices, "
              f"channels {ch}, TTA={self.use_tta}, post-proc min {self.min_voxels} voxels")

    def volumes(self, pid):
        return self.mod.get_patient_volumes(pid)

    def release(self):
        del self.model
        torch.cuda.empty_cache()


# ============================================================
# PREDICTION (TTA exactly as in scripts 14/17) + ENTROPY
# ============================================================

FLIPS = [(), (3,), (2,), (2, 3)]


@torch.no_grad()
def predict_condition(b, transform=None, want_entropy=False, loader=None, ids=None):
    """Returns post-processed predictions and, if want_entropy, a dict of
    uncertainty volumes: entropy (total), mutual_info and disagreement (TTA)."""
    loader = b.loader if loader is None else loader
    ids = b.test_ids if ids is None else ids
    shape = {pid: b.volumes(pid)["label"].shape for pid in ids}
    preds = {pid: np.zeros(shape[pid], np.uint8) for pid in ids}
    unc = ({k: {pid: np.zeros(shape[pid], np.float16) for pid in ids}
            for k in ("entropy", "mutual_info", "disagreement")} if want_entropy else None)
    flips = FLIPS if b.use_tta else FLIPS[:1]
    n_classes = b.mod.NUM_CLASSES
    log_c = math.log(n_classes)

    for batch in loader:
        image = batch["image"].to(b.device, non_blocking=True)
        pids = list(batch["patient_id"])
        zs = batch["slice_index"]
        zs = zs.cpu().numpy().tolist() if torch.is_tensor(zs) else list(zs)
        if transform is not None:
            image = transform(image, pids, zs)
        probs = None
        flip_probs = []
        for dims in flips:
            x = torch.flip(image, dims=dims) if dims else image
            with torch.autocast(device_type=b.device.type, enabled=b.use_amp):
                logits = b.model(x)
            p = torch.softmax(logits.float(), dim=1)
            p = torch.flip(p, dims=dims) if dims else p
            probs = p if probs is None else probs + p
            if want_entropy:
                flip_probs.append(p)
        probs = probs / len(flips)
        pred_t = torch.argmax(probs, dim=1)
        pred = pred_t.cpu().numpy().astype(np.uint8)
        if want_entropy:
            ent_t = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=1)
            mean_flip_ent = torch.stack([-(q * torch.log(q.clamp_min(1e-8))).sum(dim=1) for q in flip_probs]).mean(0)
            mi_t = (ent_t - mean_flip_ent).clamp_min(0.0)
            dis_t = torch.stack([(torch.argmax(q, dim=1) != pred_t).float() for q in flip_probs]).mean(0)
            vals = {"entropy": (ent_t / log_c), "mutual_info": (mi_t / log_c), "disagreement": dis_t}
            vals = {k: v.cpu().numpy().astype(np.float16) for k, v in vals.items()}
        for i, pid in enumerate(pids):
            preds[pid][:, :, int(zs[i])] = pred[i]
            if want_entropy:
                for k in unc:
                    unc[k][pid][:, :, int(zs[i])] = vals[k][i]

    preds = {pid: b.mod.remove_small_components(p, b.min_voxels) for pid, p in preds.items()}
    return preds, unc


def compare_with_saved(b, preds):
    """Reproducibility: intact re-prediction must match the saved test predictions."""
    rows = []
    for pid, pred in preds.items():
        saved = np.asanyarray(nib.load(str(b.mod.FINAL_LOG_DIR / "predictions_nifti" / f"{pid}_pred.nii.gz"))
                              .dataobj).astype(np.uint8)
        if saved.shape != pred.shape:
            raise RuntimeError(f"{b.name} {pid}: saved prediction shape {saved.shape} != {pred.shape}")
        rows.append({"model": b.name, "patient_id": pid, "mismatch_fraction": float((saved != pred).mean())})
    worst = max(r["mismatch_fraction"] for r in rows)
    print(f"  [{b.name}] intact vs saved test predictions: worst voxel mismatch {worst:.2e}")
    if worst > MAX_INTACT_MISMATCH:
        raise RuntimeError(f"{b.name}: intact predictions differ from the saved test predictions "
                           f"(worst {worst:.2e} > {MAX_INTACT_MISMATCH}); inference is not identical.")
    return rows


# ============================================================
# 1. CHANNEL ABLATION
# ============================================================

# ------------------------------------------------------------
# ANATOMICALLY CONTROLLED COUNTERFACTUAL DISTANCE-MAP PERTURBATION
# (replaces random swapping; NOT a clinically validated anatomical replacement)
#
# Space check: script 10 resamples every patient onto a grid built from that
# patient's OWN FLAIR (create_reference_grid(flair)); registration was only
# intra-patient (T1/T2/masks -> FLAIR). Patients are therefore NOT in a common
# anatomical space, and equal array shapes do not imply alignment. Every
# donor mask is registered to the target before use.
#
#   1. donor candidates = development patients ranked by ventricle similarity
#      (volume, centroid relative to the brain centroid, spread per axis; mm,
#      world coordinates from each image's affine)
#   2. registration: donor brain -> target brain, axis-aligned affine in world
#      mm (brain centroid translation + per-axis scaling by brain spread);
#      the donor ventricle mask is resampled with nearest neighbour
#   3. plausibility / registration QC (any failure -> next candidate):
#      brain Dice after registration, scale range, ventricle volume ratio,
#      fraction of transformed ventricle inside the target brain, centroid
#      inside the brain; voxels outside the target brain are removed
#   4. distance map recomputed from the transformed mask exactly as in
#      15a_make_ventricle_distance.py (3D EDT in mm, clip at D_max, / D_max)
#   5. affected voxels = target-brain voxels (outside both ventricle masks)
#      whose 10 mm PV/deep assignment differs between original and
#      counterfactual distance; predictions are compared there and overall
#   No suitable donor after MAX_DONOR_CANDIDATES -> swap skipped, reason saved.
# ------------------------------------------------------------

MAX_DONOR_CANDIDATES = 5          # nearest donors tried in order
CF_MIN_BRAIN_DICE = 0.85          # donor brain vs target brain after registration
CF_SCALE_RANGE = (0.75, 1.33)     # per-axis scaling allowed
CF_VENT_VOLUME_RATIO = (1 / 3, 3.0)  # transformed donor / target ventricle volume
CF_MIN_INSIDE_BRAIN = 0.95        # fraction of transformed ventricle voxels inside the target brain

_ANAT_CACHE = {}


def anatomy(pid):
    """Expert ventricle mask, brain mask (non-zero FLAIR), spacing and affine on the patient's own FLAIR grid."""
    if pid not in _ANAT_CACHE:
        vent_img = nib.load(str(PREPROCESSED_ROOT / pid / f"{pid}_Vent_Mask.nii.gz"))
        flair_img = nib.load(str(PREPROCESSED_ROOT / pid / f"{pid}_FLAIR.nii.gz"))
        vent = np.asanyarray(vent_img.dataobj) > 0
        brain = np.asanyarray(flair_img.dataobj) != 0
        if vent.shape != brain.shape or not np.allclose(vent_img.affine, flair_img.affine, atol=1e-4):
            raise RuntimeError(f"{pid}: ventricle mask and FLAIR grids differ")
        if not vent.any():
            raise RuntimeError(f"{pid}: empty ventricle mask")
        _ANAT_CACHE[pid] = {"vent": vent, "brain": brain, "affine": flair_img.affine.astype(np.float64),
                            "spacing": np.array(flair_img.header.get_zooms()[:3], dtype=np.float64)}
    return _ANAT_CACHE[pid]


def world_points(mask, affine):
    """Voxel indices of a mask -> world (scanner) coordinates in mm, honouring orientation."""
    ijk = np.argwhere(mask).astype(np.float64)
    return ijk @ affine[:3, :3].T + affine[:3, 3]


def ventricle_features(pid):
    a = anatomy(pid)
    v, br = world_points(a["vent"], a["affine"]), world_points(a["brain"], a["affine"])
    volume_ml = len(v) * float(np.prod(a["spacing"])) / 1000.0
    return np.concatenate([[volume_ml], v.mean(0) - br.mean(0), v.std(0)])


VENT_FEATURE_NAMES = ["vent_volume_ml", "vent_centroid_rel_brain_x_mm", "vent_centroid_rel_brain_y_mm",
                      "vent_centroid_rel_brain_z_mm", "vent_spread_x_mm", "vent_spread_y_mm", "vent_spread_z_mm"]


def register_donor_to_target(donor, target):
    """Axis-aligned affine registration in world mm: donor brain centroid -> target
    brain centroid, per-axis scaling = target / donor brain spread.
    Returns (vent, brain, scale): the donor ventricle mask and donor brain mask, both
    resampled with nearest neighbour on the target grid, and the per-axis scale.
    Registration QC is computed separately in plausibility_check().
    Equal array shapes are never assumed to be aligned."""
    ad, at = anatomy(donor), anatomy(target)
    bd, bt = world_points(ad["brain"], ad["affine"]), world_points(at["brain"], at["affine"])
    cd, ct = bd.mean(0), bt.mean(0)
    scale = bt.std(0) / bd.std(0)
    shape = at["vent"].shape
    ijk_t = np.indices(shape).reshape(3, -1).T.astype(np.float64)
    world_t = ijk_t @ at["affine"][:3, :3].T + at["affine"][:3, 3]
    world_d = cd + (world_t - ct) / scale                       # target world -> donor world
    inv_d = np.linalg.inv(ad["affine"])
    ijk_d = (world_d @ inv_d[:3, :3].T + inv_d[:3, 3]).T         # donor world -> donor voxel
    vent = map_coordinates(ad["vent"].astype(np.uint8), ijk_d, order=0, mode="constant", cval=0).reshape(shape) > 0
    brain = map_coordinates(ad["brain"].astype(np.uint8), ijk_d, order=0, mode="constant", cval=0).reshape(shape) > 0
    return vent, brain, scale


def plausibility_check(vent_t, brain_t, scale, target):
    """Registration / anatomical plausibility of the transformed donor ventricles.
    Returns (ok, reason, qc)."""
    at = anatomy(target)
    voxel_ml = float(np.prod(at["spacing"])) / 1000.0
    qc = {"brain_dice_after_registration": dice(brain_t, at["brain"]),
          "scale_x": float(scale[0]), "scale_y": float(scale[1]), "scale_z": float(scale[2])}
    if not vent_t.any():
        return False, "transformed donor ventricle mask is empty", qc
    inside = float((vent_t & at["brain"]).sum() / vent_t.sum())
    ratio = float(vent_t.sum() / at["vent"].sum())
    centroid = np.round(np.argwhere(vent_t).mean(0)).astype(int)
    qc.update({"fraction_vent_inside_target_brain": inside, "vent_volume_ratio_donor_to_target": ratio,
               "transformed_vent_volume_ml": float(vent_t.sum() * voxel_ml),
               "target_vent_volume_ml": float(at["vent"].sum() * voxel_ml),
               "transformed_vent_dice_vs_target_vent": dice(vent_t, at["vent"])})
    if not np.isfinite(qc["brain_dice_after_registration"]) or qc["brain_dice_after_registration"] < CF_MIN_BRAIN_DICE:
        return False, f"registration failed: brain Dice {qc['brain_dice_after_registration']:.3f} < {CF_MIN_BRAIN_DICE}", qc
    if np.any(scale < CF_SCALE_RANGE[0]) or np.any(scale > CF_SCALE_RANGE[1]):
        return False, f"implausible scaling {np.round(scale, 3).tolist()} outside {CF_SCALE_RANGE}", qc
    if not (CF_VENT_VOLUME_RATIO[0] <= ratio <= CF_VENT_VOLUME_RATIO[1]):
        return False, f"ventricle volume ratio {ratio:.2f} outside {CF_VENT_VOLUME_RATIO}", qc
    if inside < CF_MIN_INSIDE_BRAIN:
        return False, f"only {inside:.1%} of transformed ventricle inside target brain (< {CF_MIN_INSIDE_BRAIN:.0%})", qc
    if not at["brain"][tuple(centroid)]:
        return False, "transformed ventricle centroid lies outside the target brain", qc
    return True, "", qc


def distance_maps_from_mask(vent, spacing, d_max):
    """Identical to 15a_make_ventricle_distance.py: 3D EDT in mm, 0 inside, clip at D_max, / D_max."""
    d_mm = distance_transform_edt(~vent, sampling=tuple(float(s) for s in spacing)).astype(np.float32)
    d_norm = (np.clip(d_mm, 0, d_max) / d_max).astype(np.float32)
    return d_mm, d_norm


def load_d_max():
    return float(json.loads((DISTANCE_MM_ROOT / "distance_config.json").read_text(encoding="utf-8"))["D_max_mm"])


def rank_donors(target_ids, donor_ids):
    """Development patients ranked by distance in z-scored ventricle-feature space (fixed rule)."""
    feats = {p: ventricle_features(p) for p in sorted(set(target_ids) | set(donor_ids))}
    sd = np.stack([feats[p] for p in donor_ids]).std(0) + 1e-9   # per-feature scale from the donor pool
    ranked = {}
    for t in target_ids:
        dist = {d: float(np.linalg.norm((feats[d] - feats[t]) / sd)) for d in donor_ids if d != t}
        ranked[t] = sorted(dist.items(), key=lambda kv: kv[1])
    return ranked, feats


def build_counterfactuals(b, target_ids):
    """Returns (counterfactuals, skipped). counterfactuals[target] holds the
    counterfactual distance maps, the transformed donor mask, the affected-voxel
    mask and QC; skipped lists targets without a suitable donor and why."""
    if not b.has_dist:
        return {}, []
    d_max = load_d_max()
    donor_pool = b.mod.read_patient_ids(b.mod.SPLIT_ROOT / "train_70_patients.csv")
    if set(donor_pool) & set(b.test_ids):
        raise RuntimeError("Donor pool overlaps the test set.")
    ranked, feats = rank_donors(target_ids, donor_pool)
    cfs, skipped = {}, []
    for t in target_ids:
        at = anatomy(t)
        attempts = []
        chosen = None
        for rank, (donor, feat_dist) in enumerate(ranked[t][:MAX_DONOR_CANDIDATES], start=1):
            try:
                vent_t, brain_t, scale = register_donor_to_target(donor, t)
                ok, reason, reg_qc = plausibility_check(vent_t, brain_t, scale, t)
            except Exception as exc:                       # registration failure: report, try next
                ok, reason, reg_qc = False, f"registration error: {exc}", {}
            attempts.append(f"{donor}: {'OK' if ok else reason}")
            if ok:
                chosen = (rank, donor, feat_dist, vent_t & at["brain"], reg_qc)
                break
        if chosen is None:
            skipped.append({"target": t, "reason": "no suitable donor among the "
                            f"{MAX_DONOR_CANDIDATES} most similar development patients",
                            "attempts": " | ".join(attempts)})
            print(f"  SKIP {t}: no suitable donor ({' | '.join(attempts)})")
            continue
        rank, donor, feat_dist, vent_cf, reg_qc = chosen
        d_mm, d_norm = distance_maps_from_mask(vent_cf, at["spacing"], d_max)
        true_mm = distance_mm(t)
        brain = at["brain"]
        # affected voxels: brain, outside both ventricle masks, 10 mm assignment changes
        assessable = brain & ~at["vent"] & ~vent_cf
        affected = assessable & (geometry_rule(d_mm) != geometry_rule(true_mm))
        gt_ab = np.isin(b.volumes(t)["label"], ABNORMAL)
        qc = {"target": t, "donor": donor, "donor_rank": rank, "feature_distance_z": feat_dist,
              "attempts": " | ".join(attempts), **reg_qc,
              "mean_abs_distance_change_mm_in_brain": float(np.abs(d_mm - true_mm)[brain].mean()),
              "n_assessable_brain_voxels": int(assessable.sum()),
              "n_affected_voxels": int(affected.sum()),
              "pct_affected_voxels": 100.0 * float(affected.sum()) / max(int(assessable.sum()), 1),
              "n_affected_pv_to_deep": int((affected & (true_mm <= PV_THRESHOLD_MM)).sum()),
              "n_affected_deep_to_pv": int((affected & (true_mm > PV_THRESHOLD_MM)).sum()),
              "n_gt_abnormal_voxels": int(gt_ab.sum()),
              "n_gt_abnormal_affected": int((affected & gt_ab).sum()),
              "pct_gt_abnormal_affected": (100.0 * float((affected & gt_ab).sum()) / int(gt_ab.sum())
                                           if gt_ab.any() else math.nan)}
        for k, name in enumerate(VENT_FEATURE_NAMES):
            qc[f"target_{name}"] = float(feats[t][k])
            qc[f"donor_{name}"] = float(feats[donor][k])
        cfs[t] = {"norm": d_norm, "mm": d_mm, "aligned_vent": vent_cf, "affected": affected,
                  "assessable": assessable, "qc": qc}
    return cfs, skipped


def make_conditions(b, counterfactuals):
    def zero_modality(m):
        idx = b.modality_idx[m]

        def t(image, pids, zs):
            image = image.clone()
            image[:, idx] = 0.0
            return image
        return t

    def no_neighbours(image, pids, zs):
        image = image.clone()
        for ctx, centre in b.context_map:
            image[:, ctx] = image[:, centre]
        return image

    conds = {
        "intact": None,
        "no_FLAIR": zero_modality("FLAIR"),
        "no_T2": zero_modality("T2"),
        "no_T1": zero_modality("T1"),
        "no_neighbour_slices": no_neighbours,
    }
    if b.has_dist:
        def dist_zero(image, pids, zs):
            image = image.clone()
            image[:, b.dist_idx] = 0.0
            return image

        def dist_counterfactual(image, pids, zs):
            """Anatomically controlled counterfactual distance map (skipped targets keep their own map,
            and are excluded from the dist_counterfactual results)."""
            image = image.clone()
            for i, (pid, z) in enumerate(zip(pids, zs)):
                cf = counterfactuals.get(pid)
                if cf is None:
                    continue
                d = cf["norm"][:, :, int(z)]
                image[i, b.dist_idx] = torch.from_numpy(np.ascontiguousarray(d)).to(image.device)
            return image

        conds["dist_zero"] = dist_zero
        conds["dist_counterfactual"] = dist_counterfactual
    return conds


# conditions that receive the counterfactual geometry for evaluation:
#   intact               -> reference: how well do ORIGINAL predictions match the counterfactual geometry
#   dist_counterfactual  -> the counterfactual itself
# (dist_zero never saw the counterfactual map, so it must not get cf_* metrics)
CF_EVAL_CONDITIONS = ("intact", "dist_counterfactual")


def ablation_metrics(b, pred, pid, cf=None, intact_pred=None):
    alt_mm = cf["mm"] if cf is not None else None
    vols = b.volumes(pid)
    gt = vols["label"]
    row = {}
    for name in ABLATION_TARGETS:
        labels = b.mod.EVAL_TARGETS[name]
        pm, gm = np.isin(pred, labels), np.isin(gt, labels)
        row[f"dice_{name}"] = dice(pm, gm)
        row[f"sens_{name}"] = sensitivity(pm, gm)
        row[f"prec_{name}"] = precision(pm, gm)
    ab_gt, ab_pr = np.isin(gt, ABNORMAL), np.isin(pred, ABNORMAL)
    both = ab_gt & ab_pr
    # among correctly detected abnormal voxels: is PV/deep label right?
    row["pvdeep_label_agreement"] = float((pred[both] == gt[both]).mean()) if both.any() else math.nan
    # among all predicted abnormal voxels: does PV/deep follow the TRUE / COUNTERFACTUAL geometry?
    true_rule = geometry_rule(distance_mm(pid))
    alt_rule = geometry_rule(alt_mm) if alt_mm is not None else None
    if ab_pr.any():
        row["pred_matches_true_geometry"] = float((pred[ab_pr] == true_rule[ab_pr]).mean())
        row["pred_pv_fraction"] = float((pred[ab_pr] == PV_LABEL).mean())
        if alt_rule is not None:
            row["pred_matches_counterfactual_geometry"] = float((pred[ab_pr] == alt_rule[ab_pr]).mean())
    else:
        row["pred_matches_true_geometry"] = math.nan
        row["pred_pv_fraction"] = math.nan
    row["pred_abnormal_voxels"] = int(ab_pr.sum())

    # direct comparison with the original (intact) prediction of the same patient
    if intact_pred is not None:
        ab_in = np.isin(intact_pred, ABNORMAL)
        both_ab = ab_in & ab_pr
        row["cf_voxel_change_fraction"] = float((pred != intact_pred).mean())
        row["cf_abnormal_mask_dice_vs_intact"] = dice(ab_pr, ab_in)
        row["cf_pvdeep_flip_fraction"] = float((pred[both_ab] != intact_pred[both_ab]).mean()) if both_ab.any() else math.nan
        if cf is not None:
            affected, assessable = cf["affected"], cf["assessable"]
            unaffected = assessable & ~affected
            changed_pred = pred != intact_pred
            # any prediction change inside vs outside the voxels whose PV/deep assignment changed
            row["cf_n_affected_voxels"] = int(affected.sum())
            row["cf_pred_change_in_affected"] = float(changed_pred[affected].mean()) if affected.any() else math.nan
            row["cf_pred_change_in_unaffected"] = float(changed_pred[unaffected].mean()) if unaffected.any() else math.nan
            # affected voxels predicted abnormal in both runs
            aff_ab = affected & both_ab
            row["cf_n_affected_abnormal_both_runs"] = int(aff_ab.sum())
            # AGREEMENT with the new geometry (includes labels that were already consistent with it)
            row["cf_follows_geometry_change"] = float((pred[aff_ab] == alt_rule[aff_ab]).mean()) if aff_ab.any() else math.nan
            if aff_ab.any():
                # label changed between the two runs
                row["cf_switch_fraction"] = float((pred[aff_ab] != intact_pred[aff_ab]).mean())
                # label switched FROM the true-geometry label TO the counterfactual-geometry label
                # (on affected voxels true_rule != alt_rule, so this always implies a switch)
                row["cf_switch_consistent"] = float(((intact_pred[aff_ab] == true_rule[aff_ab]) &
                                                     (pred[aff_ab] == alt_rule[aff_ab])).mean())
                # how many affected voxels were consistent with the true geometry before the swap
                row["cf_intact_matched_true_geometry"] = float((intact_pred[aff_ab] == true_rule[aff_ab]).mean())
            else:
                row["cf_switch_fraction"] = row["cf_switch_consistent"] = math.nan
                row["cf_intact_matched_true_geometry"] = math.nan
            unaff_ab = unaffected & both_ab
            row["cf_stable_where_geometry_unchanged"] = (
                float((pred[unaff_ab] == intact_pred[unaff_ab]).mean()) if unaff_ab.any() else math.nan)
            # lesion detection (abnormal vs not) inside affected voxels: should NOT depend on the map
            row["cf_detection_change_in_affected"] = (
                float((ab_pr[affected] != ab_in[affected]).mean()) if affected.any() else math.nan)
    return row


def run_ablation(b, out_dir):
    counterfactuals, skipped = build_counterfactuals(b, sorted(b.test_ids))
    conds = make_conditions(b, counterfactuals)     # "intact" is always first
    rows = []
    intact_entropy, intact_preds, repro_rows, cf_preds = None, None, [], {}
    for cond, transform in conds.items():
        t0 = time.time()
        want_entropy = cond == "intact"
        preds, entropy = predict_condition(b, transform, want_entropy=want_entropy)
        if cond == "intact":
            intact_preds, intact_entropy = preds, entropy
            repro_rows = compare_with_saved(b, preds)
        if cond == "dist_counterfactual":
            cf_preds = preds
        for pid in sorted(b.test_ids):
            cf = counterfactuals.get(pid)
            if cond == "dist_counterfactual" and cf is None:
                continue                                   # skipped target, reason in counterfactual_skipped.csv
            use_cf = cf if cond in CF_EVAL_CONDITIONS else None
            r = {"model": b.name, "condition": cond, "patient_id": pid,
                 "donor": cf["qc"]["donor"] if (cf is not None and cond == "dist_counterfactual") else ""}
            r.update(ablation_metrics(b, preds[pid], pid, use_cf,
                                      intact_pred=None if cond == "intact" else intact_preds[pid]))
            rows.append(r)
        print(f"  [{b.name}] ablation {cond:20s} done ({time.time() - t0:.0f}s)")
    if b.has_dist:
        write_csv(out_dir / "counterfactual_donors.csv", [cf["qc"] for cf in counterfactuals.values()]
                  or [{"note": "no counterfactual could be built"}])
        write_csv(out_dir / "counterfactual_skipped.csv", skipped or [{"target": "", "reason": "none skipped",
                                                                         "attempts": ""}])
        print(f"  [{b.name}] counterfactual: {len(counterfactuals)} built, {len(skipped)} skipped")
    return rows, intact_preds, intact_entropy, counterfactuals, cf_preds, repro_rows


def plot_counterfactual_examples(b, counterfactuals, intact_preds, cf_preds, examples, out_dir):
    """Target vs transformed donor ventricles, original vs counterfactual distance,
    affected voxels, and both predictions (anatomically controlled counterfactual)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    d_max = load_d_max()
    for pid in examples:
        cf = counterfactuals.get(pid)
        if cf is None or pid not in cf_preds:
            continue
        z = ig_slice_for_patient(b, pid)
        flair = b.mod.load_flair(pid)
        at = anatomy(pid)
        d_true = np.clip(distance_mm(pid), 0, d_max)
        d_cf = np.clip(cf["mm"], 0, d_max)
        fig, axes = plt.subplots(1, 6, figsize=(24, 4.6))
        axes[0].imshow(np.rot90(flair[:, :, z]), cmap="gray")
        contour_if_any(axes[0], at["vent"][:, :, z], "cyan")
        contour_if_any(axes[0], cf["aligned_vent"][:, :, z], "orange")
        axes[0].set_title(f"{pid} z={z}\nown ventricles (cyan) vs transformed donor {cf['qc']['donor']} (orange)",
                          fontsize=8)
        for ax, d, title in ((axes[1], d_true, "original distance map"), (axes[2], d_cf, "counterfactual distance map")):
            im = ax.imshow(np.rot90(d[:, :, z]), cmap="viridis", vmin=0, vmax=d_max)
            ax.contour(np.rot90(d[:, :, z]), levels=[PV_THRESHOLD_MM], colors="white", linewidths=0.8)
            ax.set_title(f"{title} (mm, clipped at {d_max:g})\nwhite line = 10 mm PV/deep boundary", fontsize=8)
        fig.colorbar(im, ax=axes[2], fraction=0.046, label="distance to ventricle (mm)")
        axes[3].imshow(np.rot90(flair[:, :, z]), cmap="gray")
        aff = cf["affected"][:, :, z]
        overlay = np.zeros(aff.shape + (4,), np.float32)
        overlay[aff & (distance_mm(pid)[:, :, z] <= PV_THRESHOLD_MM)] = (1, 0.5, 0, 0.7)   # PV -> deep
        overlay[aff & (distance_mm(pid)[:, :, z] > PV_THRESHOLD_MM)] = (0, 0.6, 1, 0.7)    # deep -> PV
        axes[3].imshow(np.rot90(overlay))
        axes[3].set_title(f"affected voxels ({cf['qc']['pct_affected_voxels']:.1f}% of brain)\n"
                          "orange PV->deep, blue deep->PV", fontsize=8)
        for ax, pred, title in ((axes[4], intact_preds[pid], "original prediction"),
                                (axes[5], cf_preds[pid], "prediction, counterfactual map")):
            ax.imshow(np.rot90(flair[:, :, z]), cmap="gray")
            contour_if_any(ax, pred[:, :, z] == PV_LABEL, "red")
            contour_if_any(ax, pred[:, :, z] == DEEP_LABEL, "magenta")
            ax.set_title(f"{title}\nPV abnormal (red) / deep abnormal (magenta)", fontsize=8)
        for ax in axes:
            ax.axis("off")
        fig.suptitle(f"{b.name}: anatomically controlled counterfactual distance-map perturbation", fontsize=10)
        fig.tight_layout()
        fig.savefig(out_dir / f"{b.name}_{pid}_counterfactual.png", dpi=FIG_DPI)
        plt.close(fig)


def plot_counterfactual_summary(ablation_rows, counterfactuals, out_dir):
    """Per-patient view of the counterfactual: how much geometry changed and how predictions reacted."""
    rows = [r for r in ablation_rows if r["condition"] == "dist_counterfactual"]
    if not rows:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    pids = [r["patient_id"] for r in rows]
    x = np.arange(len(pids))
    pct_aff = [counterfactuals[p]["qc"]["pct_affected_voxels"] for p in pids]
    fig, axes = plt.subplots(3, 1, figsize=(max(10, 0.45 * len(pids)), 11), sharex=True)
    axes[0].bar(x, pct_aff, color="grey")
    axes[0].set_ylabel("% brain voxels with\nchanged PV/deep assignment")
    axes[0].set_title("Anatomically controlled counterfactual distance-map perturbation (test patients)")
    w = 0.4
    axes[1].bar(x - w / 2, [r.get("cf_pred_change_in_affected", math.nan) for r in rows], w,
                label="affected voxels", color="tab:orange")
    axes[1].bar(x + w / 2, [r.get("cf_pred_change_in_unaffected", math.nan) for r in rows], w,
                label="unaffected voxels", color="tab:blue")
    axes[1].set_ylabel("fraction of voxels whose\npredicted label changed")
    axes[1].legend(fontsize=8)
    axes[2].bar(x - w / 2, [r.get("cf_switch_consistent", math.nan) for r in rows], w,
                label="affected: label switched to match the counterfactual geometry", color="tab:green")
    axes[2].bar(x + w / 2, [r.get("cf_stable_where_geometry_unchanged", math.nan) for r in rows], w,
                label="unaffected: label unchanged", color="tab:purple")
    axes[2].set_ylim(0, 1.05)
    axes[2].set_ylabel("fraction of abnormal voxels\n(predicted abnormal in both runs)")
    axes[2].legend(fontsize=8)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(pids, rotation=90, fontsize=8)
    axes[2].set_xlabel("test patient")
    fig.tight_layout()
    fig.savefig(out_dir / "counterfactual_per_patient.png", dpi=FIG_DPI)
    plt.close(fig)


def summarize_ablation(rows):
    prefixes = ("dice_", "sens_", "prec_", "pvdeep_", "pred_matches", "pred_pv", "cf_")
    metrics = []
    for r in rows:
        for k in r:
            if k.startswith(prefixes) and k not in metrics:
                metrics.append(k)
    summary = []
    for model in sorted({r["model"] for r in rows}):
        intact = {r["patient_id"]: r for r in rows if r["model"] == model and r["condition"] == "intact"}
        for cond in [c for c in dict.fromkeys(r["condition"] for r in rows if r["model"] == model)]:
            sub = [r for r in rows if r["model"] == model and r["condition"] == cond]
            for m in metrics:
                vals = [r.get(m, math.nan) for r in sub]
                mean, lo, hi, n = bootstrap_mean_ci(vals)
                diffs = [r.get(m, math.nan) - intact[r["patient_id"]].get(m, math.nan)
                         for r in sub if r["patient_id"] in intact]
                dmean, dlo, dhi, dn = bootstrap_mean_ci(diffs)
                if n == 0:
                    continue
                summary.append({"model": model, "condition": cond, "metric": m,
                                "mean": mean, "ci95_low": lo, "ci95_high": hi, "n": n,
                                "delta_vs_intact": dmean, "delta_ci95_low": dlo, "delta_ci95_high": dhi,
                                "delta_wilcoxon_p": paired_wilcoxon(diffs) if cond != "intact" else math.nan})
    return summary


def plot_ablation(summary, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    models = sorted({s["model"] for s in summary})
    fig, axes = plt.subplots(1, len(models), figsize=(8 * len(models), 5), squeeze=False)
    for ax, model in zip(axes[0], models):
        conds = list(dict.fromkeys(s["condition"] for s in summary if s["model"] == model))
        targets = [t for t in ABLATION_TARGETS if t != "all_wmh"]
        width = 0.8 / len(conds)
        for k, cond in enumerate(conds):
            vals, err = [], []
            for t in targets:
                s = next((s for s in summary if s["model"] == model and s["condition"] == cond
                          and s["metric"] == f"dice_{t}"), None)
                vals.append(s["mean"] if s else math.nan)
                err.append([s["mean"] - s["ci95_low"], s["ci95_high"] - s["mean"]] if s else [0, 0])
            err = np.array(err).T
            ax.bar(np.arange(len(targets)) + k * width, vals, width, yerr=err, capsize=2, label=cond)
        ax.set_xticks(np.arange(len(targets)) + 0.4 - width / 2)
        ax.set_xticklabels(targets, rotation=15, fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_ylabel("mean test Dice (95% bootstrap CI)")
        ax.set_title(f"{model}: channel ablation")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "ablation_dice.png", dpi=FIG_DPI)
    plt.close(fig)

    # counterfactual geometry plot (distance model)
    dist_rows = [s for s in summary if s["model"] == "distance_unet_v2"
                 and s["metric"] in ("pred_matches_true_geometry", "pred_matches_counterfactual_geometry")
                 and s["condition"] in ("intact", "dist_counterfactual")]
    if dist_rows:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        conds = ["intact", "dist_counterfactual"]
        for k, metric in enumerate(["pred_matches_true_geometry", "pred_matches_counterfactual_geometry"]):
            vals, err = [], []
            for c in conds:
                s = next((s for s in dist_rows if s["condition"] == c and s["metric"] == metric), None)
                vals.append(s["mean"] if s else math.nan)
                err.append([s["mean"] - s["ci95_low"], s["ci95_high"] - s["mean"]] if s else [0, 0])
            ax.bar(np.arange(len(conds)) + k * 0.38, vals, 0.38, yerr=np.array(err).T, capsize=3,
                   label=metric.replace("pred_matches_", "PV/deep follows ").replace("_", " "))
        ax.set_xticks(np.arange(len(conds)) + 0.19)
        ax.set_xticklabels(conds)
        ax.set_ylim(0, 1)
        ax.set_ylabel("fraction of predicted abnormal voxels")
        ax.set_title("Distance U-Net: does PV/deep follow the distance input?")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "distance_counterfactual.png", dpi=FIG_DPI)
        plt.close(fig)


# ============================================================
# 2. PERFORMANCE BY DISTANCE TO THE VENTRICLES (saved predictions)
# ============================================================

def band_mask(d, lo, hi):
    return (d <= hi) if lo == 0.0 else ((d > lo) & (d <= hi))


def band_label(lo, hi):
    return f"{lo:g}-{hi:g} mm" if np.isfinite(hi) else f">{lo:g} mm"


def run_distance_bands(bundles_info, out_dir):
    rows, patient_rows, label_acc_rows = [], [], []
    for name, info in bundles_info.items():
        get_vols = info["get_volumes"]
        totals = {}
        for pid in info["test_ids"]:
            pred = load_saved_prediction(info, pid)
            vols = get_vols(pid)
            gt = vols["label"]
            voxel_ml = float(np.prod(vols["spacing"])) / 1000.0
            d = distance_mm(pid)
            valid = gt != 1            # exclude ground-truth ventricle voxels
            for lo, hi in DISTANCE_BANDS_MM:
                band = band_mask(d, lo, hi) & valid
                for tname, labels in BAND_TARGETS.items():
                    g = np.isin(gt, labels) & band
                    p = np.isin(pred, labels) & band
                    tp, fp, fn = int((g & p).sum()), int((p & ~g).sum()), int((g & ~p).sum())
                    key = (tname, lo, hi)
                    t = totals.setdefault(key, {"tp": 0, "fp": 0, "fn": 0, "fp_ml": []})
                    t["tp"] += tp
                    t["fp"] += fp
                    t["fn"] += fn
                    t["fp_ml"].append(fp * voxel_ml)
                    patient_rows.append({
                        "model": name, "patient_id": pid, "target": tname, "band": band_label(lo, hi),
                        "gt_voxels": tp + fn, "sensitivity": tp / (tp + fn) if tp + fn else math.nan,
                        "precision": tp / (tp + fp) if tp + fp else math.nan,
                        "fp_volume_ml": fp * voxel_ml,
                    })
                # PV/deep LOCATION: among correctly detected abnormal voxels, is the PV/deep label right?
                both = np.isin(gt, ABNORMAL) & np.isin(pred, ABNORMAL) & band
                label_acc_rows.append({
                    "model": name, "patient_id": pid, "band": band_label(lo, hi),
                    "n_detected_abnormal_voxels": int(both.sum()),
                    "pvdeep_label_accuracy": float((pred[both] == gt[both]).mean()) if both.any() else math.nan,
                })
        for (tname, lo, hi), t in totals.items():
            tp, fp, fn = t["tp"], t["fp"], t["fn"]
            sens = [r["sensitivity"] for r in patient_rows
                    if r["model"] == name and r["target"] == tname and r["band"] == band_label(lo, hi)]
            m, l, h, n = bootstrap_mean_ci(sens)
            rows.append({
                "model": name, "target": tname, "band": band_label(lo, hi), "band_low_mm": lo, "band_high_mm": hi,
                "pooled_sensitivity": tp / (tp + fn) if tp + fn else math.nan,
                "pooled_precision": tp / (tp + fp) if tp + fp else math.nan,
                "pooled_dice": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else math.nan,
                "mean_patient_sensitivity": m, "sensitivity_ci95_low": l, "sensitivity_ci95_high": h,
                "n_patients_with_gt": n,
                "mean_fp_volume_ml_per_patient": float(np.mean(t["fp_ml"])),
                "gt_voxels_total": tp + fn,
            })
    write_csv(out_dir / "distance_band_metrics.csv", rows)
    write_csv(out_dir / "distance_band_per_patient.csv", patient_rows)
    write_csv(out_dir / "pvdeep_label_accuracy_per_patient.csv", label_acc_rows)

    labels = [band_label(lo, hi) for lo, hi in DISTANCE_BANDS_MM]

    # ---- PV/deep label accuracy per band: per model, and paired distance - baseline ----
    acc_summary = []
    for name in bundles_info:
        for b_lab in labels:
            vals = [r["pvdeep_label_accuracy"] for r in label_acc_rows if r["model"] == name and r["band"] == b_lab]
            m, l, h, n = bootstrap_mean_ci(vals)
            acc_summary.append({"model": name, "band": b_lab, "mean_label_accuracy": m,
                                "ci95_low": l, "ci95_high": h, "n_patients": n})
    if {"baseline_unet_v2", "distance_unet_v2"} <= set(bundles_info):
        pvals, diff_rows = [], []
        for b_lab in labels:
            base = {r["patient_id"]: r["pvdeep_label_accuracy"] for r in label_acc_rows
                    if r["model"] == "baseline_unet_v2" and r["band"] == b_lab}
            dist = {r["patient_id"]: r["pvdeep_label_accuracy"] for r in label_acc_rows
                    if r["model"] == "distance_unet_v2" and r["band"] == b_lab}
            diffs = [dist[p] - base[p] for p in dist if p in base]
            m, l, h, n = bootstrap_mean_ci(diffs)
            p = paired_wilcoxon(diffs)
            pvals.append(p)
            diff_rows.append({"model": "distance_minus_baseline", "band": b_lab, "mean_label_accuracy": m,
                              "ci95_low": l, "ci95_high": h, "n_patients": n, "wilcoxon_p": p})
        for r, ph in zip(diff_rows, holm(pvals)):
            r["wilcoxon_p_holm"] = ph
        acc_summary.extend(diff_rows)
    write_csv(out_dir / "pvdeep_label_accuracy_by_band.csv", acc_summary)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name in bundles_info:
        sub = {r["band"]: r for r in acc_summary if r["model"] == name}
        x = np.arange(len(labels))
        y = [sub[b]["mean_label_accuracy"] for b in labels]
        ax.errorbar(x, y, yerr=[[sub[b]["mean_label_accuracy"] - sub[b]["ci95_low"] for b in labels],
                                [sub[b]["ci95_high"] - sub[b]["mean_label_accuracy"] for b in labels]],
                    marker="o", capsize=3, label=name)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.axvline(1.5, color="grey", ls=":", lw=0.8)   # 10 mm PV/deep boundary
    ax.set_xlabel("distance to ventricles")
    ax.set_ylabel("PV/deep label accuracy\n(correctly detected abnormal voxels)")
    ax.set_title("PV/deep assignment by distance to the ventricles (test set)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "pvdeep_label_accuracy_by_distance.png", dpi=FIG_DPI)
    plt.close(fig)

    for tname in BAND_TARGETS:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        for name in bundles_info:
            sub = {r["band"]: r for r in rows if r["model"] == name and r["target"] == tname}
            x = np.arange(len(labels))
            axes[0].errorbar(x, [sub[b]["mean_patient_sensitivity"] for b in labels],
                             yerr=[[sub[b]["mean_patient_sensitivity"] - sub[b]["sensitivity_ci95_low"] for b in labels],
                                   [sub[b]["sensitivity_ci95_high"] - sub[b]["mean_patient_sensitivity"] for b in labels]],
                             marker="o", capsize=3, label=name)
            axes[1].plot(x, [sub[b]["pooled_precision"] for b in labels], marker="o", label=name)
            axes[2].plot(x, [sub[b]["mean_fp_volume_ml_per_patient"] for b in labels], marker="o", label=name)
        for ax, title in zip(axes, ["sensitivity (mean over patients, 95% CI)", "precision (pooled)",
                                    "false-positive volume per patient (ml)"]):
            ax.set_xticks(np.arange(len(labels)))
            ax.set_xticklabels(labels, fontsize=8)
            ax.set_xlabel("distance to ventricles")
            ax.set_title(title, fontsize=10)
            ax.axvline(1.5, color="grey", ls=":", lw=0.8)   # 10 mm PV/deep boundary
            ax.legend(fontsize=8)
        axes[0].set_ylim(0, 1)
        axes[1].set_ylim(0, 1)
        fig.suptitle(f"{tname}: performance by distance to the ventricles (test set)")
        fig.tight_layout()
        fig.savefig(out_dir / f"{tname}_by_distance.png", dpi=FIG_DPI)
        plt.close(fig)


# ============================================================
# 3. INTEGRATED GRADIENTS
# ============================================================

def integrated_gradients(b, x, target_class, region):
    """IG from an all-zero baseline (z-scored MRI: 0 = mean brain intensity and
    background; DIST: 0 = at the ventricle). Score = sum of the softmax
    probability of target_class over the region. Float32, no AMP.
    Note: because the DIST baseline is 0 (= at the ventricle), DIST attribution is
    small by construction for voxels near the ventricles; DIST shares are therefore
    not comparable between PV and deep classes."""
    region_t = torch.from_numpy(region.astype(np.float32)).to(b.device)
    alphas = (torch.arange(IG_STEPS, device=b.device, dtype=torch.float32) + 0.5) / IG_STEPS
    total_grad = torch.zeros_like(x)

    def score_fn(inp):
        prob = torch.softmax(b.model(inp).float(), dim=1)[:, target_class]
        return (prob * region_t).sum(dim=(1, 2))

    with torch.enable_grad():
        for i in range(0, IG_STEPS, IG_CHUNK):
            a = alphas[i:i + IG_CHUNK].view(-1, 1, 1, 1)
            xi = (a * x).detach().requires_grad_(True)
            score = score_fn(xi).sum()
            grad, = torch.autograd.grad(score, xi)
            total_grad += grad.sum(dim=0, keepdim=True)
    attr = (x * total_grad / IG_STEPS)[0]
    with torch.no_grad():
        f_x = float(score_fn(x)[0])
        f_0 = float(score_fn(torch.zeros_like(x))[0])
    delta = f_x - f_0
    attr_sum = float(attr.sum())
    completeness_err = abs(attr_sum - delta) / abs(delta) if abs(delta) > 1e-8 else math.nan
    quality = {
        "score_input": f_x,
        "score_baseline": f_0,
        "score_delta": delta,
        "attr_sum": attr_sum,
        "completeness_rel_error": completeness_err,
        "baseline_uninformative": int(abs(delta) < IG_MIN_SCORE_DELTA),
        "completeness_ok": int(np.isfinite(completeness_err) and completeness_err <= IG_COMPLETENESS_TOL),
    }
    return attr.cpu().numpy(), quality


def ig_slice_for_patient(b, pid):
    """Pre-specified: the dataset slice with the most GT abnormal-WMH voxels."""
    gt = b.volumes(pid)["label"]
    zs = [z for (p, z) in b.sample_index if p == pid]
    counts = {z: int(np.isin(gt[:, :, z], ABNORMAL).sum()) for z in zs}
    return max(counts, key=counts.get)


def run_ig(b, examples):
    """IG per test patient. The explained region is the model's own prediction from a
    single forward pass (no TTA, no post-processing)."""
    rows = []
    example_maps = {}
    for pid in sorted(b.test_ids):
        z = ig_slice_for_patient(b, pid)
        sample = b.dataset[b.sample_index[(pid, z)]]
        x = sample["image"][None].to(b.device).float()
        with torch.no_grad():
            pred = torch.argmax(b.model(x).float(), dim=1)[0].cpu().numpy()
        for cname, cls in IG_CLASSES.items():
            region = pred == cls
            row = {"model": b.name, "patient_id": pid, "slice": z, "target_class": cname,
                   "region_voxels": int(region.sum())}
            if region.sum() < IG_MIN_REGION_VOXELS:
                row["skipped"] = "predicted region too small"
                rows.append(row)
                continue
            attr, quality = integrated_gradients(b, x, cls, region)
            abs_total = float(np.abs(attr).sum()) + 1e-12
            row.update(quality)
            if not quality["completeness_ok"] or quality["baseline_uninformative"]:
                print(f"  WARNING [{b.name}] {pid} {cname}: IG completeness error "
                      f"{quality['completeness_rel_error']:.3f}, delta {quality['score_delta']:.4f}")
            for g, idx in b.groups.items():
                row[f"abs_share_{g}"] = float(np.abs(attr[idx]).sum()) / abs_total
                row[f"signed_sum_{g}"] = float(attr[idx].sum())
            # attribution concentrated inside vs outside the explained region
            inside = float(np.abs(attr[:, region]).sum()) / abs_total
            row["abs_share_inside_region"] = inside
            rows.append(row)
            if pid in examples:
                example_maps[(pid, cname)] = (z, attr, pred)
        print(f"  [{b.name}] IG patient {pid} slice {z} done")
    return rows, example_maps


def plot_ig_summary(ig_rows, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = ["FLAIR_centre", "T2_centre", "T1_centre", "FLAIR_context", "T2_context", "T1_context", "DIST"]
    combos = sorted({(r["model"], r["target_class"]) for r in ig_rows})
    summary = []
    for model, cname in combos:
        sub = [r for r in ig_rows if r["model"] == model and r["target_class"] == cname and "skipped" not in r]
        for g in groups:
            vals = [r.get(f"abs_share_{g}", math.nan) for r in sub]
            mean, lo, hi, n = bootstrap_mean_ci(vals)
            if n:
                summary.append({"model": model, "target_class": cname, "group": g,
                                "mean_abs_share": mean, "ci95_low": lo, "ci95_high": hi, "n": n})
        comp = finite([r.get("completeness_rel_error", math.nan) for r in sub])
        summary.append({"model": model, "target_class": cname, "group": "completeness_rel_error_median",
                        "mean_abs_share": float(np.median(comp)) if comp.size else math.nan, "n": int(comp.size)})
        n_ok = sum(int(r.get("completeness_ok", 0)) for r in sub)
        n_bad_base = sum(int(r.get("baseline_uninformative", 0)) for r in sub)
        summary.append({"model": model, "target_class": cname,
                        "group": f"fraction_completeness_within_{IG_COMPLETENESS_TOL}",
                        "mean_abs_share": n_ok / len(sub) if sub else math.nan, "n": len(sub)})
        summary.append({"model": model, "target_class": cname, "group": "n_baseline_uninformative",
                        "mean_abs_share": n_bad_base, "n": len(sub)})
        n_skipped = sum(1 for r in ig_rows if r["model"] == model and r["target_class"] == cname and "skipped" in r)
        summary.append({"model": model, "target_class": cname, "group": "n_skipped_region_too_small",
                        "mean_abs_share": n_skipped, "n": n_skipped})
    fig, ax = plt.subplots(figsize=(12, 5))
    width = 0.8 / max(1, len(combos))
    for k, (model, cname) in enumerate(combos):
        vals, err = [], []
        for g in groups:
            s = next((s for s in summary if s["model"] == model and s["target_class"] == cname
                      and s["group"] == g), None)
            vals.append(s["mean_abs_share"] if s else 0.0)
            err.append([s["mean_abs_share"] - s["ci95_low"], s["ci95_high"] - s["mean_abs_share"]] if s else [0, 0])
        ax.bar(np.arange(len(groups)) + k * width, vals, width, yerr=np.array(err).T, capsize=2,
               label=f"{model} / {cname}")
    ax.set_xticks(np.arange(len(groups)) + 0.4 - width / 2)
    ax.set_xticklabels(groups, fontsize=8)
    ax.set_ylabel("share of total |IG attribution| (mean, 95% CI)")
    ax.set_title("Integrated Gradients: which inputs drive PV / deep abnormal-WMH predictions\n"
                 "(context groups contain 2 channels; DIST shares not comparable between classes, zero baseline)",
                 fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "ig_channel_shares.png", dpi=FIG_DPI)
    plt.close(fig)
    return summary


def plot_ig_examples(b, example_maps, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    pids = sorted({pid for pid, _ in example_maps})
    for pid in pids:
        classes = [c for c in IG_CLASSES if (pid, c) in example_maps]
        if not classes:
            continue
        flair = b.mod.load_flair(pid)
        gt = b.volumes(pid)["label"]
        panels = ["FLAIR_centre", "T2_centre", "T1_centre", "context (all)"] + (["DIST"] if b.has_dist else [])
        fig, axes = plt.subplots(len(classes), 2 + len(panels), figsize=(3.2 * (2 + len(panels)), 3.4 * len(classes)),
                                 squeeze=False)
        for r, cname in enumerate(classes):
            z, attr, pred = example_maps[(pid, cname)]
            cls = IG_CLASSES[cname]
            a = axes[r]
            a[0].imshow(np.rot90(flair[:, :, z]), cmap="gray")
            contour_if_any(a[0], gt[:, :, z] == cls, "lime")
            a[0].set_title(f"{pid} z={z} FLAIR + GT {cname}", fontsize=8)
            a[1].imshow(np.rot90(flair[:, :, z]), cmap="gray")
            contour_if_any(a[1], pred == cls, "red")
            a[1].set_title("explained region (prediction)", fontsize=8)
            maps = {
                "FLAIR_centre": np.abs(attr[b.groups["FLAIR_centre"]]).sum(0),
                "T2_centre": np.abs(attr[b.groups["T2_centre"]]).sum(0),
                "T1_centre": np.abs(attr[b.groups["T1_centre"]]).sum(0),
                "context (all)": np.abs(attr[sum((b.groups[g] for g in b.groups if g.endswith("_context")), [])]).sum(0),
            }
            if b.has_dist:
                maps["DIST"] = np.abs(attr[b.groups["DIST"]]).sum(0)
            vmax = max(np.percentile(m, 99.5) for m in maps.values()) + 1e-12   # common scale per row
            for k, name in enumerate(panels):
                ax = a[2 + k]
                ax.imshow(np.rot90(flair[:, :, z]), cmap="gray")
                im = ax.imshow(np.rot90(maps[name]), cmap="hot", alpha=0.65, vmin=0, vmax=vmax)
                ax.set_title(f"|IG| {name}", fontsize=8)
            fig.colorbar(im, ax=a[-1], fraction=0.046, label="|attribution| (common scale)")
            for ax in a:
                ax.axis("off")
        fig.suptitle(f"{b.name}: Integrated Gradients (common colour scale per row)", fontsize=10)
        fig.tight_layout()
        fig.savefig(out_dir / f"{b.name}_{pid}_ig.png", dpi=FIG_DPI)
        plt.close(fig)


# ============================================================
# 4. UNCERTAINTY
# ============================================================

def auroc_from_hist(pos_hist, neg_hist):
    n_pos, n_neg = pos_hist.sum(), neg_hist.sum()
    if n_pos == 0 or n_neg == 0:
        return math.nan
    neg_below = np.concatenate([[0], np.cumsum(neg_hist)[:-1]])
    return float((pos_hist * (neg_below + 0.5 * neg_hist)).sum() / (n_pos * n_neg))


UNCERTAINTY_MEASURES = ("entropy", "mutual_info", "disagreement")   # total vs TTA-disagreement

# in-plane structuring element: slices are ~7 mm thick, so the ROI is NOT dilated across slices
_ROI_STRUCT = np.zeros((3, 3, 3), dtype=bool)
_ROI_STRUCT[:, :, 1] = generate_binary_structure(2, 1)


def run_uncertainty(b, preds, unc):
    """Errors are abnormal-WMH DETECTION errors (PV<->deep confusions count as correct).
    AUROC is reported over the whole brain (dominated by easy background) and inside a
    lesion ROI = (GT | predicted abnormal WMH) dilated in-plane by UNC_ROI_DILATION voxels."""
    rows = []
    edges = np.linspace(0.0, 1.0, ENTROPY_BINS + 1)
    new_hists = lambda: {k: (np.zeros(ENTROPY_BINS, np.int64), np.zeros(ENTROPY_BINS, np.int64))
                         for k in UNCERTAINTY_MEASURES}
    pooled, pooled_roi = new_hists(), new_hists()
    test_rows = {r["patient_id"]: r for r in read_test_per_patient(b)}
    for pid in sorted(b.test_ids):
        vols = b.volumes(pid)
        gt, pred = vols["label"], preds[pid]
        brain = np.any(vols["mri"] != 0, axis=0)
        g, p = np.isin(gt, ABNORMAL), np.isin(pred, ABNORMAL)
        cats = {"TP": g & p & brain, "FP": ~g & p & brain, "FN": g & ~p & brain, "TN": ~g & ~p & brain}
        err = cats["FP"] | cats["FN"]
        ok = cats["TP"] | cats["TN"]
        roi = binary_dilation(g | p, structure=_ROI_STRUCT, iterations=UNC_ROI_DILATION) & brain
        row = {"model": b.name, "patient_id": pid, "n_roi_voxels": int(roi.sum())}
        for c, m in cats.items():
            row[f"n_{c}"] = int(m.sum())
        for k in UNCERTAINTY_MEASURES:
            u = unc[k][pid].astype(np.float32)
            for c, m in cats.items():
                row[f"mean_{k}_{c}"] = float(u[m].mean()) if m.any() else math.nan
            for hists, e_mask, o_mask, suffix in ((pooled, err, ok, ""),
                                                  (pooled_roi, err & roi, ok & roi, "_lesion_roi")):
                h_err, _ = np.histogram(np.clip(u[e_mask], 0, 1), bins=edges)
                h_ok, _ = np.histogram(np.clip(u[o_mask], 0, 1), bins=edges)
                hists[k][0][:] += h_err
                hists[k][1][:] += h_ok
                row[f"auroc_{k}_detects_error{suffix}"] = auroc_from_hist(h_err, h_ok)
            row[f"mean_{k}_in_predicted_abnormal"] = float(u[p & brain].mean()) if (p & brain).any() else math.nan
        tr = test_rows.get(pid, {})
        row["test_dice_abnormal_wmh"] = float(tr.get("abnormal_wmh_dice", "nan"))
        rows.append(row)
    summary = {"model": b.name}
    for k in UNCERTAINTY_MEASURES:
        summary[f"pooled_auroc_{k}_detects_error"] = auroc_from_hist(*pooled[k])
        summary[f"pooled_auroc_{k}_detects_error_lesion_roi"] = auroc_from_hist(*pooled_roi[k])
        m, lo, hi, n = bootstrap_mean_ci([r[f"auroc_{k}_detects_error_lesion_roi"] for r in rows])
        summary.update({f"mean_patient_auroc_{k}_lesion_roi": m, f"mean_patient_auroc_{k}_lesion_roi_ci95_low": lo,
                        f"mean_patient_auroc_{k}_lesion_roi_ci95_high": hi})
        summary[f"spearman_patient_{k}_vs_dice"] = spearman([r[f"mean_{k}_in_predicted_abnormal"] for r in rows],
                                                            [r["test_dice_abnormal_wmh"] for r in rows])
        for c in ("TP", "FP", "FN", "TN"):
            m, lo, hi, n = bootstrap_mean_ci([r[f"mean_{k}_{c}"] for r in rows])
            summary.update({f"mean_{k}_{c}": m, f"mean_{k}_{c}_ci95_low": lo, f"mean_{k}_{c}_ci95_high": hi})
    return rows, summary


def read_test_per_patient(b):
    path = b.mod.FINAL_LOG_DIR / "test30_per_patient_metrics.csv"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def plot_uncertainty_examples(b, preds, unc, examples, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for pid in examples:
        vols = b.volumes(pid)
        gt, pred = vols["label"], preds[pid]
        z = ig_slice_for_patient(b, pid)
        flair = b.mod.load_flair(pid)
        g, p = np.isin(gt[:, :, z], ABNORMAL), np.isin(pred[:, :, z], ABNORMAL)
        err = np.zeros(g.shape + (3,), np.float32)
        err[g & p] = (0, 1, 0)
        err[~g & p] = (1, 0, 0)
        err[g & ~p] = (0, 0.4, 1)
        fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
        axes[0].imshow(np.rot90(flair[:, :, z]), cmap="gray")
        axes[0].set_title(f"{pid} z={z} FLAIR", fontsize=9)
        axes[1].imshow(np.rot90(flair[:, :, z]), cmap="gray")
        axes[1].imshow(np.rot90(err), alpha=np.rot90((g | p).astype(float)) * 0.8)
        axes[1].set_title("abnormal WMH: TP green / FP red / FN blue", fontsize=9)
        for ax, k, title in ((axes[2], "entropy", "predictive entropy (total, normalised)"),
                             (axes[3], "mutual_info", "TTA disagreement (mutual information)")):
            u = unc[k][pid][:, :, z].astype(np.float32)
            vmax = 1.0 if k == "entropy" else max(float(np.percentile(u, 99.9)), 1e-6)
            im = ax.imshow(np.rot90(u), cmap="magma", vmin=0, vmax=vmax)
            ax.set_title(title, fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046)
        for ax in axes:
            ax.axis("off")
        fig.suptitle(b.name, fontsize=10)
        fig.tight_layout()
        fig.savefig(out_dir / f"{b.name}_{pid}_uncertainty.png", dpi=FIG_DPI)
        plt.close(fig)


# ============================================================
# 5. RULE-BASED PV/DEEP COMPARISON (saved predictions)
# ============================================================

RULE_COMPARISONS = [
    # (description, (model, variant) A, (model, variant) B) -> difference A - B
    ("distance model minus baseline + 10 mm rule",
     ("distance_unet_v2", "model"), ("baseline_unet_v2", "model+10mm_rule")),
    ("baseline + 10 mm rule minus baseline",
     ("baseline_unet_v2", "model+10mm_rule"), ("baseline_unet_v2", "model")),
    ("distance model + 10 mm rule minus distance model",
     ("distance_unet_v2", "model+10mm_rule"), ("distance_unet_v2", "model")),
]
RULE_METRICS = ("dice_pv", "dice_deep", "dice_abnormal_wmh")


def run_rule_based(bundles_info, out_dir):
    """Does the distance model do better than simply splitting a pooled lesion
    prediction into PV/deep with the 10 mm rule (same expert-mask distance map)?"""
    rows = []
    for name, info in bundles_info.items():
        get_vols = info["get_volumes"]
        for pid in info["test_ids"]:
            pred = load_saved_prediction(info, pid)
            gt = get_vols(pid)["label"]
            ab = np.isin(pred, ABNORMAL)
            rule = pred.copy()
            rule[ab] = geometry_rule(distance_mm(pid))[ab]
            for variant, p in (("model", pred), ("model+10mm_rule", rule)):
                rows.append({"model": name, "variant": variant, "patient_id": pid,
                             "dice_pv": dice(p == PV_LABEL, gt == PV_LABEL),
                             "dice_deep": dice(p == DEEP_LABEL, gt == DEEP_LABEL),
                             "dice_abnormal_wmh": dice(np.isin(p, ABNORMAL), np.isin(gt, ABNORMAL))})
    write_csv(out_dir / "rule_based_per_patient.csv", rows)

    summary = []
    for (model, variant) in sorted({(r["model"], r["variant"]) for r in rows}):
        for m in RULE_METRICS:
            mean, lo, hi, n = bootstrap_mean_ci([r[m] for r in rows if r["model"] == model and r["variant"] == variant])
            summary.append({"model": model, "variant": variant, "metric": m,
                            "mean": mean, "ci95_low": lo, "ci95_high": hi, "n": n})
    write_csv(out_dir / "rule_based_summary.csv", summary)

    def values(key, m):
        return {r["patient_id"]: r[m] for r in rows if (r["model"], r["variant"]) == key}

    stats = []
    available = {(r["model"], r["variant"]) for r in rows}
    for desc, a_key, b_key in RULE_COMPARISONS:
        if a_key not in available or b_key not in available:
            continue
        for m in ("dice_pv", "dice_deep"):
            a, bb = values(a_key, m), values(b_key, m)
            diffs = [a[p] - bb[p] for p in a if p in bb and np.isfinite(a[p]) and np.isfinite(bb[p])]
            mean, lo, hi, n = bootstrap_mean_ci(diffs)
            stats.append({"comparison": desc, "metric": m, "mean_diff": mean, "ci95_low": lo, "ci95_high": hi,
                          "n": n, "n_improved": int(sum(d > 0 for d in diffs)),
                          "n_worsened": int(sum(d < 0 for d in diffs)), "wilcoxon_p": paired_wilcoxon(diffs)})
    for s, ph in zip(stats, holm([s["wilcoxon_p"] for s in stats])):
        s["wilcoxon_p_holm"] = ph
    write_csv(out_dir / "rule_based_comparison.csv", stats)
    for s in stats:
        print(f"  rule-based: {s['comparison']:50s} {s['metric']:10s} diff {s['mean_diff']:+.4f} "
              f"[{s['ci95_low']:+.4f}, {s['ci95_high']:+.4f}]  p_Holm {s['wilcoxon_p_holm']:.3g}")


# ============================================================
# MAIN
# ============================================================

RUN_FOLDERS = {"baseline_unet_v2": "baseline_unet_v2", "distance_unet_v2": "distance_map_unet_v2"}


def preflight_files():
    """Fail early if any required input is missing (no model is loaded here)."""
    missing = []
    test_csv = PROJECT_DIR / "splits" / "baseline_2d" / "test_30_patients.csv"
    if not test_csv.exists():
        missing.append(test_csv)
        test_ids = []
    else:
        with test_csv.open("r", encoding="utf-8", newline="") as f:
            test_ids = [r["patient_id"].strip().zfill(3) for r in csv.DictReader(f)]
    for name, folder in RUN_FOLDERS.items():
        need = [PROJECT_DIR / "checkpoints" / folder / "final70" / "final70_model.pth",
                PROJECT_DIR / "logs" / folder / "final_results" / "test30_summary.json",
                PROJECT_DIR / "logs" / folder / "final_results" / "test30_per_patient_metrics.csv",
                PROJECT_DIR / "logs" / folder / "cv_summary" / "postprocessing_choice.json"]
        need += [PROJECT_DIR / "logs" / folder / "final_results" / "predictions_nifti" / f"{pid}_pred.nii.gz"
                 for pid in test_ids]
        missing += [p for p in need if not p.exists()]
    missing += [DISTANCE_MM_ROOT / pid / f"{pid}_VentDist_mm.nii.gz" for pid in test_ids
                if not (DISTANCE_MM_ROOT / pid / f"{pid}_VentDist_mm.nii.gz").exists()]
    # counterfactual donors: development patients' ventricle masks + FLAIR, and D_max
    dev_csv = PROJECT_DIR / "splits" / "baseline_2d" / "train_70_patients.csv"
    if not dev_csv.exists():
        missing.append(dev_csv)
        dev_ids = []
    else:
        with dev_csv.open("r", encoding="utf-8", newline="") as f:
            dev_ids = [r["patient_id"].strip().zfill(3) for r in csv.DictReader(f)]
    for pid in dev_ids + test_ids:
        for suffix in ("Vent_Mask", "FLAIR"):
            p = PREPROCESSED_ROOT / pid / f"{pid}_{suffix}.nii.gz"
            if not p.exists():
                missing.append(p)
    if not (DISTANCE_MM_ROOT / "distance_config.json").exists():
        missing.append(DISTANCE_MM_ROOT / "distance_config.json")
    if len(test_ids) != 30:
        missing.append(f"expected 30 test patients, found {len(test_ids)}")
    if missing:
        raise FileNotFoundError("Preflight failed, missing:\n  " + "\n  ".join(str(m) for m in missing[:40]))
    print(f"Preflight: all required files present for {list(RUN_FOLDERS)} and {len(test_ids)} test patients.")


def run_check():
    """--check: one test patient per model, every code path once, NOTHING written."""
    from torch.utils.data import DataLoader, Subset
    for name, filename in MODELS.items():
        print(f"\n=== CHECK {name} ===")
        b = ModelBundle(name, filename)
        pid = sorted(b.test_ids)[0]
        idx = [i for i, (_, p, _) in enumerate(b.dataset.samples) if p == pid]
        sub_loader = DataLoader(Subset(b.dataset, idx), batch_size=b.loader.batch_size, shuffle=False, num_workers=0)
        counterfactuals, skipped = build_counterfactuals(b, [pid]) if b.has_dist else ({}, [])
        if b.has_dist:
            # the recomputation must reproduce the stored map when applied to the patient's OWN mask
            own_mm, _ = distance_maps_from_mask(anatomy(pid)["vent"], anatomy(pid)["spacing"], load_d_max())
            err = float(np.abs(own_mm - distance_mm(pid)).max())
            print(f"  distance recomputation vs stored map: max abs diff {err:.2e} mm")
            if err > 1e-3:
                raise RuntimeError(f"{name}: distance recomputation differs from 15a output ({err:.3g} mm)")
            if pid in counterfactuals:
                q = counterfactuals[pid]["qc"]
                print(f"  counterfactual donor {q['donor']} (rank {q['donor_rank']}), brain Dice "
                      f"{q['brain_dice_after_registration']:.3f}, affected voxels {q['n_affected_voxels']} "
                      f"({q['pct_affected_voxels']:.1f}%)")
            else:
                print(f"  counterfactual skipped: {skipped[0]['reason']} [{skipped[0]['attempts']}]")
        intact_sub = None
        for cond, transform in make_conditions(b, counterfactuals).items():
            preds, unc = predict_condition(b, transform, want_entropy=(cond == "intact"), loader=sub_loader, ids=[pid])
            if cond == "intact":
                intact_sub = preds
                compare_with_saved(b, preds)
                for k in UNCERTAINTY_MEASURES:
                    u = unc[k][pid].astype(np.float32)
                    if not (np.isfinite(u).all() and u.min() >= -1e-3 and u.max() <= 1 + 1e-3):
                        raise RuntimeError(f"{name}: {k} outside [0, 1]")
            cf = counterfactuals.get(pid) if cond in CF_EVAL_CONDITIONS else None
            m = ablation_metrics(b, preds[pid], pid, cf, None if cond == "intact" else intact_sub[pid])
            print(f"  {cond:20s} abnormal Dice {m['dice_abnormal_wmh']:.4f}"
                  + (f", pred change in affected {m.get('cf_pred_change_in_affected', math.nan):.3f}, "
                     f"switch consistent {m.get('cf_switch_consistent', math.nan):.3f}"
                     if cond == "dist_counterfactual" else ""))
        z = ig_slice_for_patient(b, pid)
        x = b.dataset[b.sample_index[(pid, z)]]["image"][None].to(b.device).float()
        with torch.no_grad():
            pred = torch.argmax(b.model(x).float(), dim=1)[0].cpu().numpy()
        for cname, cls in IG_CLASSES.items():
            region = pred == cls
            if region.sum() >= IG_MIN_REGION_VOXELS:
                _, q = integrated_gradients(b, x, cls, region)
                print(f"  IG {cname}: delta {q['score_delta']:.4f}, completeness error "
                      f"{q['completeness_rel_error']:.4f} ({'OK' if q['completeness_ok'] else 'FLAGGED'})")
                break
        else:
            print(f"  IG: no PV/deep region >= {IG_MIN_REGION_VOXELS} voxels on {pid} slice {z} (not an error)")
        b.release()
    print("\nCHECK PASSED — nothing was written. Run without --check for the full analysis.")


def main():
    preflight_files()
    if CHECK_ONLY:
        run_check()
        return
    if OUT_ROOT.exists() and any(OUT_ROOT.iterdir()) and not OVERWRITE:
        raise RuntimeError(f"{OUT_ROOT} already contains results. Refusing to overwrite; "
                           "move it away or pass --overwrite.")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_json(OUT_ROOT / "explainability_config.json", {
        "models": MODELS,
        "test_set": "30 locked test patients; final 70-patient models; no training or tuning",
        "prediction": "TTA and post-processing exactly as in scripts 14/17 (imported)",
        "ablation_conditions": ["intact", "no_FLAIR", "no_T2", "no_T1", "no_neighbour_slices",
                                "dist_zero (distance model; unusual-input stress test, NOT a "
                                "'no distance information' condition)",
                                "dist_counterfactual (distance model)"],
        "ablation_zero_value": "0 = mean brain intensity of z-scored MRI (and background)",
        "dist_counterfactual": {
            "name": "anatomically controlled counterfactual distance-map perturbation "
                    "(not a clinically validated anatomical replacement)",
            "space_check": "patients are in individual native FLAIR spaces (script 10: per-patient reference "
                           "grid, intra-patient registration only); equal array shapes are not treated as aligned",
            "donor_pool": "70 development patients (never test patients)",
            "donor_ranking": f"z-scored ventricle features {VENT_FEATURE_NAMES}; up to "
                             f"{MAX_DONOR_CANDIDATES} nearest candidates tried in order",
            "registration": "donor brain -> target brain, axis-aligned affine in world mm (centroid "
                            "translation + per-axis scaling by brain spread); ventricle mask resampled with "
                            "nearest neighbour; voxels outside the target brain removed",
            "plausibility_checks": {"min_brain_dice": CF_MIN_BRAIN_DICE, "scale_range": list(CF_SCALE_RANGE),
                                    "vent_volume_ratio": list(CF_VENT_VOLUME_RATIO),
                                    "min_fraction_inside_brain": CF_MIN_INSIDE_BRAIN,
                                    "centroid_inside_brain": True},
            "distance": "recomputed as in 15a: 3D EDT in mm on the target grid, clip at D_max, / D_max",
            "affected_voxels": "target-brain voxels outside both ventricle masks whose 10 mm PV/deep "
                               "assignment differs between original and counterfactual distance",
            "metrics": {
                "cf_follows_geometry_change": "AGREEMENT of the counterfactual prediction with the new geometry "
                                              "(affected voxels predicted abnormal in both runs); includes labels "
                                              "that were already consistent with it",
                "cf_switch_fraction": "fraction of those voxels whose label changed between runs",
                "cf_switch_consistent": "fraction whose label switched from the true-geometry label to the "
                                        "counterfactual-geometry label (primary counterfactual metric)",
            },
            "cf_metrics_computed_for": list(CF_EVAL_CONDITIONS),
            "skipping": "no suitable donor -> target skipped and reported in counterfactual_skipped.csv",
            "limitation": "low-DOF registration; no deformable registration (e.g. ANTs SyN) is used",
        },
        "pv_threshold_mm": PV_THRESHOLD_MM,
        "distance_bands_mm": [[lo, hi if np.isfinite(hi) else "inf"] for lo, hi in DISTANCE_BANDS_MM],
        "distance_band_label_accuracy": "PV/deep label accuracy among correctly detected abnormal voxels per band; "
                                        "paired distance - baseline difference, Wilcoxon + Holm across bands",
        "integrated_gradients": {"steps": IG_STEPS, "baseline": "all zeros (DIST = 0 = at the ventricle; DIST "
                                 "shares are therefore not comparable between PV and deep classes)",
                                 "score": "sum of softmax prob of the target class over the model's own predicted "
                                          "region", "region": "single forward pass, no TTA, no post-processing",
                                 "slice": "per patient, the slice with most GT abnormal WMH",
                                 "min_region_voxels": IG_MIN_REGION_VOXELS},
        "ig_quality_flags": {"completeness_tolerance": IG_COMPLETENESS_TOL,
                             "min_score_delta": IG_MIN_SCORE_DELTA},
        "uncertainty": {"entropy": "normalised entropy of the TTA-mean softmax (total predictive uncertainty)",
                        "mutual_info": "H(mean) - mean H(flip), normalised (TTA disagreement)",
                        "disagreement": "fraction of TTA flips whose argmax differs from the final label",
                        "error_definition": "abnormal-WMH detection errors (FP/FN); PV<->deep confusions count "
                                            "as correct",
                        "error_detection": "AUROC for FP/FN vs correct voxels, (a) whole brain, (b) lesion ROI = "
                                           f"(GT | predicted abnormal) dilated in-plane by {UNC_ROI_DILATION} voxels"},
        "rule_based": {"definition": "each model's saved prediction with abnormal voxels relabelled PV/deep by "
                                     "the 10 mm rule on the expert-mask distance map",
                       "comparisons": [c[0] for c in RULE_COMPARISONS],
                       "statistics": "paired Wilcoxon over patients, Holm across all comparisons, bootstrap CI"},
        "ground_truth_dependencies": [
            "distance channel computed from the EXPERT ventricle mask, also for test patients (script 17)",
            "the 10 mm rule in the rule-based comparison uses the same expert-mask distance map",
            "GT labels used only for fixed slice / example selection and for evaluation, never as input",
        ],
        "reproducibility_check": f"intact predictions must match saved test predictions "
                                 f"(max voxel mismatch {MAX_INTACT_MISMATCH})",
        "figure_dpi": FIG_DPI,
        "bootstrap": {"n": N_BOOTSTRAP, "seed": BOOTSTRAP_SEED, "ci": "95% percentile, over patients"},
        "references": ["Sundararajan et al., ICML 2017 (Integrated Gradients)",
                       "Wang et al., Neurocomputing 2019 (TTA uncertainty)",
                       "Griffanti et al., NeuroImage 2018 (10 mm periventricular rule)"],
    })

    ablation_rows, ig_rows, unc_rows, unc_summaries, repro_rows = [], [], [], [], []
    bands_info = {}

    for name, filename in MODELS.items():
        print(f"\n=== {name} ({filename}) ===")
        b = ModelBundle(name, filename)
        examples = example_patients(b.mod, b.test_ids, N_EXAMPLE_PATIENTS)
        bands_info[name] = {"pred_dir": b.mod.FINAL_LOG_DIR / "predictions_nifti",
                            "get_volumes": b.mod.get_patient_volumes, "test_ids": list(b.test_ids)}

        intact_preds = intact_entropy = None
        if not SKIP_ABLATION:
            rows, intact_preds, intact_entropy, counterfactuals, cf_preds, repro = run_ablation(
                b, OUT_ROOT / "ablation")
            ablation_rows.extend(rows)
            repro_rows.extend(repro)
            if b.has_dist:
                plot_counterfactual_examples(b, counterfactuals, intact_preds, cf_preds, examples,
                                             OUT_ROOT / "ablation" / "counterfactual_examples")
                plot_counterfactual_summary(rows, counterfactuals, OUT_ROOT / "ablation" / "plots")
        elif not SKIP_UNCERTAINTY:
            intact_preds, intact_entropy = predict_condition(b, None, want_entropy=True)
            repro_rows.extend(compare_with_saved(b, intact_preds))

        if not SKIP_UNCERTAINTY:
            rows, summary = run_uncertainty(b, intact_preds, intact_entropy)
            unc_rows.extend(rows)
            unc_summaries.append(summary)
            plot_uncertainty_examples(b, intact_preds, intact_entropy, examples, OUT_ROOT / "uncertainty" / "examples")

        if not SKIP_IG:
            rows, maps = run_ig(b, examples)
            ig_rows.extend(rows)
            plot_ig_examples(b, maps, OUT_ROOT / "attribution" / "examples")

        b.release()

    if repro_rows:
        write_csv(OUT_ROOT / "reproducibility_vs_saved_predictions.csv", repro_rows)
    if ablation_rows:
        write_csv(OUT_ROOT / "ablation" / "ablation_per_patient.csv", ablation_rows)
        summary = summarize_ablation(ablation_rows)
        write_csv(OUT_ROOT / "ablation" / "ablation_summary.csv", summary)
        plot_ablation(summary, OUT_ROOT / "ablation" / "plots")
    if unc_rows:
        write_csv(OUT_ROOT / "uncertainty" / "uncertainty_per_patient.csv", unc_rows)
        write_csv(OUT_ROOT / "uncertainty" / "uncertainty_summary.csv", unc_summaries)
    if ig_rows:
        write_csv(OUT_ROOT / "attribution" / "ig_per_patient.csv", ig_rows)
        write_csv(OUT_ROOT / "attribution" / "ig_summary.csv", plot_ig_summary(ig_rows, OUT_ROOT / "attribution" / "plots"))
    if not SKIP_BANDS:
        run_distance_bands(bands_info, OUT_ROOT / "distance_bands")
    if not SKIP_RULE:
        run_rule_based(bands_info, OUT_ROOT / "rule_based")

    print(f"\nExplainability outputs written to {OUT_ROOT}")


if __name__ == "__main__":
    main()