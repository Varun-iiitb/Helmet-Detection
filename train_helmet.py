"""
train_helmet.py
Train YOLO11l on the unified helmet dataset.

Steps (in order, gated):
  1. Pre-flight dataset verification — STOPS the script on any failure.
  2. Train.
  3. Print final metrics.
  4. Verify the trained model's class names.
  5. Quick sanity-check inference on a val image.
  6. Copy best.pt into ./models/helmet_model.pt for the submission tree.

Critical project rules from the spec:
  - batch=16 (do NOT raise; batch=128 was observed to cause class collapse).
  - fl_gamma=2.0; if the installed Ultralytics rejects it, fall back to 1.5
    (and finally drop it) so a CLI version bump doesn't sink the run.
  - No `freeze` — full network trains from epoch 0.
"""

# Why os / shutil / Path: filesystem walks, copying weights, building paths.
import os
import shutil
from pathlib import Path
# Why yaml: parse data.yaml to read class names + count before training.
import yaml
# Why Counter: tally per-class boxes across train labels in one pass.
from collections import Counter


# ---------- Paths ----------

# Our actual on-disk layout is `unified_dataset/{images,labels}/{train,val}`
# rather than the `train/labels` layout the spec assumed; the spec explicitly
# says "change to your actual merged dataset path", so we do.
DATASET_DIR = "unified_dataset"
DATA_YAML = f"{DATASET_DIR}/data.yaml"

# Where YOLO will write the run; also where we'll read best.pt from later.
RUN_NAME = "helmet_yolo11l_final"
BEST_WEIGHTS = f"runs/detect/{RUN_NAME}/weights/best.pt"


# =====================================================================
# Step 2 — Verify dataset BEFORE training
# =====================================================================

def verify_dataset():
    """
    Hard-fail (raise AssertionError) if anything looks wrong. Why hard-fail:
    if the dataset is malformed we want to know NOW, before burning hours
    of GPU time on a doomed run.
    Returns the parsed data.yaml config so the training step can reuse it.
    """
    print("===== DATASET VERIFICATION =====")

    # ---- Check 1: data.yaml exists with the right class count + names ----
    with open(DATA_YAML) as f:
        cfg = yaml.safe_load(f)
    print(f"Classes: {cfg['names']}")
    print(f"Number of classes: {cfg['nc']}")

    assert cfg["nc"] == 2, f"Expected 2 classes, got {cfg['nc']}"
    # Lowercased flat-text search — covers ['helmet', 'without_helmet'],
    # ['helmet', 'no_helmet'], etc. without enforcing a specific spelling.
    names_text = str(cfg["names"]).lower()
    assert "helmet" in names_text, "helmet class missing"
    assert "no_helmet" in names_text or "without" in names_text, (
        "no-helmet / without-helmet class missing"
    )
    print("OK data.yaml verified")

    # ---- Check 2: per-class box counts across the training labels ----
    # Path follows our `labels/train` layout, not the spec's `train/labels`.
    label_dir = Path(DATASET_DIR) / "labels" / "train"
    class_counts = Counter()
    empty_files = 0
    total_files = 0
    for f in label_dir.glob("*.txt"):
        total_files += 1
        lines = f.read_text().strip().split("\n")
        if not lines or lines == [""]:
            # Empty label = no positive boxes (model would learn background-only
            # signal). Counted but skipped at the class-tally step.
            empty_files += 1
            continue
        for line in lines:
            if line.strip():
                # First whitespace-delimited token is the class id.
                cls = int(line.split()[0])
                class_counts[cls] += 1

    print(f"\nTraining label files: {total_files}")
    print(f"Empty label files:    {empty_files}")
    print("Class distribution:")
    for cls_id, count in sorted(class_counts.items()):
        # cfg["names"] is a list or dict depending on YAML — handle both.
        if isinstance(cfg["names"], dict):
            name = cfg["names"].get(cls_id, str(cls_id))
        else:
            name = cfg["names"][cls_id]
        print(f"  Class {cls_id} ({name}): {count} boxes")

    # ---- Check 3: class-imbalance warning (informational, doesn't fail) ----
    # Why >5x: empirically the imbalance threshold where standard cross-entropy
    # starts to misbehave on detection; below this YOLO + focal loss handles
    # it fine without resampling.
    fl_gamma_recommended = False
    if len(class_counts) == 2:
        counts = list(class_counts.values())
        ratio = max(counts) / min(counts)
        if ratio > 5:
            print(f"\nWARNING: Class imbalance ratio = {ratio:.1f}x")
            print("   Will use fl_gamma=2.0 to handle this")
            fl_gamma_recommended = True
        else:
            print(f"\nOK Class balance ratio: {ratio:.1f}x - acceptable")

    # ---- Check 4: at least one training image exists ----
    img_dir = Path(DATASET_DIR) / "images" / "train"
    img_count = len(list(img_dir.glob("*.jpg"))) + len(list(img_dir.glob("*.png")))
    print(f"\nTraining images: {img_count}")
    assert img_count > 0, "No training images found"
    print("OK Dataset verification complete")
    print("================================\n")

    return cfg, fl_gamma_recommended


# =====================================================================
# Step 3 — Train with version-tolerant fl_gamma fallback
# =====================================================================

def train_model():
    # Imported lazily so Step 2's verification can run/fail quickly without
    # waiting for the (slow) ultralytics import chain.
    from ultralytics import YOLO

    # YOLO11l pretrained on COCO — strong starting point for transfer learning.
    model = YOLO("yolo11l.pt")

    # Hyperparameters per the spec. Comments explain every non-obvious choice.
    train_kwargs = dict(
        # ---- Core ----
        data=DATA_YAML,
        # Why epochs=150: long enough that early stopping (patience=30) can
        # actually trigger before the cap on most well-behaved runs.
        epochs=150,
        imgsz=640,
        # Why batch=16: locked by the spec. batch=128 caused class collapse
        # in prior runs (likely a BN-statistics issue on small datasets).
        batch=16,
        # device=0 = first CUDA GPU. Change to 'cpu' on machines without one.
        device=0,
        workers=4,

        # ---- Run identity ----
        name=RUN_NAME,
        exist_ok=True,

        # ---- Early stopping ----
        # Why 30: long enough to ride out plateaus, short enough that we
        # don't spend epochs chasing diminishing returns.
        patience=30,

        # ---- LR schedule ----
        # Standard YOLO recipe: 0.01 start, cosine decay to 0.01 * lr0 = 1e-4.
        lr0=0.01,
        lrf=0.01,
        # Why warmup=5: helps the new detection head settle before the
        # backbone gets large gradients via BN.
        warmup_epochs=5,

        # ---- Loss weights ----
        cls=0.5,
        # Why fl_gamma=2.0: focal-loss focusing parameter; higher = more
        # emphasis on hard examples, useful for the ~2:1 class imbalance.
        fl_gamma=2.0,

        # ---- Augmentation (tuned for motorcycle crops) ----
        # Why these specific values: spec defaults; they reflect the user's
        # earlier experiments on similar motorcycle-helmet data.
        hsv_h=0.015,    # mild hue jitter
        hsv_s=0.7,      # generous saturation (handles tinted helmets)
        hsv_v=0.5,      # brightness swing - covers day/night/CCTV exposure
        degrees=5.0,    # small rotation - helmets are mostly upright
        flipud=0.0,     # NEVER vertical-flip a helmet (riders aren't upside-down)
        fliplr=0.5,     # horizontal flip is fine; helmets are symmetric
        mosaic=1.0,     # always-on 4-way mosaic
        mixup=0.1,      # occasional pair-blending
        copy_paste=0.1, # paste helmet/no-helmet objects onto new backgrounds
        erasing=0.3,    # random erase to simulate occluded helmets
        # crop_fraction removed: deprecated in current Ultralytics; it was a
        # classification-only augmentation and doesn't affect detection anyway.
    )

    # Try the configured fl_gamma; if Ultralytics rejects it (older or newer
    # versions sometimes drop the keyword), retry with 1.5, then drop it
    # entirely. Why this ladder: per the spec, fl_gamma=2.0 is the target,
    # 1.5 is the fallback; we add the final no-fl_gamma branch so a Ultralytics
    # version that removed the flag still trains, just without focal-loss tuning.
    def _is_fl_gamma_error(exc):
        msg = str(exc).lower()
        return "fl_gamma" in msg

    # Why SyntaxError too: Ultralytics 8.4+ raises SyntaxError (not TypeError)
    # from check_dict_alignment when it sees an unknown argument like fl_gamma.
    fl_gamma_errors = (TypeError, ValueError, KeyError, SyntaxError)
    try:
        return model.train(**train_kwargs)
    except fl_gamma_errors as exc:
        if not _is_fl_gamma_error(exc):
            raise
        print(f"\nfl_gamma=2.0 rejected ({exc}); retrying with fl_gamma=1.5")
        train_kwargs["fl_gamma"] = 1.5
        try:
            return model.train(**train_kwargs)
        except fl_gamma_errors as exc2:
            if not _is_fl_gamma_error(exc2):
                raise
            print(f"\nfl_gamma=1.5 also rejected ({exc2}); training without fl_gamma")
            train_kwargs.pop("fl_gamma", None)
            return model.train(**train_kwargs)


# =====================================================================
# Step 4 — Print results summary
# =====================================================================

def print_results(results):
    print("\n===== TRAINING COMPLETE =====")
    # results.results_dict keys come from Ultralytics' metric names. We use
    # .get(...) so a renamed key in a future version doesn't crash the print.
    rd = results.results_dict
    print(f"Best mAP@50:    {rd.get('metrics/mAP50(B)', float('nan')):.3f}")
    print(f"Best mAP@50-95: {rd.get('metrics/mAP50-95(B)', float('nan')):.3f}")
    print(f"Precision:      {rd.get('metrics/precision(B)', float('nan')):.3f}")
    print(f"Recall:         {rd.get('metrics/recall(B)', float('nan')):.3f}")
    print(f"\nBest weights: {BEST_WEIGHTS}")


# =====================================================================
# Step 5 — Verify trained model classes
# =====================================================================

def verify_trained_model():
    from ultralytics import YOLO

    trained = YOLO(BEST_WEIGHTS)
    print("\n===== MODEL VERIFICATION =====")
    print(f"Model classes: {trained.names}")
    # Why this assertion: catches the case where label ids got shuffled
    # during training (a known footgun if you mix data.yaml class order
    # with a checkpoint trained on different order).
    cls0 = str(trained.names[0]).lower()
    assert cls0 in ("helmet", "with_helmet", "withhelmet"), (
        f"Class 0 should be helmet, got {trained.names[0]}"
    )
    print("OK Class indices verified")
    return trained


# =====================================================================
# Step 6 — Quick sanity-check inference on a val image
# =====================================================================

def sample_inference(trained):
    # Our val folder is `images/val` (not `valid/images`), so we check both
    # spellings in case someone runs this against a Roboflow-shaped dataset.
    candidate_dirs = [
        Path(DATASET_DIR) / "images" / "val",
        Path(DATASET_DIR) / "valid" / "images",
    ]
    val_imgs = []
    for d in candidate_dirs:
        if d.exists():
            val_imgs = list(d.glob("*.jpg")) + list(d.glob("*.png"))
            if val_imgs:
                break

    if not val_imgs:
        print("\nNo val images found, skipping sample inference.")
        return

    test_img = str(val_imgs[0])
    print(f"\nTesting on: {test_img}")

    r = trained(test_img, conf=0.25)
    print(f"Detections: {len(r[0].boxes)}")
    for box in r[0].boxes:
        cls = int(box.cls)
        conf = float(box.conf)
        print(f"  {trained.names[cls]:12} conf: {conf:.2f}")

    # Why save: lets the user eyeball whether the model output looks sane
    # before they trust the numeric metrics.
    r[0].save("sample_output.jpg")
    print("Sample output saved to sample_output.jpg")


# =====================================================================
# Step 7 — Copy best weights into ./models for the submission tree
# =====================================================================

def save_to_models_dir():
    # models/ is where the final TrafficViolationDetector class will load
    # weights from per the project spec — single canonical location.
    Path("models").mkdir(exist_ok=True)
    dst = "models/helmet_model.pt"
    shutil.copy(BEST_WEIGHTS, dst)
    size_mb = Path(dst).stat().st_size / 1e6
    print(f"\nOK Final weights saved to {dst}")
    print(f"   File size: {size_mb:.1f} MB")


# =====================================================================
# Orchestrator
# =====================================================================

def main():
    # Step 2 first — failures here mean we never reach the (expensive) train call.
    cfg, _ = verify_dataset()

    # Step 3: train.
    results = train_model()

    # Step 4: print summary.
    print_results(results)

    # Step 5: model class verification on the saved checkpoint.
    trained = verify_trained_model()

    # Step 6: one inference sanity check.
    sample_inference(trained)

    # Step 7: stage the weights.
    save_to_models_dir()


if __name__ == "__main__":
    main()
