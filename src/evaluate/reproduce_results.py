"""
Reproduce the headline result of the thesis without a GPU.

Best configuration:
    Late Fusion (weighted) = DistilBERT (text) + ConvNeXt-Tiny (image)
    Fusion rule:  p = (1 - alpha) * p_text + alpha * p_img,  alpha = 0.35 (image weight)
    Decision:     per-label thresholds (one threshold per genre)
    Target score: Micro-F1 = 0.6463 on the test set (648 samples, 19 genres)

This script works entirely from *saved prediction probabilities*, so it runs on a
plain CPU in seconds. No model weights, no dataset download, no GPU required.

Usage:
    python reproduce_results.py
    python reproduce_results.py --results-dir /path/to/results
"""

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

# alpha is the IMAGE weight in  p = (1 - alpha) * p_text + alpha * p_img
IMG_WEIGHT = 0.35


def micro_f1(prob, y_true, thresholds):
    """Apply per-label thresholds and return micro-averaged F1, precision, recall."""
    pred = (prob >= thresholds).astype(int)
    return (
        f1_score(y_true, pred, average="micro", zero_division=0),
        precision_score(y_true, pred, average="micro", zero_division=0),
        recall_score(y_true, pred, average="micro", zero_division=0),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results",
        help="Path to the results/ directory (default: ../results relative to this file)",
    )
    args = parser.parse_args()
    r = args.results_dir

    # --- load saved artifacts -------------------------------------------------
    labels = np.load(r / "label_order.npy", allow_pickle=True)
    y_test = np.load(r / "y_test.npy").astype(int)
    thresholds = np.load(r / "thresholds" / "lf_weighted_perlabel_thresholds.npy")
    fusion_prob = np.load(r / "probs" / "lf_weighted_test_prob.npy")

    n_samples, n_labels = y_test.shape
    print(f"Loaded test set: {n_samples} samples x {n_labels} genres")
    print(f"Genres: {list(labels)}\n")

    # --- 1) primary reproduction: saved fusion probabilities ------------------
    f1, prec, rec = micro_f1(fusion_prob, y_test, thresholds)
    print("[1] Late Fusion (weighted), per-label thresholds  -- from saved fusion probs")
    print(f"    Micro-F1 = {f1:.4f}   (precision {prec:.4f}, recall {rec:.4f})\n")

    # --- 2) transparency: rebuild the fusion from its two components ----------
    # This recomputes the weighted late-fusion rule from the raw text and image
    # probabilities, then checks that it matches the saved fusion probabilities.
    p_text = np.load(r / "probs" / "text_distilbert_test_prob.npy")
    p_img = np.load(r / "probs" / "image_convnext_test_prob.npy")
    p_rebuilt = (1.0 - IMG_WEIGHT) * p_text + IMG_WEIGHT * p_img

    max_diff = float(np.abs(p_rebuilt - fusion_prob).max())
    matches = np.allclose(p_rebuilt, fusion_prob, atol=1e-5)
    f1_rebuilt, _, _ = micro_f1(p_rebuilt, y_test, thresholds)

    print("[2] Rebuilt fusion from components: 0.65 * text + 0.35 * image")
    print(f"    Formula matches saved fusion: {matches}  (max abs diff {max_diff:.2e})")
    print(f"    Micro-F1 from rebuilt probs = {f1_rebuilt:.4f}")
    if not matches:
        print("    NOTE: components do not match the saved fusion exactly; the saved-prob")
        print("          result in [1] is the authoritative number.")


if __name__ == "__main__":
    main()
