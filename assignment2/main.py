"""Reproduce training, full-test evaluation, and report figures.

Run: python assignment2/main.py --data-dir assignment2-data/mnist
Only NumPy is used for model training and metric calculations.
"""
import argparse
import gzip
import hashlib
import json
import platform
import struct
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Ellipse
import numpy as np

from naive_bayes import NaiveBayesClassifier
from bayes_net import BayesNetClassifier, FEATURE_NAMES, FEATURE_PARENTS


DATA_FILES = (
    "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz",
)


def read_idx(path):
    """Read the unsigned-byte IDX format used in the supplied MNIST files."""
    with gzip.open(path, "rb") as stream:
        header = stream.read(4)
        if len(header) != 4 or header[:3] != b"\x00\x00\x08" or header[3] not in (1, 3):
            raise ValueError(f"Invalid unsigned-byte MNIST IDX header: {path}")
        ndim = header[3]
        shape_data = stream.read(4 * ndim)
        if len(shape_data) != 4 * ndim:
            raise ValueError(f"Truncated dimensions: {path}")
        shape = struct.unpack(">" + "I" * ndim, shape_data)
        array = np.frombuffer(stream.read(), dtype=np.uint8)
        if array.size != int(np.prod(shape)):
            raise ValueError(f"Wrong IDX payload size: {path}")
        return array.reshape(shape)


def load_data(data_dir):
    manifest_path = data_dir / "manifest.json"
    hashes = {name: hashlib.sha256((data_dir / name).read_bytes()).hexdigest() for name in DATA_FILES}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for name, expected in manifest.get("sha256", {}).items():
            if name in hashes and hashes[name] != expected:
                raise ValueError(f"SHA256 mismatch: {name}")
    arrays = [read_idx(data_dir / name) for name in DATA_FILES]
    for arr, shape in zip(arrays, ((60000, 28, 28), (60000,), (10000, 28, 28), (10000,))):
        if arr.shape != shape:
            raise ValueError(f"Expected full MNIST shape {shape}, got {arr.shape}")
    return arrays[0].reshape(-1, 784), arrays[1], arrays[2].reshape(-1, 784), arrays[3], hashes


def evaluate(y_true, y_pred):
    """Rows are true labels, columns predicted labels; zero division yields zero."""
    cm = np.bincount(10 * y_true.astype(int) + y_pred.astype(int), minlength=100).reshape(10, 10)
    tp = cm.diagonal().astype(float)
    precision = np.divide(tp, cm.sum(axis=0), out=np.zeros(10), where=cm.sum(axis=0) != 0)
    recall = np.divide(tp, cm.sum(axis=1), out=np.zeros(10), where=cm.sum(axis=1) != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(10), where=precision + recall != 0)
    return {
        "accuracy": float(tp.sum() / cm.sum()),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "confusion_matrix": cm.tolist(),
        "per_class": [{"digit": i, "precision": float(precision[i]), "recall": float(recall[i]),
                       "f1": float(f1[i]), "support": int(cm[i].sum())} for i in range(10)],
    }


def confusion_pairs(cm):
    pairs = [(int(cm[a, b] + cm[b, a]), a, b) for a in range(10) for b in range(a + 1, 10)]
    pairs.sort(key=lambda entry: (-entry[0], entry[1], entry[2]))
    return [{"digits": [a, b], "total": total, "a_to_b": int(cm[a, b]), "b_to_a": int(cm[b, a])}
            for total, a, b in pairs[:5]]


def draw_confusion(cm, title, path):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=9,
                    color="white" if cm[i, j] > cm.max() * .55 else "black")
    ax.set(xticks=range(10), yticks=range(10), xlabel="Predicted digit", ylabel="True digit", title=title)
    fig.colorbar(im, ax=ax, shrink=.8, label="Number of images")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def draw_errors(X, y, predictions, other, pairs, title, path):
    # Choose the first example in each direction of the top three confused pairs.
    selected = []
    for pair in pairs[:3]:
        a, b = pair["digits"]
        for source, target in ((a, b), (b, a)):
            candidates = np.flatnonzero((y == source) & (predictions == target))
            if len(candidates):
                selected.append(int(candidates[0]))
    for i in np.flatnonzero(y != predictions):
        if len(selected) >= 6:
            break
        if int(i) not in selected:
            selected.append(int(i))
    records = [{"test_index": i, "true": int(y[i]), "predicted": int(predictions[i]),
                "other_model_predicted": int(other[i])} for i in selected]
    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    for ax in axes.flat:
        ax.axis("off")
    for ax, entry in zip(axes.flat, records):
        ax.imshow(X[entry["test_index"]].reshape(28, 28), cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"Test #{entry['test_index']} | true {entry['true']} -> pred {entry['predicted']}\n"
                     f"Other model: {entry['other_model_predicted']}", fontsize=11)
    fig.suptitle(title, fontsize=15)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return records


def draw_dag(path):
    fig, ax = plt.subplots(figsize=(12, 11))
    positions = {"Y": (0, 0)}
    for i in range(12):
        angle = np.pi / 2 - i * np.pi / 6
        positions[i] = (4.2 * np.cos(angle), 4.2 * np.sin(angle))
    nodes = {}
    for key, (x, y) in positions.items():
        nodes[key] = Ellipse((x, y), 1.95 if key != "Y" else 1.25, .9,
                             facecolor="#e9f2f8" if key != "Y" else "#c9e4d6", edgecolor="#24445e", zorder=3)
        ax.add_patch(nodes[key])
        label = "Y: digit" if key == "Y" else f"F{key + 1}\n{FEATURE_NAMES[key].replace('_', ' ')}"
        if key in (10, 11):
            label = label.replace(" asymmetry", "\nasymmetry")
        ax.text(x, y, label, ha="center", va="center", fontsize=10, zorder=4)
    for i in range(12):
        arrow = FancyArrowPatch(positions["Y"], positions[i], arrowstyle="-|>", mutation_scale=13,
                                color="#7f9fb8", linestyle="--", alpha=.85,
                                patchA=nodes["Y"], patchB=nodes[i], shrinkA=3, shrinkB=3)
        ax.add_patch(arrow)
    for i, parent in enumerate(FEATURE_PARENTS):
        if parent is not None:
            ax.add_patch(FancyArrowPatch(positions[parent], positions[i], arrowstyle="-|>",
                                        mutation_scale=17, color="#d2691e", lw=2.3,
                                        patchA=nodes[parent], patchB=nodes[i], shrinkA=3, shrinkB=3,
                                        connectionstyle="arc3,rad=-0.25"))
    ax.set(xlim=(-5.5, 5.5), ylim=(-5.3, 5.1), aspect="equal",
           title="Manual Bayesian network: 13 nodes, 18 edges")
    ax.text(0, -5.05, "Dashed blue: Y -> every feature | Orange: six within-row feature dependencies",
            ha="center", fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent.parent / "assignment2-data" / "mnist")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()
    figures = args.output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    X_train, y_train, X_test, y_test, hashes = load_data(args.data_dir)
    results = {"train_size": len(y_train), "test_size": len(y_test), "alpha": args.alpha,
               "pixel_threshold": 128, "sha256": hashes,
               "environment": {"python": platform.python_version(), "numpy": np.__version__,
                               "matplotlib": matplotlib.__version__, "platform": platform.platform()},
               "models": {}}
    predictions = {}
    for name, model in (("naive_bayes", NaiveBayesClassifier(args.alpha)), ("bayes_net", BayesNetClassifier(args.alpha))):
        start = time.perf_counter()
        model.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - start
        start = time.perf_counter()
        pred = model.predict(X_test)
        predict_seconds = time.perf_counter() - start
        predictions[name] = pred
        scores = evaluate(y_test, pred)
        scores.update(fit_seconds=fit_seconds, predict_seconds=predict_seconds)
        cm = np.asarray(scores["confusion_matrix"])
        scores["top_confused_pairs"] = confusion_pairs(cm)
        if name == "bayes_net":
            scores["cpt_parent_configurations"] = sum(int(s.size) for s in model.parent_counts_)
            scores["unseen_parent_configurations"] = sum(int((s == 0).sum()) for s in model.parent_counts_)
        results["models"][name] = scores
        draw_confusion(cm, name.replace("_", " ").title(), figures / f"{name}_confusion.png")
        print(f"{name}: accuracy={scores['accuracy']:.4%}, macro F1={scores['macro_f1']:.4%}, "
              f"train={fit_seconds:.3f}s, predict={predict_seconds:.3f}s", flush=True)
    for name, other in (("naive_bayes", "bayes_net"), ("bayes_net", "naive_bayes")):
        results["models"][name]["error_examples"] = draw_errors(
            X_test, y_test, predictions[name], predictions[other], results["models"][name]["top_confused_pairs"],
            name.replace("_", " ").title() + ": misclassified examples", figures / f"{name}_errors.png")
    draw_dag(figures / "bayes_net_dag.png")
    nb_ok = predictions["naive_bayes"] == y_test
    bn_ok = predictions["bayes_net"] == y_test
    results["comparison"] = {"both_correct": int((nb_ok & bn_ok).sum()),
                             "only_nb_correct": int((nb_ok & ~bn_ok).sum()),
                             "only_bn_correct": int((~nb_ok & bn_ok).sum()),
                             "both_wrong": int((~nb_ok & ~bn_ok).sum())}
    (args.output_dir / "metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(args.output_dir / "predictions.npz", y_true=y_test, **predictions)
    print(f"Results saved to {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
