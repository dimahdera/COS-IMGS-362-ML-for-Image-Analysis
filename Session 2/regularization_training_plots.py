#!/usr/bin/env python3
"""Regularization: Training + Plots (Before vs After)

Python script version of the Jupyter notebook.

What it does:
  - Trains simple models on MNIST to compare:
      (1) Baseline (no regularization)
      (2) L2 / weight decay
      (3) Dropout + BatchNorm (+ small L2/weight decay)
  - Saves figures (loss/accuracy curves) to disk.

Outputs:
  - PNG figures saved to --out_dir (default: ./figs_regularization)

Usage:
  python regularization_training_plots.py --epochs 15 --out_dir figs
  python regularization_training_plots.py --skip_torch
  python regularization_training_plots.py --skip_tf
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def savefig(path: Path) -> None:
    """Save current matplotlib figure and close."""
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def run_tensorflow(epochs: int, batch_size: int, out_dir: Path, seed: int):
    import tensorflow as tf

    print("TensorFlow version:", tf.__version__)
    tf.random.set_seed(seed)
    np.random.seed(seed)

    # Load MNIST
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalize + flatten
    x_train = (x_train.astype("float32") / 255.0).reshape(-1, 784)
    x_test = (x_test.astype("float32") / 255.0).reshape(-1, 784)

    # Train/val split
    val_size = 10000
    x_val, y_val = x_train[:val_size], y_train[:val_size]
    x_tr, y_tr = x_train[val_size:], y_train[val_size:]

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def build_baseline_mlp():
        return tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(784,)),
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dense(10),
            ]
        )

    def build_l2_mlp(l2_strength=1e-4):
        l2 = tf.keras.regularizers.L2(l2_strength)
        return tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(784,)),
                tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=l2),
                tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=l2),
                tf.keras.layers.Dense(10),
            ]
        )

    def build_dropout_bn_mlp(dropout_p=0.3, l2_strength=1e-4):
        l2 = tf.keras.regularizers.L2(l2_strength)
        return tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(784,)),
                tf.keras.layers.Dense(256, use_bias=False, kernel_regularizer=l2),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation("relu"),
                tf.keras.layers.Dropout(dropout_p),
                tf.keras.layers.Dense(256, use_bias=False, kernel_regularizer=l2),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation("relu"),
                tf.keras.layers.Dropout(dropout_p),
                tf.keras.layers.Dense(10),
            ]
        )

    def plot_keras_histories(histories: dict, title_prefix: str):
        # Loss
        plt.figure()
        for name, h in histories.items():
            plt.plot(h.history["loss"], label=f"{name} - train")
            plt.plot(h.history["val_loss"], label=f"{name} - val")
        plt.title(f"{title_prefix} (Loss)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        savefig(out_dir / "tf_loss.png")

        # Accuracy
        plt.figure()
        for name, h in histories.items():
            plt.plot(h.history["accuracy"], label=f"{name} - train")
            plt.plot(h.history["val_accuracy"], label=f"{name} - val")
        plt.title(f"{title_prefix} (Accuracy)")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        savefig(out_dir / "tf_accuracy.png")

    baseline = build_baseline_mlp()
    baseline.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=loss_fn, metrics=["accuracy"])

    l2_model = build_l2_mlp(l2_strength=1e-4)
    l2_model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=loss_fn, metrics=["accuracy"])

    drop_bn = build_dropout_bn_mlp(dropout_p=0.3, l2_strength=1e-4)
    drop_bn.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
        loss=loss_fn,
        metrics=["accuracy"],
    )

    histories = {}
    print("\n[TF] Training Baseline...")
    histories["Baseline"] = baseline.fit(
        x_tr, y_tr, validation_data=(x_val, y_val), epochs=epochs, batch_size=batch_size, verbose=2
    )

    print("\n[TF] Training L2...")
    histories["L2"] = l2_model.fit(
        x_tr, y_tr, validation_data=(x_val, y_val), epochs=epochs, batch_size=batch_size, verbose=2
    )

    print("\n[TF] Training Dropout+BN...")
    histories["Dropout+BN"] = drop_bn.fit(
        x_tr, y_tr, validation_data=(x_val, y_val), epochs=epochs, batch_size=batch_size, verbose=2
    )

    plot_keras_histories(histories, "Keras MNIST: Baseline vs Regularization")

    def eval_model(model, name):
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        print(f"[TF] {name:12s} | test loss: {test_loss:.4f} | test acc: {test_acc:.4f}")
        return test_loss, test_acc

    return {
        "Baseline": eval_model(baseline, "Baseline"),
        "L2": eval_model(l2_model, "L2"),
        "Dropout+BN": eval_model(drop_bn, "Dropout+BN"),
    }


def run_pytorch(
    epochs: int,
    batch_size_train: int,
    batch_size_test: int,
    out_dir: Path,
    seed: int,
    subset_n: int | None,
):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, transforms

    print("PyTorch version:", torch.__version__)
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1)),  # flatten 28x28 -> 784
        ]
    )

    train_ds = datasets.MNIST(root=".", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root=".", train=False, download=True, transform=transform)

    if subset_n is not None and subset_n > 0:
        train_ds = Subset(train_ds, list(range(min(subset_n, len(train_ds)))))

    train_loader = DataLoader(train_ds, batch_size=batch_size_train, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size_test, shuffle=False)

    class TorchMLP(nn.Module):
        def __init__(self, use_dropout=False, p=0.3, use_bn=False):
            super().__init__()
            layers = []
            layers.append(nn.Linear(784, 256, bias=not use_bn))
            if use_bn:
                layers.append(nn.BatchNorm1d(256))
            layers.append(nn.ReLU())
            if use_dropout:
                layers.append(nn.Dropout(p=p))

            layers.append(nn.Linear(256, 256, bias=not use_bn))
            if use_bn:
                layers.append(nn.BatchNorm1d(256))
            layers.append(nn.ReLU())
            if use_dropout:
                layers.append(nn.Dropout(p=p))

            layers.append(nn.Linear(256, 10))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    def accuracy_from_logits(logits, y):
        preds = logits.argmax(dim=1)
        return (preds == y).float().mean().item()

    def train_one_epoch(model, loader, optimizer, criterion):
        model.train()
        total_loss, total_acc, n = 0.0, 0.0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            bs = x.size(0)
            total_loss += loss.item() * bs
            total_acc += accuracy_from_logits(logits, y) * bs
            n += bs
        return total_loss / n, total_acc / n

    @torch.no_grad()
    def eval_epoch(model, loader, criterion):
        model.eval()
        total_loss, total_acc, n = 0.0, 0.0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            bs = x.size(0)
            total_loss += loss.item() * bs
            total_acc += accuracy_from_logits(logits, y) * bs
            n += bs
        return total_loss / n, total_acc / n

    def run_training(name, model, optimizer, epochs=6):
        criterion = nn.CrossEntropyLoss()
        model = model.to(device)
        history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

        for ep in range(epochs):
            tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion)
            te_loss, te_acc = eval_epoch(model, test_loader, criterion)
            history["train_loss"].append(tr_loss)
            history["train_acc"].append(tr_acc)
            history["test_loss"].append(te_loss)
            history["test_acc"].append(te_acc)
            print(
                f"[TORCH] {name} | epoch {ep+1}/{epochs} | "
                f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
                f"test loss {te_loss:.4f} acc {te_acc:.4f}"
            )
        return history

    torch_baseline = TorchMLP(use_dropout=False, use_bn=False)
    opt_base = optim.Adam(torch_baseline.parameters(), lr=1e-3)

    torch_wd = TorchMLP(use_dropout=False, use_bn=False)
    opt_wd = optim.AdamW(torch_wd.parameters(), lr=1e-3, weight_decay=1e-4)

    torch_drop_bn = TorchMLP(use_dropout=True, p=0.3, use_bn=True)
    opt_drop_bn = optim.AdamW(torch_drop_bn.parameters(), lr=1e-3, weight_decay=1e-4)

    histories = {}
    histories["Baseline"] = run_training("Baseline", torch_baseline, opt_base, epochs=epochs)
    histories["WeightDecay"] = run_training("WeightDecay", torch_wd, opt_wd, epochs=epochs)
    histories["Dropout+BN"] = run_training("Dropout+BN", torch_drop_bn, opt_drop_bn, epochs=epochs)

    # Loss plot
    plt.figure()
    for name, h in histories.items():
        plt.plot(h["train_loss"], label=f"{name} - train")
        plt.plot(h["test_loss"], label=f"{name} - test")
    plt.title("PyTorch MNIST: Baseline vs Regularization (Loss)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    savefig(out_dir / "torch_loss.png")

    # Accuracy plot
    plt.figure()
    for name, h in histories.items():
        plt.plot(h["train_acc"], label=f"{name} - train")
        plt.plot(h["test_acc"], label=f"{name} - test")
    plt.title("PyTorch MNIST: Baseline vs Regularization (Accuracy)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    savefig(out_dir / "torch_accuracy.png")

    return {name: (h["test_loss"][-1], h["test_acc"][-1]) for name, h in histories.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8, help="Number of epochs for each run.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training.")
    parser.add_argument("--out_dir", type=str, default="figs_regularization", help="Directory to save figures.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--skip_tf", action="store_true", help="Skip TensorFlow section.")
    parser.add_argument("--skip_torch", action="store_true", help="Skip PyTorch section.")
    parser.add_argument(
        "--torch_subset_n",
        type=int,
        default=20000,
        help="Use subset of MNIST train set for PyTorch (<=0 = full train set).",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    if not args.skip_tf:
        try:
            all_results["tensorflow"] = run_tensorflow(args.epochs, args.batch_size, out_dir, args.seed)
        except Exception as e:
            print("[TF] Error:", repr(e))
            print("[TF] If TensorFlow is not installed in this environment, run with --skip_tf")

    if not args.skip_torch:
        try:
            subset_n = None if args.torch_subset_n <= 0 else args.torch_subset_n
            all_results["pytorch"] = run_pytorch(
                args.epochs, args.batch_size, 256, out_dir, args.seed, subset_n
            )
        except Exception as e:
            print("[TORCH] Error:", repr(e))
            print("[TORCH] If PyTorch/torchvision is not installed, run with --skip_torch")

    # Save summary
    summary_path = out_dir / "results_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Regularization comparison results\n")
        f.write(f"Epochs: {args.epochs}\nBatch size: {args.batch_size}\n\n")
        for section, res in all_results.items():
            f.write(f"[{section}]\n")
            for name, (loss, acc) in res.items():
                f.write(f"{name:12s}  loss={loss:.4f}  acc={acc:.4f}\n")
            f.write("\n")

    print("\nSaved figures to:", out_dir.resolve())
    print("Saved summary to:", summary_path.resolve())


if __name__ == "__main__":
    main()
