import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class _ProbeStudent(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def _load_probe_tensor(path: Path) -> torch.Tensor:
    if path.suffix.lower() == ".pt":
        loaded = torch.load(str(path), map_location="cpu")
        if isinstance(loaded, dict):
            for key in ("x", "features", "probe_x", "inputs"):
                if key in loaded:
                    loaded = loaded[key]
                    break
        return torch.as_tensor(loaded, dtype=torch.float32)
    if path.suffix.lower() == ".npy":
        import numpy as np

        return torch.from_numpy(np.load(str(path))).float()
    raise ValueError(f"Unsupported probe tensor file type: {path}")


def _load_probe_labels(path: Path) -> torch.Tensor:
    if path.suffix.lower() == ".pt":
        loaded = torch.load(str(path), map_location="cpu")
        if isinstance(loaded, dict):
            for key in ("y", "labels", "probe_y", "targets"):
                if key in loaded:
                    loaded = loaded[key]
                    break
        return torch.as_tensor(loaded, dtype=torch.long)
    if path.suffix.lower() == ".npy":
        import numpy as np

        return torch.from_numpy(np.load(str(path))).long()
    raise ValueError(f"Unsupported probe label file type: {path}")


def _train_student(
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
) -> float:
    num_classes = teacher_logits.shape[-1]
    student = _ProbeStudent(num_classes, hidden_dim, num_classes)
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    dataset = TensorDataset(teacher_logits, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    student.train()
    for _ in range(epochs):
        for x_batch, y_batch in loader:
            logits = student(x_batch)
            loss = criterion(logits, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    student.eval()
    with torch.no_grad():
        preds = student(teacher_logits).argmax(dim=-1)
        acc = (preds == labels).float().mean().item()
    return acc


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Release-time verification: train a small probe student on reference logits "
            "and compare accuracy gain against a baseline on the same probe set."
        )
    )
    parser.add_argument(
        "--reference-logits",
        required=True,
        help="Path to reference model logits or probabilities (.pt/.npy).",
    )
    parser.add_argument(
        "--probe-labels",
        required=True,
        help="Path to probe labels (.pt/.npy).",
    )
    parser.add_argument(
        "--baseline-acc",
        type=float,
        required=True,
        help="Baseline accuracy of the reference model on the same probe set (no student).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Max allowed accuracy gain for gate to pass.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--output-json",
        required=True,
        help="Path to write verification report JSON.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ref_logits = _load_probe_tensor(Path(args.reference_logits).expanduser().resolve())
    probe_labels = _load_probe_labels(Path(args.probe_labels).expanduser().resolve())
    if ref_logits.shape[0] != probe_labels.shape[0]:
        raise ValueError("reference logits and probe labels must share the same first dimension")

    student_acc = _train_student(
        teacher_logits=ref_logits,
        labels=probe_labels,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    delta_acc = student_acc - args.baseline_acc
    passed = delta_acc <= args.threshold
    report = {
        "tool": "release_verify",
        "baseline_acc": args.baseline_acc,
        "student_acc": student_acc,
        "delta_acc": delta_acc,
        "threshold": args.threshold,
        "passed": passed,
        "note": "passed=true means gain is within threshold (lower residual recoverability signal).",
    }
    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
