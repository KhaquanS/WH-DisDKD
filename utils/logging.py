import csv
from pathlib import Path


class LossTracker:
    """Tracks and logs all training/validation losses and metrics."""

    def __init__(self, log_path: str, method: str):
        self.log_path = Path(log_path)
        self.method = method
        self.log_data = []

        # Initialize CSV with headers
        headers = ["epoch", "phase", "total_loss", "ce_loss", "kd_loss", "accuracy"]

        # Add method-specific loss columns
        method_headers = {
            "FitNet": ["hint_loss"],
            "CRD": ["contrastive_loss"],
            "DKD": ["tckd_loss", "nckd_loss"],
            "DisDKD": [
                "dkd_loss",
                "discriminator_loss",
                "adversarial_loss",
                "disc_accuracy",
                "fool_rate",
            ],
        }

        if method in method_headers:
            headers.extend(method_headers[method])

        self.headers = headers
        self._write_headers()

    def _write_headers(self):
        """Write CSV headers."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)

    def log_epoch(self, epoch: int, phase: str, losses: dict, accuracy: float):
        """Log losses and metrics for an epoch."""
        row = [
            epoch,
            phase,
            losses.get("total", 0),
            losses.get("ce", 0),
            losses.get("kd", 0),
            accuracy,
        ]

        # Add method-specific losses
        method_losses = {
            "FitNet": ["hint"],
            "CRD": ["contrastive"],
            "DKD": ["tckd", "nckd"],
            "DisDKD": [
                "dkd",
                "discriminator",
                "adversarial",
                "disc_accuracy",
                "fool_rate",
            ],
        }

        if self.method in method_losses:
            for loss_key in method_losses[self.method]:
                row.append(losses.get(loss_key, 0))

        # Append to CSV
        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)
