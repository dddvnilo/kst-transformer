from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"


def resolve_path(value: str, base_dir: Path) -> Path:
    """Ako `value` sadrzi separator putanje ili je apsolutna putanja, koristi se kako je data;
    inace se tretira kao ime fajla unutar `base_dir`."""
    path = Path(value)
    if path.is_absolute() or len(path.parts) > 1:
        return path
    return base_dir / value
