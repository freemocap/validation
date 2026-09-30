from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

DATABASE_PATH = REPO_ROOT / "validation.db"
OUTPUT_ROOT = REPO_ROOT / "outputs"

DATASET_ROOT = REPO_ROOT / "freemocap_validation_dataset"
DATA_ROOT = DATASET_ROOT / "data"


def output_dir(*parts: str) -> Path:
    path = OUTPUT_ROOT.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path