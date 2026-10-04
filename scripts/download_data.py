"""Download the two public datasets into data/raw/ and verify them.

    python -m scripts.download_data

Each file's SHA-256 fingerprint is checked, so everyone who runs this gets byte-identical raw data.
(If a download ever fails the check, upstream changed the file - the numbers would not be comparable.)
"""

import hashlib
import urllib.request
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

BANKING = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master"
CLINC = "https://raw.githubusercontent.com/clinc/oos-eval/master"

# local file name -> (url, expected sha256)
FILES = {
    "banking77_train.csv": (
        f"{BANKING}/banking_data/train.csv",
        "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b",
    ),
    "banking77_test.csv": (
        f"{BANKING}/banking_data/test.csv",
        "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d",
    ),
    "clinc150_data_full.json": (
        f"{CLINC}/data/data_full.json",
        "36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0",
    ),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name, (url, expected) in FILES.items():
        path = RAW_DIR / name
        if not path.exists():
            print(f"downloading {name} ...")
            urllib.request.urlretrieve(url, path)
        if sha256(path) != expected:
            raise SystemExit(f"{name}: checksum mismatch - upstream file changed, delete it and investigate")
        print(f"ok  {name}")


if __name__ == "__main__":
    main()
