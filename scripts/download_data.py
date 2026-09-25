"""Download the Zomato Bangalore restaurants dataset (~575 MB) into data/raw/."""

import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from restaurant_bot import config  # noqa: E402


def main() -> None:
    target = config.RAW_DATA_PATH
    if target.exists():
        print(f"Dataset already present at {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".part")
    print(f"Downloading {config.DATASET_URL}\n  -> {target} (~575 MB)")

    def progress(blocks, block_size, total):
        if total > 0:
            done = min(blocks * block_size, total)
            print(f"\r  {done / 1e6:7.1f} / {total / 1e6:.1f} MB", end="", flush=True)

    try:
        urllib.request.urlretrieve(config.DATASET_URL, tmp, reporthook=progress)
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        sys.exit(f"\nDownload failed: {exc}")
    tmp.rename(target)
    print("\nDone.")


if __name__ == "__main__":
    main()
