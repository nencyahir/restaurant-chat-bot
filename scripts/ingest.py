"""Run the ingestion pipeline: clean, chunk, embed and store the dataset in ChromaDB."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from restaurant_bot import config, ingest  # noqa: E402
from restaurant_bot.data_loader import DatasetError  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true",
                        help="Use the full downloaded dataset instead of the bundled 500-restaurant sample.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only ingest the first N restaurants (for quick experiments).")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        path = config.RAW_DATA_PATH if args.full else config.SAMPLE_DATA_PATH
        stats = ingest.run(raw_path=path, limit=args.limit)
    except DatasetError as exc:
        sys.exit(f"Error: {exc}")
    print(f"\nIngestion finished: {stats}")


if __name__ == "__main__":
    main()
