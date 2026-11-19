"""Copy the parsed Python docs corpus and dev questions from a local DocLens checkout.

Usage (from the repo root):

    python -m rag_from_scratch.copy_corpus --doclens ../doclens

Only ``data/parsed/python/**.json`` and ``evals/dev.json`` are copied. The held-out
split (``evals/heldout.json``) is deliberately never read or copied, so it stays
unseen during this exercise.
"""

import argparse
import shutil
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"


def copy_corpus(doclens: Path, dest: Path = DATA_DIR) -> tuple[int, Path]:
    parsed = doclens / "data" / "parsed" / "python"
    dev = doclens / "evals" / "dev.json"
    if not parsed.is_dir() or not dev.is_file():
        raise FileNotFoundError(f"{doclens} does not look like a DocLens checkout with parsed data")

    corpus_dest = dest / "corpus"
    count = 0
    for src in sorted(parsed.rglob("*.json")):
        target = corpus_dest / src.relative_to(parsed)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        count += 1
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(dev, dest / "dev.json")
    return count, dest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--doclens", type=Path, required=True, help="path to the DocLens repo")
    args = parser.parse_args()
    count, dest = copy_corpus(args.doclens.resolve())
    print(f"copied {count} parsed pages and dev.json into {dest}")


if __name__ == "__main__":
    main()
