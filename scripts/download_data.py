"""Download the corpora used in the project. Data is not committed - see .gitignore.

    python scripts/download_data.py

Writes data/shakes.txt, data/illiad.txt and their concatenation
data/shakes_illiad.txt, and verifies each file against a known checksum.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

SHAKESPEARE_URL = "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt"
ILIAD_URL = "https://www.gutenberg.org/files/6130/6130-0.txt"

# Checksums of the cleaned text, so they do not depend on how it was transferred.
SHAKESPEARE_SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"
ILIAD_SHA256 = "88c5d9e1fe445bc48656f14e4fe016603545978f58161bfce72f127d9b0d7101"


def download(url: str) -> str:
    """Fetch a text file and normalise its line endings to LF."""
    print(f"downloading {url}")
    with urllib.request.urlopen(url) as response:
        raw = response.read()
    return raw.decode("utf-8-sig").replace("\r\n", "\n")


def clean_gutenberg(text: str) -> str:
    """Strip the Project Gutenberg licence header/footer and the table of contents."""
    start = text.index("\nINTRODUCTION.\n") + 1
    end = re.search(r"\*\*\* END OF (THE|THIS) PROJECT GUTENBERG", text)
    body = text[start : end.start()] if end else text[start:]
    return re.sub(r"\n{3,}", "\n\n", body)


def write_checked(path: Path, text: str, expected_sha256: str) -> None:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if digest != expected_sha256:
        sys.exit(
            f"checksum mismatch for {path.name}\n"
            f"  expected {expected_sha256}\n  got      {digest}\n"
            "The source may have been re-released; compare the text before using it."
        )
    path.write_text(text, encoding="utf-8")
    print(f"{path.name}: ok ({len(text):,} characters)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data", help="output directory")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    shakespeare = download(SHAKESPEARE_URL)
    write_checked(args.out / "shakes.txt", shakespeare, SHAKESPEARE_SHA256)

    iliad = clean_gutenberg(download(ILIAD_URL))
    write_checked(args.out / "illiad.txt", iliad, ILIAD_SHA256)

    # The mixed-domain corpus is derived, so it is generated rather than downloaded.
    combined = args.out / "shakes_illiad.txt"
    combined.write_text(shakespeare + iliad, encoding="utf-8")
    print(f"{combined.name}: generated ({len(shakespeare) + len(iliad):,} characters)")


if __name__ == "__main__":
    main()
