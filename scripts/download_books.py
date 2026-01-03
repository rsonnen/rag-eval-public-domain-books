#!/usr/bin/env python3
"""Download books from existing metadata.

Reads metadata.json and downloads books from Project Gutenberg or Internet Archive.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import httpx
from tqdm import tqdm

# Gutenberg URL patterns to try in order
GUTENBERG_URLS = [
    "https://www.gutenberg.org/ebooks/{id}.txt.utf-8",
    "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt",
    "https://www.gutenberg.org/files/{id}/{id}-0.txt",
    "https://www.gutenberg.org/files/{id}/{id}.txt",
]

# Internet Archive base URL
ARCHIVE_URL = "https://archive.org/download/{identifier}/{filename}"


def download_gutenberg_book(
    client: httpx.Client,
    book_id: int,
    output_path: Path,
) -> bool:
    """Download a book from Project Gutenberg, trying multiple URL patterns."""
    for url_template in GUTENBERG_URLS:
        url = url_template.format(id=book_id)
        try:
            response = client.get(url)
            if response.status_code == 200:
                content = response.text
                # Try UTF-8, fall back to latin-1
                try:
                    output_path.write_text(content, encoding="utf-8")
                except UnicodeEncodeError:
                    output_path.write_text(content, encoding="latin-1")
                return True
        except httpx.HTTPError:
            continue
    return False


def download_archive_book(
    client: httpx.Client,
    identifier: str,
    filename: str,
    output_path: Path,
) -> bool:
    """Download a book from Internet Archive."""
    url = ARCHIVE_URL.format(identifier=identifier, filename=filename)
    try:
        response = client.get(url)
        if response.status_code == 200:
            output_path.write_bytes(response.content)
            return True
    except httpx.HTTPError:
        pass
    return False


def download_corpus(
    corpus_dir: Path,
    delay: float = 1.0,
    max_docs: int | None = None,
) -> None:
    """Download books listed in metadata.json.

    Args:
        corpus_dir: Corpus directory containing metadata.json.
        delay: Seconds to wait between downloads.
        max_docs: Maximum number of books to download (None for all).
    """
    metadata_path = corpus_dir / "metadata.json"
    if not metadata_path.exists():
        print(f"Error: {metadata_path} not found", file=sys.stderr)
        sys.exit(1)

    with open(metadata_path, encoding="utf-8") as f:
        metadata: dict[str, Any] = json.load(f)

    source = metadata.get("source", "gutenberg")
    books: list[dict[str, Any]] = metadata.get("books", [])

    if max_docs is not None:
        books = books[:max_docs]

    print(f"Downloading {len(books)} books from {source} to {corpus_dir}")

    headers = {"User-Agent": "RAG-Eval-Corpus-Downloader/1.0"}
    failed = 0

    with httpx.Client(headers=headers, timeout=120.0, follow_redirects=True) as client:
        for book in tqdm(books, desc="Downloading", unit="book"):
            file_path: str = book.get("file", "")
            if not file_path:
                continue

            output_path = corpus_dir / file_path
            if output_path.exists():
                continue

            output_path.parent.mkdir(parents=True, exist_ok=True)

            if source == "gutenberg":
                book_id: int = book.get("id", 0)
                if not book_id:
                    continue
                success = download_gutenberg_book(client, book_id, output_path)
            else:
                # internet_archive
                identifier: str = book.get("identifier", "")
                filename: str = book.get("source_pdf", "")
                if not identifier or not filename:
                    continue
                success = download_archive_book(
                    client, identifier, filename, output_path
                )

            if not success:
                failed += 1
                tqdm.write(f"Failed: {book.get('title', 'unknown')[:50]}")

            time.sleep(delay)

    if failed:
        print(f"Done ({failed} failed)")
    else:
        print("Done")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Download books from existing metadata"
    )
    parser.add_argument("corpus", help="Corpus name (e.g., shakespeare)")
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between requests in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Maximum number of books to download (default: all)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent
    corpus_dir = repo_root / args.corpus

    if not corpus_dir.exists():
        print(f"Error: Corpus directory not found: {corpus_dir}", file=sys.stderr)
        sys.exit(1)

    download_corpus(corpus_dir, args.delay, args.max_docs)


if __name__ == "__main__":
    main()
