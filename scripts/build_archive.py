#!/usr/bin/env python
"""Build curated Internet Archive book corpora using LLM-evaluated filtering.

Searches Internet Archive by subject/collection, evaluates each item for relevance
using a vision-capable LLM (multimodal evaluation of PDF pages), and builds a
quality corpus of on-topic documents.

Uses cursor-based iteration with persistent state for proper resume capability.
Can resume at exact position: which search term, which page within that search.

Usage:
    uv run python build_archive.py \
        --config ../corpus_specs/scientific_illustration.yaml \
        --corpus scientific_illustration \
        --data-dir /mnt/x/rag_datasets/public_domain_books
"""

import argparse
import base64
import json
import logging
import os
import shutil
import sys
import tempfile
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import fitz  # type: ignore[import-untyped]
import yaml
from dotenv import load_dotenv
from internetarchive import download, get_item, search_items
from openai import OpenAI
from pydantic import BaseModel, Field
from tqdm import tqdm

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Rate limiting - be respectful to Internet Archive
SEARCH_DELAY_SECONDS = 0.5
DOWNLOAD_DELAY_SECONDS = 10.0
MAX_RETRIES = 5
SAVE_INTERVAL = 10
ROWS_PER_PAGE = 50


@dataclass
class SearchCursor:
    """Tracks position in Archive search iteration for resume capability.

    The search proceeds through two phases:
    1. "subjects" - Iterating through subject searches
    2. "collections" - Iterating through collection searches

    Within each phase, we track which search term we're on (index),
    which page we're at (page), and offset within that page (offset).
    """

    phase: Literal["subjects", "collections"]
    index: int  # Which item in the current phase's list
    page: int = 1  # Current page number (1-indexed)
    offset: int = 0  # Items processed on current page

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "index": self.index,
            "page": self.page,
            "offset": self.offset,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SearchCursor":
        return cls(
            phase=data["phase"],
            index=data["index"],
            page=data.get("page", 1),
            offset=data.get("offset", 0),
        )


@dataclass
class BuildState:
    """Persistent state for corpus building with full resume capability."""

    corpus_name: str
    cursor: SearchCursor
    accepted: list[dict[str, Any]] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    processed_ids: set[str] = field(default_factory=set)
    total_evaluated: int = 0

    def save(self, state_path: Path) -> None:
        """Save state to disk atomically."""
        data = {
            "corpus_name": self.corpus_name,
            "cursor": self.cursor.to_dict(),
            "accepted": self.accepted,
            "rejected": self.rejected,
            "processed_ids": list(self.processed_ids),
            "total_evaluated": self.total_evaluated,
        }
        tmp_path = state_path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        tmp_path.rename(state_path)

    @classmethod
    def load(cls, state_path: Path) -> "BuildState | None":
        """Load state from disk, or return None if not found."""
        if not state_path.exists():
            return None
        try:
            with state_path.open(encoding="utf-8") as f:
                data = json.load(f)
            return cls(
                corpus_name=data["corpus_name"],
                cursor=SearchCursor.from_dict(data["cursor"]),
                accepted=data.get("accepted", []),
                rejected=data.get("rejected", []),
                processed_ids=set(data.get("processed_ids", [])),
                total_evaluated=data.get("total_evaluated", 0),
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Could not load state: {e}")
            return None


class RelevanceEvaluation(BaseModel):
    """Structured output from the LLM relevance evaluation."""

    relevant: bool = Field(description="Whether the item belongs in this corpus")
    confidence: float = Field(
        description="Confidence in the decision (0.0 to 1.0)", ge=0.0, le=1.0
    )
    reasoning: str = Field(description="Brief explanation of the decision")


EVALUATION_PROMPT = """
You are evaluating whether a document belongs in a specific corpus
based on its VISUAL content.

{validation_prompt}

DOCUMENT INFORMATION:
Title: {title}
Archive ID: {archive_id}
Creator: {creator}
Date: {date}
Subjects: {subjects}
Description: {description}

The images below are sample pages from this document.
Evaluate whether the VISUAL content matches the corpus requirements.

EVALUATION TASK:
Based on the corpus requirements above and the document images provided,
determine if this document belongs in the corpus.

Respond with JSON:
{{"relevant": true/false, "confidence": 0.0-1.0, "reasoning": "brief"}}"""


def extract_pdf_images(
    pdf_path: Path,
    max_images: int = 4,
    max_dimension: int = 1024,
) -> list[str]:
    """Extract sample page images from a PDF, skipping frontmatter.

    Returns base64-encoded PNG images suitable for vision API.

    Args:
        pdf_path: Path to the PDF file.
        max_images: Number of pages to sample.
        max_dimension: Maximum width/height for resized images.

    Returns:
        List of base64-encoded PNG strings.
    """
    images: list[str] = []
    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)

        if total_pages == 0:
            doc.close()
            return images

        # Skip frontmatter: first 10-15% of pages
        skip_pages = max(1, int(total_pages * 0.10))
        content_pages = total_pages - skip_pages

        if content_pages <= 0:
            # Very short document, just sample what we have
            sample_indices = list(range(min(max_images, total_pages)))
        else:
            # Sample evenly from content pages
            if content_pages <= max_images:
                sample_indices = list(range(skip_pages, total_pages))
            else:
                step = content_pages // max_images
                sample_indices = [skip_pages + i * step for i in range(max_images)]

        for page_idx in sample_indices:
            if page_idx >= total_pages:
                continue

            page = doc[page_idx]

            # Render at reasonable resolution
            # Default is 72 DPI, we want ~150 DPI for readability
            zoom = 150 / 72
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)

            # Resize if too large by re-rendering at lower zoom
            if pix.width > max_dimension or pix.height > max_dimension:
                scale = max_dimension / max(pix.width, pix.height)
                zoom = zoom * scale
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat)

            # Convert to PNG bytes
            png_bytes = pix.tobytes("png")

            # Base64 encode
            b64_str = base64.b64encode(png_bytes).decode("utf-8")
            images.append(b64_str)

        doc.close()

    except Exception as e:
        logger.warning(f"Failed to extract images from {pdf_path}: {e}")

    return images


def create_openai_client() -> OpenAI:
    """Create an OpenAI client for vision evaluation."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    base_url = os.environ.get("OPENAI_BASE_URL")

    return OpenAI(api_key=api_key, base_url=base_url)


def evaluate_document(
    pdf_path: Path,
    item_metadata: dict[str, Any],
    validation_prompt: str,
    client: OpenAI,
    model: str = "gpt-5-mini",
    confidence_threshold: float = 0.7,
) -> RelevanceEvaluation | None:
    """Evaluate whether a document is relevant using vision API.

    Returns None if evaluation fails.
    """
    images = extract_pdf_images(pdf_path, max_images=4)

    if not images:
        logger.warning(f"No images extracted from {pdf_path}")
        return None

    creator = item_metadata.get("creator", "Unknown")
    if isinstance(creator, list):
        creator = ", ".join(str(c) for c in creator)

    subjects = item_metadata.get("subjects", [])
    if isinstance(subjects, list):
        subjects = ", ".join(str(s) for s in subjects)

    description = item_metadata.get("description", "")
    if isinstance(description, list):
        description = " ".join(str(d) for d in description)

    prompt_text = EVALUATION_PROMPT.format(
        validation_prompt=validation_prompt,
        title=item_metadata.get("title", "Unknown"),
        archive_id=item_metadata.get("identifier", "Unknown"),
        creator=creator or "Unknown",
        date=item_metadata.get("date", "Unknown"),
        subjects=subjects or "None",
        description=description[:500] if description else "None",
    )

    # Build message with images
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt_text}]

    for img_b64 in images:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img_b64}",
                    "detail": "low",  # Use low detail for cost efficiency
                },
            }
        )

    try:
        # Type ignore: content list structure is correct per OpenAI vision API,
        # but mypy can't infer the union type from dynamic construction
        response = client.chat.completions.create(  # type: ignore[call-overload]
            model=model,
            messages=[{"role": "user", "content": content}],
            response_format={"type": "json_object"},
            max_tokens=500,
        )

        response_text = response.choices[0].message.content
        if not response_text:
            logger.warning(f"Empty response for {item_metadata.get('identifier')}")
            return None

        result_data = json.loads(response_text)
        result = RelevanceEvaluation(**result_data)

        # Low confidence on positive = reject
        if result.relevant and result.confidence < confidence_threshold:
            reason = f"Below threshold ({result.confidence:.2f}). "
            reason += result.reasoning
            return RelevanceEvaluation(
                relevant=False,
                confidence=result.confidence,
                reasoning=reason,
            )

        return result

    except Exception as e:
        logger.warning(
            f"LLM evaluation failed for {item_metadata.get('identifier')}: {e}"
        )
        return None


def load_corpus_config(config_path: Path, corpus_name: str) -> dict[str, Any]:
    """Load corpus configuration from YAML file."""
    with config_path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if corpus_name not in config:
        available = ", ".join(config.keys())
        raise ValueError(f"Corpus '{corpus_name}' not found. Available: {available}")

    corpus = config[corpus_name]

    required = ["source", "search_strategy", "target_count", "validation_prompt"]
    for req_field in required:
        if req_field not in corpus:
            raise ValueError(f"Corpus config missing required field: {req_field}")

    if corpus["source"] != "archive":
        raise ValueError(
            f"Corpus '{corpus_name}' has source '{corpus['source']}', "
            "not 'archive'. Use build_gutenberg.py for gutenberg sources."
        )

    return cast(dict[str, Any], corpus)


def build_single_search_query(
    search_type: Literal["subject", "collection"],
    search_value: str,
    date_range: list[int] | None,
    mediatype: str,
) -> str:
    """Build a search query for a single subject or collection."""
    parts: list[str] = []

    if search_type == "subject":
        parts.append(f'subject:"{search_value}"')
    else:
        parts.append(f'collection:"{search_value}"')

    if date_range and len(date_range) == 2:
        parts.append(f"date:[{date_range[0]} TO {date_range[1]}]")

    parts.append(f"mediatype:{mediatype}")

    return " AND ".join(parts)


def iter_archive_items(
    search_strategy: dict[str, Any],
    cursor: SearchCursor,
    processed_ids: set[str],
) -> Iterator[tuple[dict[str, Any], SearchCursor]]:
    """Iterate through Archive items, yielding (item, updated_cursor).

    Processes search strategy in two phases:
    1. Subject searches
    2. Collection searches

    Each item is yielded with an updated cursor for resume capability.
    """
    subjects = search_strategy.get("subjects", [])
    collections = search_strategy.get("collections", [])
    date_range = search_strategy.get("date_range")
    mediatype = search_strategy.get("mediatype", "texts")

    search_fields = [
        "identifier",
        "title",
        "creator",
        "subject",
        "date",
        "description",
        "downloads",
    ]

    # Determine starting position
    if cursor.phase == "subjects":
        subject_start = cursor.index
        collection_start = 0
    else:
        subject_start = len(subjects)
        collection_start = cursor.index

    # Phase 1: Subject searches
    for i in range(subject_start, len(subjects)):
        subject = subjects[i]
        query = build_single_search_query("subject", subject, date_range, mediatype)

        start_page = (
            cursor.page if i == cursor.index and cursor.phase == "subjects" else 1
        )
        start_offset = (
            cursor.offset if i == cursor.index and cursor.phase == "subjects" else 0
        )

        if start_page > 1 or start_offset > 0:
            logger.info(
                f"Resuming subject '{subject}' "
                f"at page {start_page}, offset {start_offset}"
            )
        else:
            logger.info(f"Searching subject: {subject}")

        current_page = start_page
        items_on_page = 0

        try:
            for item in search_items(
                query,
                fields=search_fields,
                params={"rows": ROWS_PER_PAGE, "page": current_page},
            ):
                time.sleep(SEARCH_DELAY_SECONDS)

                # Skip items until we reach our offset
                if items_on_page < start_offset:
                    items_on_page += 1
                    continue

                # Reset offset after we've caught up
                start_offset = 0

                identifier = item.get("identifier")
                if not identifier or identifier in processed_ids:
                    items_on_page += 1
                    if items_on_page >= ROWS_PER_PAGE:
                        current_page += 1
                        items_on_page = 0
                    continue

                item_data = {
                    "identifier": identifier,
                    "title": item.get("title"),
                    "creator": item.get("creator"),
                    "subjects": item.get("subject", []),
                    "date": item.get("date"),
                    "description": item.get("description"),
                    "downloads": item.get("downloads", 0),
                }

                items_on_page += 1
                new_cursor = SearchCursor(
                    phase="subjects",
                    index=i,
                    page=current_page,
                    offset=items_on_page,
                )

                if items_on_page >= ROWS_PER_PAGE:
                    current_page += 1
                    items_on_page = 0

                yield item_data, new_cursor

        except Exception as e:
            logger.warning(f"Search failed for subject '{subject}': {e}")
            continue

    # Phase 2: Collection searches
    for i in range(collection_start, len(collections)):
        collection = collections[i]
        query = build_single_search_query(
            "collection", collection, date_range, mediatype
        )

        start_page = (
            cursor.page if i == cursor.index and cursor.phase == "collections" else 1
        )
        start_offset = (
            cursor.offset if i == cursor.index and cursor.phase == "collections" else 0
        )

        if start_page > 1 or start_offset > 0:
            logger.info(
                f"Resuming collection '{collection}' "
                f"at page {start_page}, offset {start_offset}"
            )
        else:
            logger.info(f"Searching collection: {collection}")

        current_page = start_page
        items_on_page = 0

        try:
            for item in search_items(
                query,
                fields=search_fields,
                params={"rows": ROWS_PER_PAGE, "page": current_page},
            ):
                time.sleep(SEARCH_DELAY_SECONDS)

                if items_on_page < start_offset:
                    items_on_page += 1
                    continue

                start_offset = 0

                identifier = item.get("identifier")
                if not identifier or identifier in processed_ids:
                    items_on_page += 1
                    if items_on_page >= ROWS_PER_PAGE:
                        current_page += 1
                        items_on_page = 0
                    continue

                item_data = {
                    "identifier": identifier,
                    "title": item.get("title"),
                    "creator": item.get("creator"),
                    "subjects": item.get("subject", []),
                    "date": item.get("date"),
                    "description": item.get("description"),
                    "downloads": item.get("downloads", 0),
                }

                items_on_page += 1
                new_cursor = SearchCursor(
                    phase="collections",
                    index=i,
                    page=current_page,
                    offset=items_on_page,
                )

                if items_on_page >= ROWS_PER_PAGE:
                    current_page += 1
                    items_on_page = 0

                yield item_data, new_cursor

        except Exception as e:
            logger.warning(f"Search failed for collection '{collection}': {e}")
            continue

    logger.info("All searches exhausted")


def get_pdf_file(item_id: str) -> str | None:
    """Find the best PDF file in an Internet Archive item."""
    try:
        item = get_item(item_id)
        files = item.files

        pdf_files = [f for f in files if f.get("name", "").lower().endswith(".pdf")]

        if not pdf_files:
            return None

        # Prefer non-derived PDFs (original scans)
        for pdf in pdf_files:
            source = pdf.get("source", "")
            if source == "original" or not source:
                return cast(str, pdf["name"])

        return cast(str, pdf_files[0]["name"])

    except Exception as e:
        logger.debug(f"Error getting PDF for {item_id}: {e}")
        return None


def download_pdf_file(
    item_id: str,
    pdf_filename: str,
    output_path: Path,
) -> bool:
    """Download a PDF file from Internet Archive."""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        file_list: list[str] = [pdf_filename]
        download(
            item_id,
            files=file_list,  # type: ignore[arg-type]
            destdir=str(output_path.parent.parent),
            no_directory=False,
            retries=MAX_RETRIES,
            verbose=False,
        )

        downloaded_path = output_path.parent.parent / item_id / pdf_filename

        if downloaded_path.exists():
            shutil.move(str(downloaded_path), str(output_path))
            item_dir = output_path.parent.parent / item_id
            if item_dir.exists() and not any(item_dir.iterdir()):
                item_dir.rmdir()
            return True

        return False

    except Exception as e:
        logger.warning(f"Failed to download {item_id}/{pdf_filename}: {e}")
        return False


def write_final_metadata(
    corpus_dir: Path,
    corpus_name: str,
    search_strategy: dict[str, Any],
    accepted: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    total_evaluated: int,
) -> None:
    """Write final corpus metadata file."""
    metadata = {
        "corpus": corpus_name,
        "source": "internet_archive",
        "search_strategy": search_strategy,
        "curated_at": datetime.now(UTC).isoformat(),
        "total_books": len(accepted),
        "books_evaluated": total_evaluated,
        "acceptance_rate": len(accepted) / total_evaluated if total_evaluated else 0,
        "books": accepted,
    }

    with (corpus_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    if rejected:
        rejection_log = {
            "corpus": corpus_name,
            "total_rejected": len(rejected),
            "books": rejected,
        }
        with (corpus_dir / "rejected.json").open("w", encoding="utf-8") as f:
            json.dump(rejection_log, f, indent=2, ensure_ascii=False)


def build_corpus(
    config_path: Path,
    corpus_name: str,
    data_dir: Path,
    limit: int | None = None,
    fresh: bool = False,
) -> None:
    """Build a curated corpus by streaming items and evaluating them.

    Uses cursor-based iteration with persistent state for proper resume.
    """
    corpus_config = load_corpus_config(config_path, corpus_name)

    corpus_dir = data_dir / corpus_name
    books_dir = corpus_dir / "books"
    books_dir.mkdir(parents=True, exist_ok=True)

    state_path = corpus_dir / "build_state.json"
    temp_dir = Path(tempfile.mkdtemp(prefix="archive_build_"))

    target_count = limit if limit is not None else corpus_config["target_count"]
    validation_prompt = corpus_config["validation_prompt"]
    search_strategy = corpus_config["search_strategy"]
    confidence_threshold = corpus_config.get("confidence_threshold", 0.7)
    evaluator_model = corpus_config.get("evaluator_model", "gpt-5-mini")

    # Load or create state
    state: BuildState | None = None
    if not fresh:
        state = BuildState.load(state_path)

    if state is None:
        state = BuildState(
            corpus_name=corpus_name,
            cursor=SearchCursor(phase="subjects", index=0, page=1, offset=0),
        )

    if len(state.accepted) >= target_count:
        logger.info(f"Target already reached: {len(state.accepted)}/{target_count}")
        return

    logger.info(f"Building corpus: {corpus_name}")
    logger.info(f"Target: {target_count} items (have {len(state.accepted)})")
    logger.info(
        f"Resuming from: {state.cursor.phase}[{state.cursor.index}] "
        f"page={state.cursor.page} offset={state.cursor.offset}"
    )

    client = create_openai_client()
    items_since_save = 0

    try:
        item_iter = iter_archive_items(
            search_strategy=search_strategy,
            cursor=state.cursor,
            processed_ids=state.processed_ids,
        )

        pbar = tqdm(desc="Evaluating", unit="item")

        for item, new_cursor in item_iter:
            if len(state.accepted) >= target_count:
                break

            item_id = item["identifier"]
            pbar.set_postfix(
                accepted=len(state.accepted),
                evaluated=state.total_evaluated,
            )

            logger.info(f"Processing {item_id}: {str(item.get('title', ''))[:60]}...")

            # Find PDF file
            pdf_filename = get_pdf_file(item_id)
            if not pdf_filename:
                logger.warning(f"No PDF found for {item_id}, skipping")
                state.processed_ids.add(item_id)
                state.cursor = new_cursor
                continue

            # Download to temp location
            time.sleep(DOWNLOAD_DELAY_SECONDS)
            temp_pdf = temp_dir / f"{item_id}.pdf"
            if not download_pdf_file(item_id, pdf_filename, temp_pdf):
                state.processed_ids.add(item_id)
                state.cursor = new_cursor
                continue

            evaluation = evaluate_document(
                pdf_path=temp_pdf,
                item_metadata=item,
                validation_prompt=validation_prompt,
                client=client,
                model=evaluator_model,
                confidence_threshold=confidence_threshold,
            )

            if evaluation is None:
                temp_pdf.unlink(missing_ok=True)
                state.processed_ids.add(item_id)
                state.cursor = new_cursor
                continue

            state.processed_ids.add(item_id)
            state.cursor = new_cursor
            state.total_evaluated += 1
            items_since_save += 1
            pbar.update(1)

            if evaluation.relevant:
                filename = f"{item_id}.pdf"
                final_path = books_dir / filename
                try:
                    shutil.move(str(temp_pdf), str(final_path))
                    item["file"] = f"books/{filename}"
                    item["source_pdf"] = pdf_filename
                    state.accepted.append(item)
                    logger.info(
                        f"  ACCEPT ({len(state.accepted)}/{target_count}) "
                        f"[{evaluation.confidence:.2f}]: "
                        f"{evaluation.reasoning[:50]}"
                    )
                except OSError as e:
                    logger.error(f"Failed to move file: {e}")
                    temp_pdf.unlink(missing_ok=True)
            else:
                temp_pdf.unlink(missing_ok=True)
                state.rejected.append(
                    {
                        **item,
                        "rejection_reason": evaluation.reasoning,
                        "rejection_confidence": evaluation.confidence,
                    }
                )
                logger.info(
                    f"  REJECT [{evaluation.confidence:.2f}]: "
                    f"{evaluation.reasoning[:60]}"
                )

            # Incremental save
            if items_since_save >= SAVE_INTERVAL:
                state.save(state_path)
                items_since_save = 0

        pbar.close()

        # Final save
        state.save(state_path)

        # Write final metadata
        write_final_metadata(
            corpus_dir=corpus_dir,
            corpus_name=corpus_name,
            search_strategy=search_strategy,
            accepted=state.accepted,
            rejected=state.rejected,
            total_evaluated=state.total_evaluated,
        )

        rate = (
            len(state.accepted) / state.total_evaluated if state.total_evaluated else 0
        )

        logger.info("=" * 60)
        logger.info(f"Build complete: {len(state.accepted)}/{target_count} items")
        logger.info(f"Evaluated: {state.total_evaluated}, Acceptance rate: {rate:.1%}")
        logger.info(f"Output: {corpus_dir}")
        logger.info("=" * 60)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build curated Internet Archive corpus with LLM vision evaluation",
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to corpus config YAML"
    )
    parser.add_argument(
        "--corpus", type=str, required=True, help="Name of corpus to build"
    )
    parser.add_argument(
        "--data-dir", type=Path, default=None, help="Data directory path"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Override target count (for testing)"
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore existing progress and start fresh",
    )

    args = parser.parse_args()

    if not args.config.exists():
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)

    if not os.environ.get("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable not set")
        sys.exit(1)

    data_dir = args.data_dir or Path("/mnt/x/rag_datasets/public_domain_books")
    data_dir.mkdir(parents=True, exist_ok=True)

    try:
        build_corpus(
            config_path=args.config,
            corpus_name=args.corpus,
            data_dir=data_dir,
            limit=args.limit,
            fresh=args.fresh,
        )
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("\nInterrupted")
        sys.exit(130)


if __name__ == "__main__":
    main()
