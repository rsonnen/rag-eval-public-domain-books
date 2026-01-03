#!/usr/bin/env python
"""Build curated Project Gutenberg book corpora using LLM-evaluated filtering.

Searches Project Gutenberg by topic/subject, evaluates each book for relevance
using an LLM-as-judge, and builds a quality corpus of on-topic books.

Uses cursor-based iteration with persistent state for proper resume capability.
Can resume at exact position: which search query, which page within that query.

Usage:
    uv run python build_gutenberg.py \
        --config ../corpus_specs/gothic_fiction.yaml \
        --corpus gothic_fiction \
        --data-dir /mnt/x/rag_datasets/public_domain_books
"""

import argparse
import contextlib
import json
import logging
import os
import random
import shutil
import sys
import tempfile
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import httpx
import yaml
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, SecretStr, ValidationError
from tqdm import tqdm

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

GUTENDEX_API_URL = "https://gutendex.com/books"

# Rate limiting - Gutendex is a free service, be respectful
BASE_DELAY_SECONDS = 1.0
SEARCH_DELAY_SECONDS = 1.0
DOWNLOAD_DELAY_SECONDS = 2.0
MAX_RETRIES = 5
BACKOFF_FACTOR = 2.0
MAX_BACKOFF_SECONDS = 120
SAVE_INTERVAL = 10


@dataclass
class SearchCursor:
    """Tracks position in Gutenberg search iteration for resume capability.

    The search proceeds through two phases:
    1. "topics" - Iterating through topic searches (bookshelves/subjects)
    2. "searches" - Iterating through author/title text searches

    Within each phase, we track which search term we're on (index) and
    which pagination URL we're at (page_url).
    """

    phase: Literal["topics", "searches"]
    index: int  # Which item in the current phase's list
    page_url: str | None = None  # Next page URL, None = start fresh

    def to_dict(self) -> dict[str, Any]:
        return {"phase": self.phase, "index": self.index, "page_url": self.page_url}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SearchCursor":
        return cls(
            phase=data["phase"],
            index=data["index"],
            page_url=data.get("page_url"),
        )


@dataclass
class BuildState:
    """Persistent state for corpus building with full resume capability."""

    corpus_name: str
    cursor: SearchCursor
    accepted: list[dict[str, Any]] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    processed_ids: set[int] = field(default_factory=set)
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
        # Write atomically via temp file
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

    relevant: bool = Field(description="Whether the book belongs in this corpus")
    confidence: float = Field(
        description="Confidence in the decision (0.0 to 1.0)", ge=0.0, le=1.0
    )
    reasoning: str = Field(description="Brief explanation of the decision")


EVALUATION_PROMPT = """
You are evaluating whether a book belongs in a specific literary corpus.

{validation_prompt}

BOOK INFORMATION:
Title: {title}
Gutenberg ID: {gutenberg_id}
Authors: {authors}
Subjects: {subjects}
Bookshelves: {bookshelves}
Languages: {languages}

Text excerpt (first ~4000 characters):
{text_excerpt}

EVALUATION TASK:
Based on the corpus requirements above and the book information provided,
determine if this book belongs in the corpus.

Respond with:
- relevant: true/false
- confidence: your confidence in this decision (0.0 to 1.0)
- reasoning: brief explanation (1-2 sentences)"""


def request_with_retry(
    client: httpx.Client,
    url: str,
    params: dict[str, Any] | None = None,
) -> httpx.Response:
    """Make an HTTP request with exponential backoff on rate limit/server errors."""
    delay = BASE_DELAY_SECONDS
    last_exception: Exception | None = None

    for attempt in range(MAX_RETRIES + 1):
        if attempt > 0:
            jitter = random.uniform(0, delay * 0.1)  # noqa: S311 - not crypto
            sleep_time = delay + jitter
            logger.info(f"Retry {attempt}/{MAX_RETRIES}, waiting {sleep_time:.1f}s")
            time.sleep(sleep_time)
            delay = min(delay * BACKOFF_FACTOR, MAX_BACKOFF_SECONDS)
        else:
            time.sleep(BASE_DELAY_SECONDS)

        try:
            response = client.get(url, params=params, follow_redirects=True)

            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                if retry_after:
                    with contextlib.suppress(ValueError):
                        delay = max(float(retry_after), delay)
                last_exception = httpx.HTTPStatusError(
                    "Rate limited (429)",
                    request=response.request,
                    response=response,
                )
                continue

            if response.status_code >= 500:
                last_exception = httpx.HTTPStatusError(
                    f"Server error ({response.status_code})",
                    request=response.request,
                    response=response,
                )
                continue

            response.raise_for_status()
            return response

        except httpx.TimeoutException as e:
            last_exception = e
            logger.warning(f"Timeout: {e}")
            continue
        except httpx.RequestError as e:
            last_exception = e
            logger.warning(f"Request failed: {e}")
            continue

    if last_exception:
        raise last_exception
    raise httpx.HTTPError("All retries exhausted")


def create_evaluator(
    model_name: str = "gpt-5-mini",
    temperature: float = 0.0,
) -> ChatOpenAI:
    """Create an LLM instance for book evaluation."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    base_url = os.environ.get("OPENAI_BASE_URL")

    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        api_key=SecretStr(api_key),
        base_url=base_url,
    )


def evaluate_book(
    text_content: str,
    book_metadata: dict[str, Any],
    validation_prompt: str,
    llm: ChatOpenAI,
    confidence_threshold: float = 0.7,
) -> RelevanceEvaluation | None:
    """Evaluate whether a book is relevant to the corpus topic.

    Returns None if LLM call fails.
    """
    authors = ", ".join(
        a.get("name", "Unknown") for a in book_metadata.get("authors", [])
    )
    subjects = ", ".join(book_metadata.get("subjects", []))
    bookshelves = ", ".join(book_metadata.get("bookshelves", []))
    languages = ", ".join(book_metadata.get("languages", []))

    prompt = EVALUATION_PROMPT.format(
        validation_prompt=validation_prompt,
        title=book_metadata.get("title", "Unknown"),
        gutenberg_id=book_metadata.get("id", "Unknown"),
        authors=authors or "Unknown",
        subjects=subjects or "None",
        bookshelves=bookshelves or "None",
        languages=languages or "Unknown",
        text_excerpt=text_content[:4000],
    )

    try:
        structured_llm = llm.with_structured_output(RelevanceEvaluation)
        raw_result = structured_llm.invoke(prompt)

        if raw_result is None:
            logger.warning(f"LLM returned None for {book_metadata.get('id')}")
            return None

        result = cast(RelevanceEvaluation, raw_result)

        # Low confidence on positive = reject
        if result.relevant and result.confidence < confidence_threshold:
            return RelevanceEvaluation(
                relevant=False,
                confidence=result.confidence,
                reasoning=f"Below threshold ({result.confidence:.2f}). "
                f"{result.reasoning}",
            )

        return result

    except (ValidationError, Exception) as e:
        logger.warning(f"LLM evaluation failed for {book_metadata.get('id')}: {e}")
        return None


def get_text_urls(book_id: int) -> list[str]:
    """Generate URLs to try for downloading plain text, in priority order.

    Project Gutenberg has multiple URL patterns for text files. We try them
    in order of reliability rather than trusting the API's formats field,
    which can point to readme files for audiobooks.
    """
    return [
        f"https://www.gutenberg.org/ebooks/{book_id}.txt.utf-8",
        f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}.txt",
    ]


def has_text_format(formats: dict[str, str]) -> bool:
    """Check if the book has any plain text format available."""
    return any(mime.startswith("text/plain") for mime in formats)


def strip_gutenberg_headers(text: str) -> str:
    """Remove Project Gutenberg header and footer boilerplate."""
    start_markers = [
        "*** START OF THE PROJECT GUTENBERG EBOOK",
        "*** START OF THIS PROJECT GUTENBERG EBOOK",
        "***START OF THE PROJECT GUTENBERG EBOOK",
    ]
    end_markers = [
        "*** END OF THE PROJECT GUTENBERG EBOOK",
        "*** END OF THIS PROJECT GUTENBERG EBOOK",
        "***END OF THE PROJECT GUTENBERG EBOOK",
    ]

    # Find content start
    start_pos = 0
    for marker in start_markers:
        if marker in text:
            pos = text.find(marker)
            # Move past the marker line
            newline_pos = text.find("\n", pos)
            if newline_pos != -1:
                start_pos = newline_pos + 1
            break

    # Find content end
    end_pos = len(text)
    for marker in end_markers:
        if marker in text:
            end_pos = text.rfind(marker)
            break

    return text[start_pos:end_pos].strip()


def load_corpus_config(config_path: Path, corpus_name: str) -> dict[str, Any]:
    """Load corpus configuration from YAML file."""
    with config_path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if corpus_name not in config:
        available = ", ".join(config.keys())
        raise ValueError(f"Corpus '{corpus_name}' not found. Available: {available}")

    corpus = config[corpus_name]

    # Validate required fields for new format
    required = ["source", "search_strategy", "target_count", "validation_prompt"]
    for req_field in required:
        if req_field not in corpus:
            raise ValueError(f"Corpus config missing required field: {req_field}")

    if corpus["source"] != "gutenberg":
        raise ValueError(
            f"Corpus '{corpus_name}' has source '{corpus['source']}', "
            "not 'gutenberg'. Use build_archive.py for archive sources."
        )

    return cast(dict[str, Any], corpus)


def _build_search_params(
    topic: str | None,
    search: str | None,
    languages: list[str],
    author_years: tuple[int, int] | None,
) -> dict[str, str]:
    """Build query parameters for Gutendex API."""
    params: dict[str, str] = {"languages": ",".join(languages)}

    if topic:
        params["topic"] = topic
    if search:
        params["search"] = search
    if author_years:
        params["author_year_start"] = str(author_years[0])
        params["author_year_end"] = str(author_years[1])

    return params


def _parse_book(book: dict[str, Any]) -> dict[str, Any] | None:
    """Parse a book from API response, filtering non-text content.

    Returns None if book should be skipped (audiobook, no text format).
    """
    book_id = book.get("id")
    if not book_id:
        return None

    # Filter: must be Text media type (not Sound/audiobook)
    media_type = book.get("media_type", "")
    if media_type != "Text":
        return None

    # Must have text/plain format available
    if not has_text_format(book.get("formats", {})):
        return None

    return {
        "id": book_id,
        "title": book.get("title"),
        "authors": book.get("authors", []),
        "subjects": book.get("subjects", []),
        "bookshelves": book.get("bookshelves", []),
        "languages": book.get("languages", []),
        "download_count": book.get("download_count", 0),
    }


def _iter_search_results(
    client: httpx.Client,
    topic: str | None = None,
    search: str | None = None,
    languages: list[str] | None = None,
    author_years: tuple[int, int] | None = None,
    start_page_url: str | None = None,
) -> Iterator[tuple[dict[str, Any], str | None]]:
    """Iterate through paginated search results, yielding (book, next_page_url).

    Args:
        client: HTTP client
        topic: Topic string to search (bookshelves/subjects)
        search: Search string for author names/titles
        languages: Language filter
        author_years: Optional (start, end) tuple for author birth years
        start_page_url: Resume from this pagination URL (None = start fresh)

    Yields:
        Tuple of (book_metadata, next_page_url). The next_page_url is provided
        with each book so the caller can persist cursor state.
    """
    languages = languages or ["en"]
    params = _build_search_params(topic, search, languages, author_years)

    # Determine starting URL
    if start_page_url:
        next_url: str | None = start_page_url
        use_params = False  # Pagination URLs include params
    else:
        next_url = GUTENDEX_API_URL
        use_params = True

    while next_url:
        time.sleep(SEARCH_DELAY_SECONDS)

        try:
            if use_params:
                response = request_with_retry(client, next_url, params=params)
                use_params = False  # Only use params on first request
            else:
                response = request_with_retry(client, next_url)

            data = response.json()
        except (httpx.HTTPError, json.JSONDecodeError) as e:
            search_term = topic or search
            logger.warning(f"Search failed for '{search_term}': {e}")
            return

        next_url = data.get("next")

        for raw_book in data.get("results", []):
            book = _parse_book(raw_book)
            if book:
                yield book, next_url


def iter_gutenberg_books(
    client: httpx.Client,
    search_strategy: dict[str, Any],
    cursor: SearchCursor,
    processed_ids: set[int],
) -> Iterator[tuple[dict[str, Any], SearchCursor]]:
    """Iterate through Gutenberg books, yielding (book, updated_cursor).

    Processes search strategy in two phases:
    1. Topic searches (bookshelves/subjects)
    2. Author/title text searches

    Each book is yielded with an updated cursor that can be persisted for
    resume capability. The cursor tracks exactly which search and page we're at.

    Args:
        client: HTTP client
        search_strategy: Dict with topics, searches, languages, author_years
        cursor: Starting position (which search, which page)
        processed_ids: Set of book IDs to skip (already processed)

    Yields:
        Tuple of (book_metadata, updated_cursor). Cursor reflects position
        AFTER yielding this book (ready for next iteration).
    """
    topics = search_strategy.get("topics", [])
    searches = search_strategy.get("searches", [])
    languages = search_strategy.get("languages", ["en"])
    author_years_raw = search_strategy.get("author_years")
    author_years = tuple(author_years_raw) if author_years_raw else None

    # Determine starting position
    if cursor.phase == "topics":
        topic_start = cursor.index
        search_start = 0
    else:
        topic_start = len(topics)  # Skip topics entirely
        search_start = cursor.index

    # Phase 1: Topic searches
    for i in range(topic_start, len(topics)):
        topic = topics[i]
        start_page = cursor.page_url if i == cursor.index else None

        if start_page:
            logger.info(f"Resuming topic: {topic}")
        else:
            logger.info(f"Searching topic: {topic}")

        for book, next_page in _iter_search_results(
            client,
            topic=topic,
            languages=languages,
            author_years=author_years,
            start_page_url=start_page,
        ):
            if book["id"] in processed_ids:
                continue

            # Cursor points to current position (this topic, next page)
            new_cursor = SearchCursor(phase="topics", index=i, page_url=next_page)
            yield book, new_cursor

    # Phase 2: Author/title text searches
    for i in range(search_start, len(searches)):
        search_term = searches[i]
        start_page = (
            cursor.page_url
            if cursor.phase == "searches" and i == cursor.index
            else None
        )

        if start_page:
            logger.info(f"Resuming search: {search_term}")
        else:
            logger.info(f"Searching: {search_term}")

        for book, next_page in _iter_search_results(
            client,
            search=search_term,
            languages=languages,
            author_years=author_years,
            start_page_url=start_page,
        ):
            if book["id"] in processed_ids:
                continue

            new_cursor = SearchCursor(phase="searches", index=i, page_url=next_page)
            yield book, new_cursor

    logger.info("All searches exhausted")


def download_text(
    client: httpx.Client,
    book_id: int,
    output_path: Path,
) -> bool:
    """Download plain text for a book, trying multiple URL patterns.

    Tries URLs in priority order with delays between attempts.
    Strips Gutenberg headers/footers before saving.
    """
    urls = get_text_urls(book_id)

    for url in urls:
        time.sleep(DOWNLOAD_DELAY_SECONDS)
        try:
            response = request_with_retry(client, url)

            # Decode content
            try:
                text = response.content.decode("utf-8")
            except UnicodeDecodeError:
                text = response.content.decode("latin-1")

            # Verify we got actual book content, not an error page
            if len(text) < 1000:
                logger.debug(f"Content too short from {url}, trying next")
                continue

            # Strip Gutenberg boilerplate
            text = strip_gutenberg_headers(text)

            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(text, encoding="utf-8")
            return True

        except httpx.HTTPError as e:
            logger.debug(f"Failed to download from {url}: {e}")
            continue

    logger.warning(f"Failed to download book {book_id} from any URL")
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
        "source": "gutenberg",
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
    """Build a curated corpus by streaming books and evaluating them.

    Uses cursor-based iteration with persistent state for proper resume.
    Can resume at exact position: which search query, which page.

    Args:
        config_path: Path to corpus config YAML.
        corpus_name: Name of corpus to build.
        data_dir: Base data directory.
        limit: Override target count (for testing).
        fresh: If True, ignore existing progress and start fresh.
    """
    corpus_config = load_corpus_config(config_path, corpus_name)

    corpus_dir = data_dir / corpus_name
    books_dir = corpus_dir / "books"
    books_dir.mkdir(parents=True, exist_ok=True)

    state_path = corpus_dir / "build_state.json"
    temp_dir = Path(tempfile.mkdtemp(prefix="gutenberg_build_"))

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
        # Start fresh from beginning of topics
        state = BuildState(
            corpus_name=corpus_name,
            cursor=SearchCursor(phase="topics", index=0, page_url=None),
        )

    if len(state.accepted) >= target_count:
        logger.info(f"Target already reached: {len(state.accepted)}/{target_count}")
        return

    logger.info(f"Building corpus: {corpus_name}")
    logger.info(f"Target: {target_count} books (have {len(state.accepted)})")
    logger.info(
        f"Resuming from: {state.cursor.phase}[{state.cursor.index}] "
        f"page={state.cursor.page_url is not None}"
    )

    headers = {
        "User-Agent": "BiteSizeRAG-Corpus-Builder/2.0 (literary research)",
    }

    books_since_save = 0

    try:
        with httpx.Client(headers=headers, timeout=60.0) as client:
            llm = create_evaluator(model_name=evaluator_model)

            book_iter = iter_gutenberg_books(
                client=client,
                search_strategy=search_strategy,
                cursor=state.cursor,
                processed_ids=state.processed_ids,
            )

            pbar = tqdm(desc="Evaluating", unit="book")

            for book, new_cursor in book_iter:
                if len(state.accepted) >= target_count:
                    break

                book_id = book["id"]
                pbar.set_postfix(
                    accepted=len(state.accepted),
                    evaluated=state.total_evaluated,
                )

                logger.info(f"Downloading {book_id}: {book['title'][:60]}...")

                temp_txt = temp_dir / f"{book_id}.txt"
                if not download_text(client, book_id, temp_txt):
                    state.processed_ids.add(book_id)
                    state.cursor = new_cursor
                    continue

                # Read text for evaluation
                try:
                    text_content = temp_txt.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    text_content = temp_txt.read_text(encoding="latin-1")

                evaluation = evaluate_book(
                    text_content=text_content,
                    book_metadata=book,
                    validation_prompt=validation_prompt,
                    llm=llm,
                    confidence_threshold=confidence_threshold,
                )

                # Handle evaluation failure
                if evaluation is None:
                    temp_txt.unlink(missing_ok=True)
                    state.processed_ids.add(book_id)
                    state.cursor = new_cursor
                    continue

                state.processed_ids.add(book_id)
                state.cursor = new_cursor
                state.total_evaluated += 1
                books_since_save += 1
                pbar.update(1)

                if evaluation.relevant:
                    filename = f"{book_id}.txt"
                    final_path = books_dir / filename
                    try:
                        shutil.move(str(temp_txt), str(final_path))
                        book["file"] = f"books/{filename}"
                        state.accepted.append(book)
                        logger.info(
                            f"  ACCEPT ({len(state.accepted)}/{target_count}) "
                            f"[{evaluation.confidence:.2f}]: "
                            f"{evaluation.reasoning[:50]}"
                        )
                    except OSError as e:
                        logger.error(f"Failed to move file: {e}")
                        temp_txt.unlink(missing_ok=True)
                else:
                    temp_txt.unlink(missing_ok=True)
                    state.rejected.append(
                        {
                            **book,
                            "rejection_reason": evaluation.reasoning,
                            "rejection_confidence": evaluation.confidence,
                        }
                    )
                    logger.info(
                        f"  REJECT [{evaluation.confidence:.2f}]: "
                        f"{evaluation.reasoning[:60]}"
                    )

                # Incremental save
                if books_since_save >= SAVE_INTERVAL:
                    state.save(state_path)
                    books_since_save = 0

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
        logger.info(f"Build complete: {len(state.accepted)}/{target_count} books")
        logger.info(f"Evaluated: {state.total_evaluated}, Acceptance rate: {rate:.1%}")
        logger.info(f"Output: {corpus_dir}")
        logger.info("=" * 60)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build curated Project Gutenberg corpus with LLM evaluation",
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
