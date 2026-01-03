# rag-eval-public-domain-books

Evaluation corpus of public domain literary works for testing RAG systems.

## What This Is

This repository contains **evaluation data for RAG systems**:

- **corpus.yaml** - Evaluation scenarios (in each corpus directory)
- **metadata.json** - Book inventory with source IDs
- **Generated questions** - Validated Q/A pairs (where available)

The actual book files are not included. Use `download_books.py` to fetch them.

## Quick Start

```bash
cd scripts
uv sync
uv run python download_books.py shakespeare --max-docs 5
```

## Available Corpora

### Project Gutenberg (Plain Text)

| Corpus | Books | Description |
|--------|-------|-------------|
| `shakespeare` | 50 | Complete plays, sonnets, poems |
| `victorian_detective` | 50 | Doyle, Collins, Poe detective fiction |
| `early_american` | 50 | Hawthorne, Melville, Poe, Twain |
| `gothic_fiction` | 50 | Shelley, Stoker, Poe, Radcliffe |
| `philosophy_19c` | 50 | Nietzsche, Mill, Kant, Schopenhauer |
| `early_scifi` | 50 | Verne, Wells science fiction |

### Internet Archive (PDF Scans)

| Corpus | Books | Description |
|--------|-------|-------------|
| `scientific_illustration` | 100 | Natural history, botanical, anatomical works |
| `industrial_age_technical` | 100 | Engineering, machinery, manufacturing |

All corpora were built December 2025.

## Directory Structure

```
<corpus>/
    corpus.yaml         # Evaluation configuration
    metadata.json       # Book inventory
    books/              # Downloaded files (gitignored)

scripts/
    download_books.py   # Fetch books from existing metadata
    build_gutenberg.py  # Build new Gutenberg corpora with LLM curation
    build_archive.py    # Build new Archive corpora with LLM curation

corpus_specs/
    *.yaml              # Build configurations
```

## Downloading Books

The download script reads metadata.json and fetches books from the appropriate source:

```bash
cd scripts
uv run python download_books.py shakespeare --max-docs 10
uv run python download_books.py scientific_illustration
```

| Option | Description |
|--------|-------------|
| `corpus` | Corpus name (e.g., shakespeare) |
| `--max-docs` | Maximum books to download (default: all) |
| `--delay` | Delay between requests in seconds (default: 1.0) |

## Building New Corpora

The build scripts search sources and curate using LLM evaluation:

```bash
cd scripts
# Gutenberg corpus
uv run python build_gutenberg.py \
    --config ../corpus_specs/gothic_fiction.yaml \
    --corpus gothic_fiction

# Internet Archive corpus
uv run python build_archive.py \
    --config ../corpus_specs/scientific_illustration.yaml \
    --corpus scientific_illustration
```

## Licensing

**This repository**: MIT License

**Books**:
- **Project Gutenberg**: Public domain (U.S. copyright expired)
- **Internet Archive**: Varies by item; most are public domain or CC-licensed
