# rag-eval-public-domain-books

Evaluation corpus of public domain literary works from Project Gutenberg and Internet Archive for testing RAG systems.

## What This Is

This repository contains **evaluation data for RAG systems**:

- **corpus.yaml** - Evaluation configuration defining domain context and testing scenarios
- **Generated questions** - Validated Q/A pairs for evaluation (where available)
- **metadata.json** - Book inventory with source IDs
- **Build tools** - Scripts for curating collections from Gutenberg and Internet Archive

The actual book files are not included - they are public domain works. Use the build scripts to fetch them.

## Purpose

Literary collections represent cultural and educational use cases. This corpus tests:

- **Retrieval**: Long-form narrative, character/theme tracking, cross-work literary analysis
- **Document processing**: Clean prose, chapter structure, front/back matter
- **Query types**: "What happens in chapter X?", "How does author portray Y?", comparative analysis

## Two Sources, Two Scripts

| Script | Source | Format | Best For |
|--------|--------|--------|----------|
| `download_gutenberg.py` | Project Gutenberg | Plain text | Clean machine-readable text |
| `download_archive.py` | Internet Archive | PDF scans | Historical/rare books |

## Config-Based Approach

Both scripts read from a shared `corpora.yaml` config file that defines book + author pairs for each corpus.

### Config Format

```yaml
corpus_name:
  source: gutenberg | archive
  books:
    - title: "Exact Title"
      author: "Last, First"
    - title: "Another Title"
      author: "Last, First"
```

### Usage

```bash
cd scripts/corpora/public_domain_books/scripts

# Download a Gutenberg corpus (plain text)
uv run python download_gutenberg.py --config ../corpora.yaml --corpus shakespeare

# Download an Archive corpus (PDF scans)
uv run python download_archive.py --config ../corpora.yaml --corpus scientific_classics

# Dry run - search and display results without downloading
uv run python download_gutenberg.py --config ../corpora.yaml --corpus shakespeare --dry-run
```

## Target Corpora

Each corpus targets 100-500 documents.

### Gutenberg Corpora (Plain Text)

| Corpus | Description |
|--------|-------------|
| `shakespeare` | Complete plays, sonnets, and poems |
| `victorian_detective` | Doyle, Collins, Poe detective fiction |
| `early_american` | Hawthorne, Melville, Poe, Twain |
| `gothic_fiction` | Shelley, Stoker, Poe, Radcliffe |
| `philosophy_19c` | Nietzsche, Mill, Kant, Schopenhauer |
| `early_scifi` | Verne, Wells science fiction |

### Archive Corpora (PDF Scans)

| Corpus | Description |
|--------|-------------|
| `scientific_classics` | Newton, Darwin, Galileo, Faraday originals |
| `greek_roman` | Homer, Plato, Aristotle, Virgil translations |

## Output Structure

```
<corpus>/
    corpus.yaml         # Evaluation configuration
    metadata.json       # Book metadata
    books/              # Text/PDF files (gitignored)
        1524.txt        # Gutenberg: ID.txt
        item_id.pdf     # Archive: identifier.pdf

scripts/
    build_gutenberg.py  # Build Gutenberg corpora with LLM curation
    build_archive.py    # Build Internet Archive corpora with LLM curation
```

### Metadata Format

```json
{
  "corpus": "shakespeare",
  "source": "gutenberg",
  "total_books": 42,
  "books": [
    {
      "id": 1524,
      "title": "Hamlet",
      "authors": [{"name": "Shakespeare, William"}],
      "subjects": ["Tragedies"],
      "file": "books/1524.txt",
      "config_title": "Hamlet",
      "config_author": "Shakespeare, William"
    }
  ]
}
```

## Features

- **Resumable downloads**: Re-run commands to continue interrupted downloads
- **Rate limiting**: 1 second base delay with exponential backoff on errors
- **Dry run mode**: Search and display results without downloading
- **Metadata tracking**: JSON metadata for all downloaded documents
- **Encoding handling**: UTF-8 with Latin-1 fallback for Gutenberg texts

## API Reference

### Gutendex (Project Gutenberg)

- **Base URL**: `https://gutendex.com/books`
- **Parameters**:
  - `search`: Search author names and titles
  - `copyright=false`: Public domain only
  - `languages=en`: English language filter
- **Docs**: https://github.com/garethbjohnson/gutendex

### Internet Archive

- **Library**: `internetarchive` (official Python client)
- **Query syntax**:
  - `creator:name`: Search by creator
  - `title:(terms)`: Search by title
  - `mediatype:texts`: Text materials only
- **Docs**: https://archive.org/developers/

## Licensing

**This repository** (scripts, configurations): MIT License

**Books**:
- **Project Gutenberg**: Public domain (U.S. copyright expired)
- **Internet Archive**: Varies by item; most are public domain or CC-licensed

## Requirements

- Python 3.11+
- Dependencies: `httpx`, `tqdm`, `internetarchive`, `pyyaml` (see pyproject.toml)
