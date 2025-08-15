#!/usr/bin/env python3
"""
ChronologyBuilder CLI

Usage examples:
  python main.py ./data/docs
  python main.py ./a.pdf ./b.txt ./more_docs --max-pages 20 --max-words 4000

Accepts one or more paths. Each path can be a file, directory, or glob.
Recursively loads supported files, extracts text, and builds a timeline using
agentActions.ChronologyBuilder (GPT-enabled by default).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from glob import glob
from pathlib import Path
from typing import Iterable, List, Optional

from agentActions.ChronologyBuilder import ChronologyBuilder, GPTAgent
from readingFile.convert_pdf import pdf_to_text, extract_first_n_pages

SUPPORTED_EXTS = {".pdf", ".txt", ".md"}


def iter_input_paths(inputs: List[str]) -> Iterable[Path]:
    """Yield file paths from a mix of files/dirs/globs, recursively for dirs.
    Only yields files with SUPPORTED_EXTS.
    """
    for raw in inputs:
        # Expand ~ and env vars, then expand globs
        expanded = os.path.expandvars(os.path.expanduser(raw))
        candidates = glob(expanded, recursive=True) or [expanded]
        for c in candidates:
            p = Path(c)
            if p.is_dir():
                for f in p.rglob("*"):
                    if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS:
                        yield f
            elif p.is_file():
                if p.suffix.lower() in SUPPORTED_EXTS:
                    yield p
            # else: ignore non-existent


def load_text_from_file(path: Path, max_pages: int, max_words: int) -> Optional[str]:
    ext = path.suffix.lower()
    try:
        if ext == ".pdf":
            with path.open("rb") as f:
                pdf_bytes = f.read()
            pdf_bytes = extract_first_n_pages(pdf_bytes, n=max_pages)
            text = pdf_to_text(pdf_bytes, num_pages=max_pages)
            return text
        elif ext in {".txt", ".md"}:
            txt = path.read_text(encoding="utf-8", errors="ignore")
            words = txt.split()
            if len(words) > max_words:
                words = words[:max_words]
            return " ".join(words)
    except Exception as e:
        print(f"[WARN] Failed to read {path}: {e}", file=sys.stderr)
        return None
    return None


def build_corpus(paths: Iterable[Path], max_pages: int, max_words: int, limit_docs: Optional[int]) -> List[str]:
    corpus: List[str] = []
    for p in paths:
        text = load_text_from_file(p, max_pages=max_pages, max_words=max_words)
        if text:
            corpus.append(text)
            if limit_docs is not None and len(corpus) >= limit_docs:
                break
    return corpus


def parse_args(argv: List[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Mine docs, extract events, and propose a timeline.")
    ap.add_argument("inputs", nargs="+", help="Files, directories, or globs (e.g., ./docs, '*.pdf')")
    ap.add_argument("--max-pages", type=int, default=15, help="Max PDF pages to read per file (default: 15)")
    ap.add_argument("--max-words", type=int, default=3000, help="Max words to read from text files (default: 3000)")
    ap.add_argument("--limit-docs", type=int, default=None, help="Optional hard limit on number of files to process")
    ap.add_argument("--model", type=str, default=None, help="OpenAI model override (otherwise uses env OPENAI_MODEL)")
    ap.add_argument("--no-sources", action="store_true", help="Exclude source indices from final timeline output")
    return ap.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)

    paths = list(iter_input_paths(args.inputs))
    if not paths:
        print("No supported files found.", file=sys.stderr)
        return 2

    print(f"[INFO] Found {len(paths)} file(s). Loading…", file=sys.stderr)
    corpus = build_corpus(paths, max_pages=args.max_pages, max_words=args.max_words, limit_docs=args.limit_docs)
    print(f"[INFO] Built corpus with {len(corpus)} document(s). Running agent…", file=sys.stderr)

    builder = ChronologyBuilder(corpus=corpus)
    agent = GPTAgent(model=args.model) if args.model else None
    state = builder.run(agent=agent)

    # Optionally drop sources in the reported timeline
    timeline = state.get("timeline", [])
    if args.no_sources:
        for e in timeline:
            e.pop("source", None)

    output = {
        "num_docs": len(corpus),
        "clusters": state.get("clusters"),
        "timeline": timeline,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))