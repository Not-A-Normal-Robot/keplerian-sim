#!/usr/bin/env python3
"""
Scan all .rs files for HTTP/HTTPS links (excluding Wikipedia) and verify
each unique link appears in CREDITS.md. Exit with non-zero if any link
is missing from CREDITS.md.

Usage: python .github/scripts/check_rs_links_against_credits.py [--credits PATH]
"""
from __future__ import annotations
import re
import sys
import argparse
import pathlib
from urllib.parse import urlparse

URL_RE = re.compile(
    r"""(?xi)
    (?:
      # markdown-style link (text)(url)
      \([\'"]?(https?://[^\s\)\]\}>'"]+)[\'"]?\)
    |
      # bare URL
      (https?://[^\s\)\]\}>'",;]+)
    )
    """
)

WIKI_PATTERN = re.compile(r"(^|\.)wikipedia\.org$", re.I)

def find_urls_in_text(text: str) -> list[str]:
    results = []
    for m in URL_RE.finditer(text):
        url = m.group(1) or m.group(2)
        if url:
            # strip trailing punctuation commonly attached to URLs
            url = url.rstrip('.,;:!)"\']')
            results.append(url)
    return results

def normalize_url(url: str) -> str:
    """
    Normalize URL for comparison:
      - lowercase netloc
      - strip leading scheme
      - drop 'www.' prefix from netloc
      - strip trailing slash from path
      - include query to avoid false merges
    """
    parsed = urlparse(url)
    netloc = parsed.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    path = parsed.path.rstrip('/')
    # Rebuild a normalized "netloc + path + ?query"
    norm = netloc + path
    if parsed.query:
        norm += '?' + parsed.query
    return norm

def is_wikipedia(url: str) -> bool:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    return host.endswith("wikipedia.org")

def is_own_repo(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.netloc.endswith("github.com") and parsed.path.startswith("/Not-A-Normal-Robot/keplerian-sim")

def collect_rs_links(root: pathlib.Path) -> dict[str, set[str]]:
    """
    Returns mapping file_path -> set(urls found)
    """
    mapping: dict[str, set[str]] = {}
    for path in root.rglob("*.rs"):
        # skip typical vendored/dependency directories if present:
        if any(part in ("target", ".git") for part in path.parts):
            # still allow top-level src files; this just avoids target
            if "target" in path.parts:
                continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        urls = set(find_urls_in_text(text))
        # Filter out wikipedia and self links now
        urls = {u for u in urls if not (is_wikipedia(u) or is_own_repo(u))}
        if urls:
            mapping[str(path)] = urls
    return mapping

def collect_credits_links(credits_path: pathlib.Path) -> set[str]:
    if not credits_path.exists():
        return set()
    text = credits_path.read_text(encoding="utf-8", errors="ignore")
    urls = set(find_urls_in_text(text))
    return urls

def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--credits", default="CREDITS.md", help="Path to CREDITS.md")
    args = parser.parse_args(argv)

    root = pathlib.Path(".").resolve()
    credits_path = pathlib.Path(args.credits)

    rs_links_map = collect_rs_links(root)
    credits_links = collect_credits_links(credits_path)

    # Build normalized sets for comparison
    normalized_credits = {normalize_url(u) for u in credits_links}
    missing_map: dict[str, set[str]] = {}
    for path, urls in rs_links_map.items():
        for u in urls:
            nu = normalize_url(u)
            if nu not in normalized_credits:
                missing_map.setdefault(path, set()).add(nu)

    if not missing_map:
        print("OK: All links found in CREDITS.md")
        return 0

    # Print a helpful report
    print("ERROR: Found links in .rs files that are not listed in CREDITS.md\n")
    for path, urls in sorted(missing_map.items()):
        print(f"In {path}:")
        for u in sorted(urls):
            print(f"  - {u}")
        print()

    # Exit non-zero to fail the job
    return 2

if __name__ == "__main__":
    raise SystemExit(main())