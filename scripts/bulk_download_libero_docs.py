#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bulk download Sphinx docs sources from a GitHub Pages site.

- Crawls all .html pages under BASE_HTML
- For each page, tries to download its corresponding source file under _sources/
  (tries .md.txt, then .rst.txt)
- Saves files to OUT_DIR preserving the directory structure

Usage:
  python bulk_download_libero_docs.py

Optional:
  python bulk_download_libero_docs.py --base https://lifelong-robot-learning.github.io/LIBERO/html/ --out libero_docs
"""

import argparse
import os
import re
import sys
import time
from collections import deque
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse, urldefrag

import requests


class LinkExtractor(HTMLParser):
    # Simple HTML link extractor without external deps (BeautifulSoup not required)
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() != "a":
            return
        for k, v in attrs:
            if k.lower() == "href" and v:
                self.links.append(v)


def is_under_base(url: str, base: str) -> bool:
    # Check if url is within base path
    u = urlparse(url)
    b = urlparse(base)
    if u.scheme != b.scheme or u.netloc != b.netloc:
        return False
    return u.path.startswith(b.path)


def normalize_url(url: str) -> str:
    # Remove fragments and normalize
    url, _frag = urldefrag(url)
    return url


def safe_write(path: str, content: bytes):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(content)


def fetch(session: requests.Session, url: str, timeout=20):
    # Basic GET with retry
    for attempt in range(3):
        try:
            r = session.get(url, timeout=timeout)
            return r
        except requests.RequestException as e:
            if attempt == 2:
                raise
            time.sleep(1.0 * (attempt + 1))
    raise RuntimeError("unreachable")


def try_download_source(session: requests.Session, base_html: str, page_url: str, out_dir: str, sleep_s: float):
    """
    Given a page like .../html/algo_data/lifelong_learning_algorithms.html
    try:
      .../html/_sources/algo_data/lifelong_learning_algorithms.md.txt
      .../html/_sources/algo_data/lifelong_learning_algorithms.rst.txt
    """
    base_html = base_html.rstrip("/") + "/"
    parsed = urlparse(page_url)
    page_path = parsed.path

    # Convert ".../html/<rel>.html" -> "<rel>"
    base_path = urlparse(base_html).path
    rel_html = page_path[len(base_path):] if page_path.startswith(base_path) else None
    if not rel_html or not rel_html.endswith(".html"):
        return False

    rel_no_ext = rel_html[:-5]  # strip ".html"
    candidates = [
        f"_sources/{rel_no_ext}.md.txt",
        f"_sources/{rel_no_ext}.rst.txt",
        f"_sources/{rel_no_ext}.txt",
    ]

    for rel_src in candidates:
        src_url = urljoin(base_html, rel_src)
        r = fetch(session, src_url)
        time.sleep(sleep_s)

        if r.status_code == 200 and r.content:
            # Save mirroring the _sources/ structure under out_dir
            local_path = os.path.join(out_dir, rel_src.replace("/", os.sep))
            safe_write(local_path, r.content)
            print(f"[OK] {src_url} -> {local_path}")
            return True

        # If 404, just try next candidate
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base",
        default="https://lifelong-robot-learning.github.io/LIBERO/html/",
        help="Base HTML root of the docs (ends with /html/).",
    )
    ap.add_argument(
        "--out",
        default="libero_docs_sources",
        help="Output directory to save downloaded sources.",
    )
    ap.add_argument(
        "--sleep",
        type=float,
        default=0.15,
        help="Sleep seconds between requests (be polite to the server).",
    )
    ap.add_argument(
        "--max-pages",
        type=int,
        default=5000,
        help="Safety cap for number of HTML pages to crawl.",
    )
    args = ap.parse_args()

    base_html = args.base.rstrip("/") + "/"
    out_dir = args.out
    sleep_s = args.sleep

    start_url = urljoin(base_html, "index.html")

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "libero-docs-downloader/1.0 (+https://github.com/) python-requests",
        }
    )

    visited = set()
    q = deque([start_url])
    html_pages = []

    print(f"Base: {base_html}")
    print(f"Start: {start_url}")
    print(f"Out: {out_dir}")

    # Crawl HTML pages
    while q and len(visited) < args.max_pages:
        url = normalize_url(q.popleft())
        if url in visited:
            continue
        visited.add(url)

        # Only crawl .html under base
        if not is_under_base(url, base_html):
            continue
        if not urlparse(url).path.endswith(".html"):
            continue

        try:
            r = fetch(session, url)
        except Exception as e:
            print(f"[WARN] fetch failed: {url} ({e})")
            continue
        time.sleep(sleep_s)

        if r.status_code != 200 or "text/html" not in r.headers.get("Content-Type", ""):
            continue

        html_pages.append(url)
        parser = LinkExtractor()
        try:
            parser.feed(r.text)
        except Exception:
            pass

        for href in parser.links:
            # Skip mailto/javascript etc.
            if href.startswith("mailto:") or href.startswith("javascript:"):
                continue
            abs_url = normalize_url(urljoin(url, href))

            # Ignore static assets and common noisy pages
            p = urlparse(abs_url).path
            if any(seg in p for seg in ["/_static/", "/_images/", "/_downloads/"]):
                continue
            if re.search(r"/(search|genindex|py-modindex)\.html$", p):
                continue

            if is_under_base(abs_url, base_html) and p.endswith(".html"):
                if abs_url not in visited:
                    q.append(abs_url)

    print(f"\nCrawled HTML pages: {len(html_pages)}")
    if len(visited) >= args.max_pages:
        print("[WARN] Reached --max-pages cap; crawl may be incomplete.")

    # Download sources for each HTML page
    ok = 0
    miss = 0
    for page_url in html_pages:
        got = try_download_source(session, base_html, page_url, out_dir, sleep_s)
        if got:
            ok += 1
        else:
            miss += 1
            print(f"[MISS] no source found for {page_url}")

    print(f"\nDone. downloaded={ok}, missing={miss}")
    print(f"Saved under: {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    sys.exit(main())
