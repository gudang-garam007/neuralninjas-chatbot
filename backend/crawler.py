"""
Crawls neuralninjas.in, chunks page content, embeds it, and upserts into
Supabase (nn_documents). Uses plain requests + BeautifulSoup (no headless
browser) so it fits comfortably in a 512MB free-tier RAM budget.

Usage:
    python crawler.py https://neuralninjas.in
"""
import re
import sys
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from rag import embed_text, supabase

CHUNK_SIZE = 800       # characters
CHUNK_OVERLAP = 150
REQUEST_TIMEOUT = 15
HEADERS = {"User-Agent": "NeuralNinjasBot/1.0 (+https://neuralninjas.in)"}


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap
    return [c.strip() for c in chunks if len(c.strip()) > 50]


def extract_text_and_links(url: str, html: str, domain: str):
    soup = BeautifulSoup(html, "lxml")

    title = soup.title.string.strip() if soup.title and soup.title.string else url

    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "svg"]):
        tag.decompose()

    main = soup.find("main") or soup.find("article") or soup.body
    text = main.get_text(separator="\n", strip=True) if main else ""

    links = []
    for a in soup.find_all("a", href=True):
        href = urljoin(url, a["href"]).split("#")[0]
        if urlparse(href).netloc == domain and href.startswith("http"):
            links.append(href)

    return title, text, links


def crawl_site(start_url: str, max_pages: int = 200):
    domain = urlparse(start_url).netloc
    visited = set()
    to_visit = [start_url]

    while to_visit and len(visited) < max_pages:
        url = to_visit.pop(0)
        if url in visited:
            continue
        visited.add(url)

        try:
            resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
        except Exception as e:
            print(f"skip {url}: {e}")
            continue

        content_type = resp.headers.get("content-type", "")
        if "text/html" not in content_type:
            continue

        title, text, links = extract_text_and_links(url, resp.text, domain)
        index_page(url, title, text)

        for link in links:
            if link not in visited and link not in to_visit:
                to_visit.append(link)

    print(f"Done. Crawled {len(visited)} pages.")


def index_page(url: str, title: str, text: str):
    if not text.strip():
        return

    supabase.table("nn_documents").delete().eq("url", url).execute()

    chunks = chunk_text(text)
    rows = []
    for chunk in chunks:
        rows.append({
            "url": url,
            "title": title,
            "content": chunk,
            "embedding": embed_text(chunk),
        })

    if rows:
        supabase.table("nn_documents").insert(rows).execute()
        print(f"Indexed {len(rows)} chunks from {url}")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "https://neuralninjas.in"
    crawl_site(target)
