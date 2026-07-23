"""
Crawls neuralninjas.in, chunks page content, embeds it, and upserts into
Supabase (nn_documents). First tries WordPress/XML sitemaps to get a
complete, reliable list of every post and page (this avoids missing pages
that are only linked via JS-rendered carousels on the homepage). Falls back
to plain link-crawling if no sitemap is found.

Usage:
    python crawler.py https://neuralninjas.in
"""
import re
import sys
from urllib.parse import urljoin, urlparse
from xml.etree import ElementTree

import requests
from bs4 import BeautifulSoup

from rag import embed_text, supabase

CHUNK_SIZE = 800       # characters
CHUNK_OVERLAP = 150
REQUEST_TIMEOUT = 15
HEADERS = {"User-Agent": "NeuralNinjasBot/1.0 (+https://neuralninjas.in)"}

SITEMAP_CANDIDATES = [
    "/sitemap_index.xml",
    "/sitemap.xml",
    "/wp-sitemap.xml",
]

SKIP_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".pdf",
                    ".zip", ".css", ".js", ".xml", ".ico")


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap
    return [c.strip() for c in chunks if len(c.strip()) > 50]


def fetch_sitemap_urls(base_url: str) -> list[str]:
    """Recursively resolve sitemap index files into a flat list of page URLs."""
    urls = []
    to_parse = []

    for path in SITEMAP_CANDIDATES:
        try:
            resp = requests.get(urljoin(base_url, path), headers=HEADERS, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200 and "<" in resp.text[:100]:
                to_parse.append(resp.text)
                break
        except Exception:
            continue

    seen_sitemaps = set()
    while to_parse:
        xml_text = to_parse.pop()
        try:
            root = ElementTree.fromstring(xml_text)
        except ElementTree.ParseError:
            continue

        ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        locs = [el.text.strip() for el in root.findall(".//sm:loc", ns) if el.text]

        for loc in locs:
            if loc.endswith(".xml") and loc not in seen_sitemaps:
                seen_sitemaps.add(loc)
                try:
                    r = requests.get(loc, headers=HEADERS, timeout=REQUEST_TIMEOUT)
                    if r.status_code == 200:
                        to_parse.append(r.text)
                except Exception:
                    continue
            elif not loc.endswith(".xml"):
                urls.append(loc)

    return urls


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


def crawl_site(start_url: str, max_pages: int = 400):
    domain = urlparse(start_url).netloc
    visited = set()

    sitemap_urls = fetch_sitemap_urls(start_url)
    sitemap_urls = [u for u in sitemap_urls
                     if urlparse(u).netloc == domain
                     and not u.lower().endswith(SKIP_EXTENSIONS)]

    if sitemap_urls:
        print(f"Found {len(sitemap_urls)} URLs via sitemap.")
        to_visit = list(dict.fromkeys(sitemap_urls))  # dedupe, keep order
        follow_links = False
    else:
        print("No sitemap found, falling back to link-crawling.")
        to_visit = [start_url]
        follow_links = True

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

        if follow_links:
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
