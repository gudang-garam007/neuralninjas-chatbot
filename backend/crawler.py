"""
Crawls neuralninjas.in, chunks page content, embeds it, and upserts into
Supabase (nn_documents). Run manually or trigger via the /crawl endpoint
(and later, an n8n webhook on "post published").

Usage:
    python crawler.py https://neuralninjas.in
"""
import asyncio
import sys
import re

from crawl4ai import AsyncWebCrawler
from rag import embed_text, supabase

CHUNK_SIZE = 800       # characters
CHUNK_OVERLAP = 150


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap
    return [c.strip() for c in chunks if len(c.strip()) > 50]


async def crawl_site(start_url: str, max_pages: int = 200):
    visited = set()
    to_visit = [start_url]
    domain = start_url.split("/")[2]

    async with AsyncWebCrawler(verbose=True) as crawler:
        while to_visit and len(visited) < max_pages:
            url = to_visit.pop(0)
            if url in visited:
                continue
            visited.add(url)

            try:
                result = await crawler.arun(url=url)
            except Exception as e:
                print(f"skip {url}: {e}")
                continue

            if not result.success:
                continue

            title = result.metadata.get("title", url) if result.metadata else url
            markdown = result.markdown or ""

            index_page(url, title, markdown)

            for link in (result.links or {}).get("internal", []):
                href = link.get("href", "")
                if domain in href and href not in visited:
                    to_visit.append(href)

    print(f"Done. Crawled {len(visited)} pages.")


def index_page(url: str, title: str, markdown: str):
    if not markdown.strip():
        return

    # remove old chunks for this URL so re-crawls don't duplicate
    supabase.table("nn_documents").delete().eq("url", url).execute()

    chunks = chunk_text(markdown)
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
    asyncio.run(crawl_site(target))
