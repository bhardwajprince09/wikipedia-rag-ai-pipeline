from pathlib import Path
import os, json, math, orjson, gc
from dataclasses import dataclass

# --- Choose your generator model route ---
USE_OPENAI = False  # set True if you want OpenAI for generation
OPENAI_MODEL = "gpt-4o-mini"  # or gpt-4o / your choice

# If using OpenAI, set your key here (or via Colab "Secrets"):
os.environ.setdefault("OPENAI_API_KEY", "")  # paste if USE_OPENAI=True

# Embedding model (fast + small, 384-dim)
EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMB_DIM = 384

# Data & index locations
DATA_DIR = Path("/content/wiki_rag")
DATA_DIR.mkdir(parents=True, exist_ok=True)
DOCS_JSONL = DATA_DIR / "docs.jsonl"           # chunked docs
FAISS_INDEX = DATA_DIR / "faiss.index"         # FAISS file
META_JSONL  = DATA_DIR / "meta.jsonl"          # parallel metadata

import wikipediaapi
from tqdm.auto import tqdm
import re

WIKI_LANG = "simple"    # "simple" or "en"
N_PAGES   = 50         # quick start; raise later
SEED_TITLES = [
    "Physics", "Chemistry", "Biology", "Mathematics", "Computer_science",
    "Artificial_intelligence", "Machine_learning", "India", "United_States",
    "World_War_II", "Oxygen", "Photosynthesis", "Python_(programming_language)"
]

wiki = wikipediaapi.Wikipedia(WIKI_LANG, extract_format=wikipediaapi.ExtractFormat.WIKI)

def clean_text(txt: str) -> str:
    # very light cleanup (we'll chunk later)
    t = re.sub(r'\n{2,}', '\n', txt or "")
    return t.strip()

seen = set()
pages = []
for title in tqdm(SEED_TITLES, desc="Seed"):
    p = wiki.page(title)
    if p.exists():
        seen.add(p.title)
        pages.append((p.title, p.fullurl, clean_text(p.text)))

# Expand by crawling linked pages (bounded)
def collect_links(title, limit=50):
    p = wiki.page(title)
    out = []
    for k, v in p.links.items():
        if len(out) >= limit: break
        if v.exists() and v.title not in seen:
            seen.add(v.title)
            out.append((v.title, v.fullurl, clean_text(v.text)))
    return out

for seed in tqdm(list(seen)[:min(20, len(seen))], desc="Crawl"):
    pages.extend(collect_links(seed, limit=50))
    if len(pages) >= N_PAGES:
        break

print(f"Collected pages: {len(pages)} (lang={WIKI_LANG})")

def chunk_text(text, size=800, overlap=120):
    text = text.strip()
    if not text:
        return []
    chunks, i = [], 0
    while i < len(text):
        j = min(len(text), i + size)
        chunks.append(text[i:j])
        i = j - overlap
        if i < 0:
            i = 0
        if i >= len(text):
            break
    return [c for c in chunks if c.strip()]

def iter_records(pages, size=800, overlap=120):
    """Generator: yields records one by one, no memory explosion"""
    for title, url, body in tqdm(pages, desc="Chunking"):
        for ci, c in enumerate(chunk_text(body, size, overlap)):
            yield {
                "title": title,
                "url": url,
                "chunk_index": ci,
                "content": c
            }
import json

# ✅ Write directly to file in small batches
batch_size = 100   # tune: smaller = safer for RAM
total = 0

with open("wiki_chunks.jsonl", "w") as f:
    batch = []
    for rec in iter_records(pages):
        batch.append(rec)
        if len(batch) >= batch_size:
            for r in batch:
                f.write(json.dumps(r) + "\n")
            total += len(batch)
            batch.clear()
    # flush last batch
    if batch:
        for r in batch:
            f.write(json.dumps(r) + "\n")
        total += len(batch)

print("✅ Saved chunks:", total)

import gzip, re, json
import mwparserfromhell
from lxml import etree
from tqdm.auto import tqdm

def clean_wikitext(raw):
    wt = mwparserfromhell.parse(raw).strip_code()
    wt = re.sub(r'\n{2,}', '\n', wt)
    return wt.strip()

def chunk_text(text, size=800, overlap=120):
    text = text.strip()
    if not text:
        return []
    i = 0
    while i < len(text):
        j = min(len(text), i + size)
        chunk = text[i:j]
        if chunk.strip():
            yield chunk
        i = j - overlap
        if i < 0:
            i = 0
        if i >= len(text):
            break

def iter_pages(xml_path):
    """Stream pages from dump, never storing all in RAM"""
    with gzip.open(xml_path, 'rb') as f:
        for _, elem in etree.iterparse(f, events=('end',), tag='{*}page'):
            title = elem.findtext('{*}title') or ''
            ns    = elem.findtext('{*}ns')
            if ns != '0':   # skip non-article pages
                elem.clear(); continue
            redirect = elem.find('{*}redirect')
            if redirect is not None:
                elem.clear(); continue
            rev = elem.find('{*}revision')
            if rev is None:
                elem.clear(); continue
            text = rev.findtext('{*}text') or ''
            body = clean_wikitext(text)
            if body and len(body) > 300:  # skip stubs
                url = f"https://simple.wikipedia.org/wiki/{title.replace(' ', '_')}"
                yield title, url, body
            elem.clear()

# ✅ Streaming write to JSONL
dump_file = "simplewiki-latest-pages-articles.xml.gz"
batch_size = 100
total = 0

with open("wiki_chunks.jsonl", "w") as f:
    batch = []
    for title, url, body in tqdm(iter_pages(dump_file), desc="Pages"):
        for ci, chunk in enumerate(chunk_text(body)):
            rec = {
                "title": title,
                "url": url,
                "chunk_index": ci,
                "content": chunk
            }
            batch.append(rec)
            if len(batch) >= batch_size:
                for r in batch:
                    f.write(json.dumps(r) + "\n")
                total += len(batch)
                batch.clear()
    # flush last batch
    if batch:
        for r in batch:
            f.write(json.dumps(r) + "\n")
        total += len(batch)

print("✅ Saved chunks:", total)
