#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import html
import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable


INDEX_URL = "https://nalog.garant.ru/fns/nk/a815d76f185337a0dac89020e1545838/"
ARTICLE_URL = "https://nalog.garant.ru/fns/nk/{slug}/"
ARTICLE_PATH = re.compile(r"/fns/nk/([a-f0-9]{32})/")
ARTICLE_NUMBER = re.compile(r"^Статья\s+([0-9]+(?:[.-][0-9]+)*)\b")
USER_AGENT = "ragview-tax-code-corpus/1.0 (+https://github.com/br0mberg/ragview-rag-types)"


@dataclass(frozen=True)
class ArticleLink:
    slug: str
    part: int
    title: str


@dataclass(frozen=True)
class FullArticle:
    id: str
    part: int
    article: str
    title: str
    text: str
    sourceUrl: str
    sourceSha256: str


class IndexParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[ArticleLink] = []
        self._href: str | None = None
        self._text: list[str] = []
        self._part = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            self._href = dict(attrs).get("href")
            self._text = []

    def handle_data(self, data: str) -> None:
        if self._href is not None:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag != "a" or self._href is None:
            return
        text = normalize_space(" ".join(self._text))
        if text == "Часть первая":
            self._part = 1
        elif text == "Часть вторая":
            self._part = 2
        match = ARTICLE_PATH.search(self._href)
        if match and self._part and ARTICLE_NUMBER.match(text):
            self.links.append(ArticleLink(match.group(1), self._part, text))
        self._href = None
        self._text = []


class ArticleParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title = ""
        self.paragraphs: list[str] = []
        self._in_content = False
        self._capture: str | None = None
        self._text: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        classes = set((attributes.get("class") or "").split())
        if tag == "div" and "table_content" in classes and not self._in_content:
            self._in_content = True
            return
        if not self._in_content:
            return
        if tag == "table" and "nav_bottom" in classes:
            self._in_content = False
            return
        if tag in {"script", "style"}:
            self._skip_depth += 1
            return
        if self._skip_depth == 0 and tag == "h1" and "huge" in classes:
            self._capture = "title"
            self._text = []
        elif self._skip_depth == 0 and tag == "p":
            self._capture = "paragraph"
            self._text = []
        elif self._capture and tag == "br":
            self._text.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_content and self._skip_depth == 0 and self._capture:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self._skip_depth:
            if tag in {"script", "style"}:
                self._skip_depth -= 1
            return
        if self._capture == "title" and tag == "h1":
            self.title = normalize_space(" ".join(self._text))
            self._capture = None
        elif self._capture == "paragraph" and tag == "p":
            paragraph = normalize_space(" ".join(self._text))
            if paragraph:
                self.paragraphs.append(paragraph)
            self._capture = None


def normalize_space(value: str) -> str:
    value = html.unescape(value).replace("\xa0", " ")
    return re.sub(r"[ \t\r\f\v]+", " ", value).strip()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch(url: str, attempts: int = 4) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                return response.read()
        except (TimeoutError, urllib.error.URLError) as error:
            if attempt == attempts - 1:
                raise RuntimeError(f"Cannot fetch {url}") from error
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def decode_page(data: bytes) -> str:
    for encoding in ("utf-8", "windows-1251"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("windows-1251", errors="replace")


def parse_index(data: bytes) -> list[ArticleLink]:
    parser = IndexParser()
    parser.feed(decode_page(data))
    unique = {link.slug: link for link in parser.links}
    links = list(unique.values())
    if len(links) < 500:
        raise ValueError(f"Index yielded only {len(links)} article links")
    return links


def parse_article(link: ArticleLink, data: bytes) -> FullArticle:
    parser = ArticleParser()
    parser.feed(decode_page(data))
    title = parser.title or link.title
    number_match = ARTICLE_NUMBER.match(title)
    if not number_match:
        raise ValueError(f"Cannot parse article number: {title}")
    paragraphs = parser.paragraphs
    if paragraphs and normalize_space(paragraphs[0]) == normalize_space(title):
        paragraphs = paragraphs[1:]
    text = "\n\n".join(paragraphs).strip()
    if not text:
        raise ValueError(f"Article body is empty: {title}")
    article = number_match.group(1)
    return FullArticle(
        id=f"nk-{link.part}-article-{article}",
        part=link.part,
        article=article,
        title=title,
        text=text,
        sourceUrl=ARTICLE_URL.format(slug=link.slug),
        sourceSha256=sha256(data),
    )


def chunk_article(article: FullArticle, max_chars: int) -> list[dict[str, str]]:
    chunks: list[str] = []
    start = 0
    while start < len(article.text):
        end = min(start + max_chars, len(article.text))
        if end < len(article.text):
            paragraph_boundary = article.text.rfind("\n\n", start + max_chars // 2, end)
            sentence_boundary = article.text.rfind(" ", start + max_chars // 2, end)
            if paragraph_boundary >= 0:
                end = paragraph_boundary + 2
            elif sentence_boundary >= 0:
                end = sentence_boundary + 1
        chunks.append(article.text[start:end])
        start = end

    rebuilt = "".join(chunks)
    if rebuilt != article.text:
        raise AssertionError(f"Lossless chunk validation failed for {article.id}")
    width = max(3, len(str(len(chunks))))
    return [
        {
            "id": f"{article.id}-chunk-{index:0{width}d}",
            "title": f"НК РФ ч.{article.part}, {article.title}",
            "text": chunk,
        }
        for index, chunk in enumerate(chunks, start=1)
    ]


def validate_articles(articles: list[FullArticle], expected_count: int) -> None:
    if len(articles) != expected_count:
        raise ValueError(f"Expected {expected_count} articles, got {len(articles)}")
    ids = [article.id for article in articles]
    urls = [article.sourceUrl for article in articles]
    if len(ids) != len(set(ids)):
        raise ValueError("Parsed articles contain duplicate IDs")
    if len(urls) != len(set(urls)):
        raise ValueError("Parsed articles contain duplicate source URLs")
    if any(not article.text.strip() or article.text.strip() == article.title.strip() for article in articles):
        raise ValueError("Parsed articles contain an empty or title-only body")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    raw_dir = (args.raw_dir or output / "raw").resolve()
    raw_dir.mkdir(parents=True, exist_ok=True)
    index_path = raw_dir / "index.html"
    if args.refresh or not index_path.exists():
        index_data = fetch(INDEX_URL)
        index_path.write_bytes(index_data)
    else:
        index_data = index_path.read_bytes()
    links = parse_index(index_data)

    def load_article(link: ArticleLink) -> FullArticle:
        path = raw_dir / "articles" / f"{link.slug}.html"
        if args.refresh or not path.exists():
            data = fetch(ARTICLE_URL.format(slug=link.slug))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        else:
            data = path.read_bytes()
        return parse_article(link, data)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        articles = list(executor.map(load_article, links))
    validate_articles(articles, len(links))
    articles.sort(
        key=lambda article: (article.part, tuple(map(int, re.split(r"[.-]", article.article))))
    )

    chunks = [chunk for article in articles for chunk in chunk_article(article, args.max_chars)]
    snapshot = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    manifest = {
        "schemaVersion": 2,
        "createdAt": snapshot,
        "source": INDEX_URL,
        "sourceOwner": "ФНС России; сводный текст предоставлен системой ГАРАНТ",
        "articleCount": len(articles),
        "chunkCount": len(chunks),
        "chunking": {
            "algorithm": "paragraph-greedy-lossless-v1",
            "maxChars": args.max_chars,
            "truncation": False,
            "validation": "concatenating chunks exactly reproduces each normalized article text",
        },
        "indexSha256": sha256(index_data),
    }
    write_json(output / "full_articles.json", [asdict(article) for article in articles])
    write_json(output / "corpus.json", chunks)
    write_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a lossless Tax Code retrieval corpus")
    parser.add_argument("--output", type=Path, default=Path("data/tax-code-snapshot"))
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--max-chars", type=int, default=4000)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args(argv)
    if args.max_chars < 1000:
        parser.error("--max-chars must be at least 1000")
    if not 1 <= args.workers <= 12:
        parser.error("--workers must be between 1 and 12")
    return args


if __name__ == "__main__":
    build(parse_args())
