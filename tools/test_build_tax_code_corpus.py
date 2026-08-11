from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("build_tax_code_corpus.py")
SPEC = importlib.util.spec_from_file_location("build_tax_code_corpus", MODULE_PATH)
assert SPEC and SPEC.loader
builder = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = builder
SPEC.loader.exec_module(builder)


class TaxCodeCorpusBuilderTest(unittest.TestCase):
    def test_decode_page_supports_utf8_without_mojibake(self) -> None:
        self.assertEqual(builder.decode_page("Статья 1".encode()), "Статья 1")

    def test_article_parser_extracts_body_without_navigation(self) -> None:
        page = """
        <div class="table_content">
          <h1 class="huge">Статья 1. Заголовок</h1>
          <div class="block"><p>Статья 1. Заголовок</p></div>
          <div class="block"><p>1. Первый пункт.</p><p>2. Второй пункт.</p></div>
          <table class="nav_bottom"><tr><td>Статья 2. Навигация</td></tr></table>
        </div>
        """.encode("windows-1251")
        link = builder.ArticleLink("a" * 32, 1, "Статья 1. Заголовок")

        article = builder.parse_article(link, page)

        self.assertEqual(article.title, "Статья 1. Заголовок")
        self.assertEqual(article.text, "1. Первый пункт.\n\n2. Второй пункт.")
        self.assertNotIn("Навигация", article.text)

    def test_article_parser_tolerates_unbalanced_source_divs(self) -> None:
        page = """
        <div class="table_content">
          <h1 class="short">Статья 249. Доходы от реализации</h1>
          <div class="block"></div>
          <p>Статья 249. Доходы от реализации</p>
          </div>
          <div class="block"></div>
          <p>1. Доходом признается выручка.</p>
          </div>
          <p>2. Выручка определяется по всем поступлениям.</p>
          <table class="nav_bottom"><tr><td>Статья 250</td></tr></table>
        </div>
        """.encode("windows-1251")
        link = builder.ArticleLink("b" * 32, 2, "Статья 249. Доходы от реализации")

        article = builder.parse_article(link, page)

        self.assertEqual(
            article.text,
            "1. Доходом признается выручка.\n\n2. Выручка определяется по всем поступлениям.",
        )
        self.assertNotIn("Статья 250", article.text)

    def test_article_parser_rejects_title_only_page(self) -> None:
        page = """
        <div class="table_content">
          <p>Статья 249. Доходы от реализации</p>
          <table class="nav_bottom"></table>
        </div>
        """.encode("windows-1251")
        link = builder.ArticleLink("c" * 32, 2, "Статья 249. Доходы от реализации")

        with self.assertRaisesRegex(ValueError, "Article body is empty"):
            builder.parse_article(link, page)

    def test_chunking_is_lossless(self) -> None:
        article = builder.FullArticle(
            id="nk-1-article-1",
            part=1,
            article="1",
            title="Статья 1. Заголовок",
            text=("Первый абзац. " * 100).strip() + "\n\n" + ("Второй абзац. " * 100).strip(),
            sourceUrl="https://example.test",
            sourceSha256="0" * 64,
        )

        chunks = builder.chunk_article(article, 1000)

        self.assertGreater(len(chunks), 1)
        self.assertEqual("".join(chunk["text"] for chunk in chunks), article.text)
        self.assertTrue(all(len(chunk["text"]) <= 1000 for chunk in chunks))

    def test_validation_rejects_duplicate_article_ids(self) -> None:
        article = builder.FullArticle(
            id="nk-1-article-1",
            part=1,
            article="1",
            title="Статья 1. Заголовок",
            text="Содержательный текст.",
            sourceUrl="https://example.test/one",
            sourceSha256="0" * 64,
        )
        duplicate = builder.FullArticle(
            id=article.id,
            part=1,
            article="1",
            title=article.title,
            text=article.text,
            sourceUrl="https://example.test/two",
            sourceSha256="1" * 64,
        )

        with self.assertRaisesRegex(ValueError, "duplicate IDs"):
            builder.validate_articles([article, duplicate], 2)

    def test_parser_does_notRequireLegacyQuestionsFile(self) -> None:
        args = builder.parse_args(["--output", "data/local/test"])

        self.assertFalse(hasattr(args, "questions"))
        self.assertIsNone(args.raw_dir)


if __name__ == "__main__":
    unittest.main()
