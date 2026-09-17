from types import SimpleNamespace

from atmospheric_research_agent.agent.tools import source_reader, web_search


def test_search_public_web_returns_candidates_not_evidence(monkeypatch) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.setattr(
        web_search,
        "_tavily_search",
        lambda query, key: {
            "results": [
                {
                    "title": "官方资料",
                    "url": "https://example.org/source",
                    "content": "候选摘要",
                    "score": 0.9,
                    "published_date": "2026-01-01",
                }
            ]
        },
    )

    result = web_search.search_public_web("副热带高压")

    assert result["status"] == "success"
    assert result["result_count"] == 1
    assert result["results"][0]["url"] == "https://example.org/source"


def test_read_public_source_extracts_content_and_metadata(monkeypatch) -> None:
    class FakeTrafilatura:
        @staticmethod
        def fetch_url(url):
            return "<html>测试</html>"

        @staticmethod
        def extract(*args, **kwargs):
            return "网页正文内容。"

        @staticmethod
        def extract_metadata(*args, **kwargs):
            return SimpleNamespace(
                title="网页标题",
                author="作者甲",
                sitename="示例机构",
                hostname="example.org",
                date="2026-01-01",
            )

    monkeypatch.setitem(__import__("sys").modules, "trafilatura", FakeTrafilatura)

    result = source_reader.read_public_source("https://example.org/article")

    assert result["status"] == "success"
    assert result["content"] == "网页正文内容。"
    assert result["source"]["display_name"] == "网页标题"
    assert result["source"]["author"] == "作者甲"


def test_read_public_source_rejects_local_url() -> None:
    result = source_reader.read_public_source("http://localhost:8000/private")

    assert result["status"] == "error"
    assert "不允许读取本地地址" in result["message"]


def test_validate_public_url_normalizes_markdown_link() -> None:
    url = source_reader._validate_public_url(
        "[Penn State](https://courses.ems.psu.edu/meteo3/node/2048)"
    )

    assert url == "https://courses.ems.psu.edu/meteo3/node/2048"
