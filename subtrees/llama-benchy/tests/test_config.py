import pytest

from llama_benchy.config import BenchmarkConfig


def test_parse_extra_body_accepts_repeated_key_value_entries():
    extra = BenchmarkConfig._parse_extra_body([
        "min_tokens=1024",
        "ignore_eos=true",
        "temperature=0",
    ])

    assert extra == {
        "min_tokens": 1024,
        "ignore_eos": True,
        "temperature": 0,
    }


def test_parse_extra_body_accepts_aiperf_style_comma_separated_entries():
    extra = BenchmarkConfig._parse_extra_body([
        "max_tokens:1024,min_tokens:1024,ignore_eos:true",
    ])

    assert extra == {
        "max_tokens": 1024,
        "min_tokens": 1024,
        "ignore_eos": True,
    }


def test_parse_extra_body_rejects_entries_without_separator():
    with pytest.raises(ValueError):
        BenchmarkConfig._parse_extra_body(["ignore_eos"])


def test_from_args_parses_pre_and_post_run_cmd(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "llama-benchy",
            "--base-url",
            "http://example.test/v1",
            "--model",
            "org/model",
            "--pre-run-cmd",
            "echo pre",
            "--post-run-cmd",
            "echo post",
        ],
    )

    config = BenchmarkConfig.from_args()

    assert config.pre_run_cmd == "echo pre"
    assert config.post_run_cmd == "echo post"
