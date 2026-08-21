import sys
import types

from llama_benchy.corpus import LightweightTokenizer, TokenizedCorpus


class _FakeEncoding:
    def __init__(self, ids):
        self.ids = ids


class _FakeRawTokenizer:
    def __init__(self):
        self.encode_calls = []
        self.decode_calls = []

    def encode(self, text, add_special_tokens=False):
        self.encode_calls.append((text, add_special_tokens))
        return _FakeEncoding([1, 2, 3])

    def decode(self, token_ids, skip_special_tokens=True):
        self.decode_calls.append((token_ids, skip_special_tokens))
        return "decoded"


def _install_tokenizers(monkeypatch, tokenizer_cls):
    monkeypatch.setitem(
        sys.modules,
        "tokenizers",
        types.SimpleNamespace(Tokenizer=tokenizer_cls),
    )


def _install_transformers(monkeypatch, auto_tokenizer_cls):
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(AutoTokenizer=auto_tokenizer_cls),
    )


def test_lightweight_tokenizer_matches_transformers_encode_decode_shape(monkeypatch):
    raw = _FakeRawTokenizer()

    class FakeTokenizer:
        @staticmethod
        def from_pretrained(name):
            assert name == "repo/model"
            return raw

    _install_tokenizers(monkeypatch, FakeTokenizer)

    tokenizer = LightweightTokenizer.from_pretrained("repo/model")

    assert tokenizer.encode("hello", add_special_tokens=False) == [1, 2, 3]
    assert tokenizer.decode([1, 2, 3]) == "decoded"
    assert raw.encode_calls == [("hello", False)]
    assert raw.decode_calls == [([1, 2, 3], False)]


def test_get_tokenizer_uses_lightweight_backend_without_transformers(monkeypatch):
    class FakeTokenizer:
        @staticmethod
        def from_pretrained(name):
            return _FakeRawTokenizer()

    class FakeAutoTokenizer:
        calls = []

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            cls.calls.append((args, kwargs))
            return object()

    _install_tokenizers(monkeypatch, FakeTokenizer)
    _install_transformers(monkeypatch, FakeAutoTokenizer)

    corpus = TokenizedCorpus.__new__(TokenizedCorpus)
    tokenizer = corpus._get_tokenizer("repo/model")

    assert isinstance(tokenizer, LightweightTokenizer)
    assert FakeAutoTokenizer.calls == []


def test_get_tokenizer_lazily_uses_transformers_when_lightweight_fails(monkeypatch):
    class FakeTokenizer:
        @staticmethod
        def from_pretrained(name):
            raise RuntimeError("missing tokenizer.json")

    class FakeAutoTokenizer:
        calls = []
        tokenizer = object()

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            cls.calls.append((args, kwargs))
            return cls.tokenizer

    _install_tokenizers(monkeypatch, FakeTokenizer)
    _install_transformers(monkeypatch, FakeAutoTokenizer)

    corpus = TokenizedCorpus.__new__(TokenizedCorpus)
    tokenizer = corpus._get_tokenizer("repo/model")

    assert tokenizer is FakeAutoTokenizer.tokenizer
    assert FakeAutoTokenizer.calls == [
        (("repo/model",), {"use_fast": True, "trust_remote_code": False})
    ]


def test_get_tokenizer_falls_back_to_lightweight_gpt2(monkeypatch):
    class FakeTokenizer:
        calls = []

        @classmethod
        def from_pretrained(cls, name):
            cls.calls.append(name)
            if name == "repo/model":
                raise RuntimeError("missing tokenizer.json")
            return _FakeRawTokenizer()

    class FakeAutoTokenizer:
        calls = []

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            cls.calls.append((args, kwargs))
            raise MemoryError("too large")

    _install_tokenizers(monkeypatch, FakeTokenizer)
    _install_transformers(monkeypatch, FakeAutoTokenizer)

    corpus = TokenizedCorpus.__new__(TokenizedCorpus)
    tokenizer = corpus._get_tokenizer("repo/model")

    assert isinstance(tokenizer, LightweightTokenizer)
    assert FakeTokenizer.calls == ["repo/model", "gpt2"]
    assert FakeAutoTokenizer.calls == [
        (("repo/model",), {"use_fast": True, "trust_remote_code": False})
    ]
