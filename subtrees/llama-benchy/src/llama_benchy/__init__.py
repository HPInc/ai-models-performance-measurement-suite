"""
llama-benchy - llama-bench style benchmarking tool for all backends

This package provides a benchmarking tool for OpenAI-compatible LLM endpoints,
generating statistics similar to `llama-bench`.
"""

try:
    from ._version import __version__
except ImportError:  # pragma: no cover
    __version__ = "0+local"
