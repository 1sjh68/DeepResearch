"""Modular web research pipeline."""

from .models import ResearchAnchor, ResearchResult, ResearchSource
from .pipeline import executor as _executor

cleanup_fetcher = _executor.cleanup_fetcher
create_hyde_search_queries = _executor.create_hyde_search_queries
enable_diagnostic_logging = _executor.enable_diagnostic_logging
enhanced_research_cycle_with_hyde = _executor.enhanced_research_cycle_with_hyde
enhanced_research_cycle_with_hyde_async = _executor.enhanced_research_cycle_with_hyde_async
ensure_fetcher_initialized = _executor.ensure_fetcher_initialized
get_diagnostic_statistics = _executor.get_diagnostic_statistics
get_diagnostic_summary = _executor.get_diagnostic_summary
get_fetcher_statistics = _executor.get_fetcher_statistics
run_research_cycle = _executor.run_research_cycle
run_research_cycle_async = _executor.run_research_cycle_async
run_structured_research_cycle = _executor.run_structured_research_cycle
run_structured_research_cycle_async = _executor.run_structured_research_cycle_async

__all__ = [
    "ResearchAnchor",
    "ResearchResult",
    "ResearchSource",
    "cleanup_fetcher",
    "create_hyde_search_queries",
    "enable_diagnostic_logging",
    "enhanced_research_cycle_with_hyde",
    "enhanced_research_cycle_with_hyde_async",
    "ensure_fetcher_initialized",
    "get_diagnostic_statistics",
    "get_diagnostic_summary",
    "get_fetcher_statistics",
    "run_research_cycle",
    "run_research_cycle_async",
    "run_structured_research_cycle",
    "run_structured_research_cycle_async",
]
