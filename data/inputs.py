"""Lightweight typed data inputs: sectors, benchmark weights, factor loadings.

No network — everything loads from local CSVs. These feed the Sprint-3 nodes
that need exogenous data:

* sectors      -> GroupCap                 (``ticker,sector`` long format)
* benchmarks   -> TrackingErrorCap, MinTrackingError
* factors      -> FactorExposure

Benchmarks and factors share a *wide* CSV shape: the first column is ``ticker``
and every remaining column is one named series (a benchmark, or a factor). So a
benchmark file with one ``bench`` benchmark is::

    ticker,bench
    AAA,0.2
    BBB,0.3
    ...

:func:`align_named` turns ``{name -> {ticker -> value}}`` into universe-ordered
arrays and validates that every universe ticker is present — surfacing a clear
error rather than silently defaulting a missing benchmark weight to zero.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_sectors(path: Path) -> dict[str, str]:
    """Load a ``ticker,sector`` CSV into ``{ticker -> sector}``."""
    df = pd.read_csv(path)
    if not {"ticker", "sector"}.issubset(df.columns):
        raise ValueError("sectors CSV must have columns: ticker, sector.")
    return dict(zip(df["ticker"].astype(str), df["sector"].astype(str), strict=True))


def load_named_series(path: Path, *, label: str) -> dict[str, dict[str, float]]:
    """Load a wide ``ticker,<name1>,<name2>,...`` CSV into ``{name -> {ticker -> value}}``."""
    df = pd.read_csv(path)
    if df.columns.empty or df.columns[0] != "ticker":
        raise ValueError(f"{label} CSV must have 'ticker' as its first column.")
    names = [c for c in df.columns[1:]]
    if not names:
        raise ValueError(f"{label} CSV must have at least one named column after 'ticker'.")
    out: dict[str, dict[str, float]] = {}
    for name in names:
        out[str(name)] = {
            str(t): float(v) for t, v in zip(df["ticker"], df[name], strict=True)
        }
    return out


def align_named(
    named: dict[str, dict[str, float]] | None,
    universe: list[str],
    *,
    label: str,
) -> dict[str, np.ndarray] | None:
    """Align ``{name -> {ticker -> value}}`` to universe-ordered arrays.

    Validates that every universe ticker is present in each named series; raises
    ``ValueError`` listing the missing tickers otherwise. Extra tickers beyond
    the universe are ignored. Returns ``None`` when ``named`` is empty so the
    compiler keeps its own "input not supplied" errors for nodes that need it.
    """
    if not named:
        return None
    out: dict[str, np.ndarray] = {}
    for name, by_ticker in named.items():
        missing = [t for t in universe if t not in by_ticker]
        if missing:
            raise ValueError(
                f"{label} {name!r} is missing values for universe tickers: {missing}."
            )
        out[name] = np.array([float(by_ticker[t]) for t in universe], dtype=float)
    return out
