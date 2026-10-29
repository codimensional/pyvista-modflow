"""Vectorized helpers for ragged arrays and sort-based uniques."""

from __future__ import annotations

from typing import TypeVar

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
ScalarT = TypeVar('ScalarT', bound=np.generic)


def ragged_positions(starts: npt.ArrayLike, lengths: IntArray) -> IntArray:
    """Return ``start[i] + 0 .. length[i] - 1`` for every ``i``, concatenated."""
    starts = np.asarray(starts, np.int64)
    return np.repeat(starts - (np.cumsum(lengths) - lengths), lengths) + np.arange(lengths.sum())


def offsets(lengths: IntArray) -> IntArray:
    """Return the ``n + 1`` offsets of ``n`` consecutive runs of ``lengths``."""
    return np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)


# np.unique is several times slower than a plain sort on NumPy 2.x, so these sort directly.
def unique(x: npt.NDArray[ScalarT]) -> npt.NDArray[ScalarT]:
    """Return the sorted distinct values of a 1D array."""
    s = np.sort(x)
    keep = np.empty(s.size, bool)
    keep[:1] = True
    np.not_equal(s[1:], s[:-1], out=keep[1:])
    return s[keep]


def unique_rows(*keys: npt.NDArray[np.generic]) -> tuple[IntArray, IntArray]:
    """Return the first index of each distinct row of ``keys`` and every row's rank.

    Rows sort by the first key, then the next.
    """
    order = np.lexsort(keys[::-1])
    new = np.zeros(order.size, bool)
    for k in keys:
        s = k[order]
        new[1:] |= s[1:] != s[:-1]
    inverse = np.empty(order.size, np.int64)
    inverse[order] = np.cumsum(new)
    return order[np.flatnonzero(np.concatenate([[True], new[1:]]))], inverse
