"""Read the geometry arrays of a MODFLOW-USG DISU file."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import TYPE_CHECKING

import numpy as np

from pyvista_modflow._arrays import FloatArray, IntArray

if TYPE_CHECKING:
    from collections.abc import Iterator

# DISU item 1: NODES NLAY NJAG IVSD NPER ITMUNI LENUNI IDSYMRD
_HEADER_INTS = 8
# IVSD = -1: every layer holds the same footprints, stacked.
_IVSD_LAYERED = -1
# Fortran edit descriptors: a repeated field such as 10E12.4 or 20I4, or a skip such as 1X.
# Anything else before the field, such as a 1P scale factor, does not move it.
_EDIT = re.compile(r'(\d*)([EFGDI])[NS]?(\d+)(?:\.(\d+))?|(\d+)X', re.IGNORECASE)
# A word on a control line; a quoted file name may hold spaces.
_WORD = re.compile(r"'[^']*'|\"[^\"]*\"|\S+")
# A fixed-format control record is LOCAT, CNSTNT, FMTIN, IPRN in columns I10, F10, A20, I10.
_LOCAT, _CNSTNT = slice(0, 10), slice(10, 20)


@dataclass(frozen=True, eq=False)
class Disu:
    """The geometry of a MODFLOW-USG unstructured discretization (DISU).

    Attributes
    ----------
    ivsd : int
        Vertical sub-discretization flag; ``-1`` means every layer holds the same
        footprints, stacked.
    nodelay : numpy.ndarray
        Number of nodes in each layer.
    top : numpy.ndarray
        Top elevation of every node.
    bot : numpy.ndarray
        Bottom elevation of every node.
    area : numpy.ndarray
        Footprint area of every node.

    """

    ivsd: int
    nodelay: IntArray
    top: FloatArray
    bot: FloatArray
    area: FloatArray

    @property
    def layered(self) -> bool:
        """Return whether every layer holds the same footprints.

        Returns
        -------
        bool
            ``True`` when ``IVSD = -1``.

        """
        return self.ivsd == _IVSD_LAYERED

    @property
    def layer(self) -> IntArray:
        """Return the layer of every node.

        Returns
        -------
        numpy.ndarray
            The 1-based layer of every node, in node order.

        """
        return np.repeat(np.arange(1, self.nodelay.size + 1), self.nodelay)


def _number(text: str) -> float:
    """Parse one Fortran number: ``D`` exponents allowed, a blank field is zero."""
    text = text.strip()
    return float(text.replace('D', 'E').replace('d', 'e')) if text else 0.0


def _repeat(token: str) -> list[str]:
    """Expand a Fortran ``n*value`` repeat."""
    count, star, value = token.partition('*')
    return [value] * int(count) if star else [token]


def _read_free(lines: Iterator[str], n: int) -> FloatArray:
    """Read ``n`` list-directed values; the rest of the line after the last one is ignored."""
    chunks: list[FloatArray] = []
    count = 0
    while count < n:
        tokens = next(lines).replace('D', 'E').replace('d', 'e').replace(',', ' ').split()
        if '*' in ''.join(tokens[: n - count]):
            tokens = [v for t in tokens for v in _repeat(t)]
        chunks.append(np.array(tokens[: n - count], dtype=float))
        count += chunks[-1].size
    return np.concatenate(chunks)


def _fixed_layout(fmt: str) -> tuple[int, int, int, int] | None:
    """Return a fixed format's leading skip, fields per line, width and implied decimals."""
    skip = 0
    for match in _EDIT.finditer(fmt):
        count, kind, width, decimals, spaces = match.groups()
        if spaces:
            skip += int(spaces)
        elif kind:
            implied = int(decimals or 0) if kind.upper() != 'I' else 0
            return skip, int(count or 1), int(width), implied
    return None  # (FREE)


def _read_fixed(lines: Iterator[str], n: int, layout: tuple[int, int, int, int]) -> FloatArray:
    """Read ``n`` fixed-width values; a real field without a point has its decimals implied."""
    skip, per_line, width, decimals = layout
    parts: list[str] = []
    while len(parts) < n:
        line = next(lines).rstrip('\n')
        take = min(per_line, n - len(parts))
        parts.extend(line[skip + i * width : skip + (i + 1) * width] for i in range(take))
    scale = 10.0**-decimals
    return np.array([_number(p) * (1.0 if '.' in p else scale) for p in parts])


def _read_values(lines: Iterator[str], n: int, fmt: str, where: str) -> FloatArray:
    """Read ``n`` values in FREE or a Fortran fixed-width format."""
    layout = _fixed_layout(fmt)
    try:
        return _read_free(lines, n) if layout is None else _read_fixed(lines, n, layout)
    except StopIteration:
        msg = f'{where}: file ended while reading an array'
        raise ValueError(msg) from None
    except ValueError as err:
        msg = f'{where}: {err}'
        raise ValueError(msg) from None


def _multiplier(text: str) -> float:
    """Return a CNSTNT multiplier; MODFLOW skips the multiplication when it is zero."""
    value = _number(text)
    return value if value != 0 else 1.0


def _read_array(lines: Iterator[str], n: int, name: str, path: Path) -> FloatArray:
    """Read one MODFLOW array-control block of ``n`` values."""
    raw = next(lines)
    head = _WORD.findall(raw)
    where = f'{path}: {name}'
    if not head:
        msg = f'{where}: empty array-control line'
        raise ValueError(msg)
    kind = head[0].upper()
    if kind == 'CONSTANT':
        return np.full(n, _number(head[1]))
    if kind == 'INTERNAL':
        fmt = (head[2:3] or ['(FREE)'])[0]
        return _multiplier(head[1]) * _read_values(lines, n, fmt, where)
    if kind == 'OPEN/CLOSE':
        fmt = (head[3:4] or ['(FREE)'])[0]
        if '(BINARY)' in fmt.upper():
            msg = f'{where}: binary OPEN/CLOSE arrays are not supported'
            raise NotImplementedError(msg)
        # Models written on Windows separate directories with backslashes.
        external = path.parent / head[1].strip('\'"').replace('\\', '/')
        with external.open(encoding='utf-8-sig', errors='replace') as f:
            return _multiplier(head[2]) * _read_values(iter(f), n, fmt, str(external))
    if kind.lstrip('-').isdigit():
        locat = int(raw[_LOCAT])
        if locat == 0:
            return np.full(n, _number(raw[_CNSTNT]))
        msg = f'{where}: a unit-number array control (LOCAT {locat}) needs the name file'
        raise NotImplementedError(msg)
    msg = f'{where}: unsupported array control {head[0]!r}'
    raise NotImplementedError(msg)


def _read_layers(lines: Iterator[str], nodelay: IntArray, name: str, path: Path) -> FloatArray:
    """Read one array per layer and join them in node order."""
    return np.concatenate(
        [_read_array(lines, n, f'{name} layer {k + 1}', path) for k, n in enumerate(nodelay)]
    )


def read_disu(path: str | Path) -> Disu:
    """Read the node counts and the TOP, BOT and AREA arrays of a MODFLOW-USG DISU file.

    Parameters
    ----------
    path : str or pathlib.Path
        The DISU file.

    Returns
    -------
    Disu
        The geometry arrays, one value per node.

    Raises
    ------
    ValueError
        If the file is not a MODFLOW-USG DISU file or its arrays are inconsistent.
    NotImplementedError
        If the file uses an input this reader does not support: MODFLOW 6, confining
        beds, binary arrays or unit-number array controls.

    Examples
    --------
    >>> import pyvista_modflow as pm
    >>> disu = pm.read_disu('model.disu')  # doctest: +SKIP
    >>> thickness = disu.top - disu.bot  # doctest: +SKIP

    """
    path = Path(path)
    # Numbers are ASCII; a byte-order mark or a comment in a Windows code page must not fail.
    with path.open(encoding='utf-8-sig', errors='replace') as f:
        try:
            return _read_disu((ln for ln in f if not ln.lstrip().startswith('#')), path)
        except StopIteration:
            msg = f'{path}: file ended early'
            raise ValueError(msg) from None


def _read_disu(lines: Iterator[str], path: Path) -> Disu:
    """Read a DISU from its non-comment lines."""
    fields = next(lines).split()
    if fields and fields[0].upper() == 'BEGIN':
        msg = f'{path}: MODFLOW 6 input is not supported yet; only MODFLOW-USG DISU is read'
        raise NotImplementedError(msg)
    try:
        header = [int(v) for v in fields[:_HEADER_INTS]]
    except ValueError:
        header = []
    if len(header) < _HEADER_INTS:
        msg = (
            f'{path}: not a MODFLOW-USG DISU file (item 1 must be NODES NLAY NJAG IVSD '
            'NPER ITMUNI LENUNI IDSYMRD); structured DIS files are not supported'
        )
        raise ValueError(msg)
    nnode, nlay, _, ivsd = header[:4]
    laycbd: list[str] = []
    while len(laycbd) < nlay:
        laycbd += next(lines).split()
    try:
        confined = any(int(c) for c in laycbd[:nlay])
    except ValueError:
        msg = f'{path}: LAYCBD must be {nlay} integers, found {laycbd[:nlay]}'
        raise ValueError(msg) from None
    if confined:
        msg = f'{path}: quasi-3D confining beds (LAYCBD != 0) are not supported'
        raise NotImplementedError(msg)
    nodelay = _read_array(lines, nlay, 'NODELAY', path).astype(np.int64)
    if nodelay.sum() != nnode:
        msg = f'{path}: NODELAY sums to {nodelay.sum()}, item 1 says {nnode} nodes'
        raise ValueError(msg)
    top = _read_layers(lines, nodelay, 'TOP', path)
    bot = _read_layers(lines, nodelay, 'BOT', path)
    if ivsd == _IVSD_LAYERED:
        if np.any(nodelay != nodelay[0]):
            msg = f'{path}: IVSD = -1 but the layers hold different node counts'
            raise ValueError(msg)
        # Every layer shares one set of footprints, so AREA is given for one layer.
        area = np.tile(_read_array(lines, int(nodelay[0]), 'AREA', path), nlay)
    else:
        area = _read_layers(lines, nodelay, 'AREA', path)
    return Disu(ivsd=ivsd, nodelay=nodelay, top=top, bot=bot, area=area)
