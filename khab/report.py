#!/usr/bin/env python3
from __future__ import annotations

import os
import sys

from collections import defaultdict
from statistics import mean
from typing import Any, Literal

from rich import box
from rich.console import Console
from rich.table import Table

from cpu import CPU
from env import module_list
from gpu import GPU


class _Tee:
    """Write-through to multiple file-like objects simultaneously."""

    def __init__(self, *files):
        self._files = files

    def write(self, data: str) -> None:
        for f in self._files:
            f.write(data)

    def flush(self) -> None:
        for f in self._files:
            f.flush()


def fmt(v: Any) -> str:
    if isinstance(v, int) and v == 0:
        return '-'
    return f"{v:.1f}" if isinstance(v, float) else str(v)


def _resolve(result, header: str) -> Any:
    """Resolve a header name to a value from spec fields then metrics."""
    spec = result.spec
    if hasattr(spec, header):
        return getattr(spec, header)
    return result.metrics.get(header, '')


def _make_console(file=None) -> Console:
    return Console(file=file, no_color=True, highlight=False)


def _is_numeric_cell(cell: str) -> bool:
    if cell == '-':
        return True
    try:
        float(cell)
        return True
    except ValueError:
        return False


def _make_result_table(headers: list[str], rows: list[list[str]], justify: list[str] | None = None) -> Table:
    table = Table(
        box=box.SIMPLE,
        show_lines=False,
        header_style=None,
        border_style=None,
        style=None,
        pad_edge=True,
    )
    if justify is None:
        justify = ['left'] * len(headers)
    for header, align in zip(headers, justify):
        table.add_column(header, justify=align, style=None, no_wrap=False)
    for row in rows:
        table.add_row(*row)
    return table


def _make_grid_table(rows: list[list[str]], justify: list[str] | None = None) -> Table:
    ncols = max((len(row) for row in rows), default=0)
    table = Table.grid(padding=(0, 2))
    if justify is None:
        justify = ['left'] * ncols
    for align in justify[:ncols]:
        table.add_column(justify=align, style=None, no_wrap=False)
    for row in rows:
        table.add_row(*row)
    return table


def _label_outputs(outdir: str) -> list[list[str]]:
    return [[os.path.join(outdir, f)] for f in sorted(os.listdir(outdir))]


def _print_info(outdir: str, cpu: CPU, gpu: GPU, console: Console) -> None:
    modules = module_list()
    hw = [('host', os.uname().nodename), ('cpu', f"{cpu.sockets} x {cpu.name}")]
    if gpu.count > 0:
        hw.append(('gpu', f"{gpu.count} x {gpu.name}"))
    if modules:
        hw.append(('env', modules))

    console.print()
    console.print('[info]', markup=False)
    console.print(_make_grid_table([[k, v] for k, v in hw]))
    console.print()
    console.print('[outputs]', markup=False)
    console.print(_make_grid_table(_label_outputs(outdir)))
    console.print()


def _make_data_row(row: list, n_keys: int, show_keys: bool, idx: str) -> list[str]:
    """Build one data row.

    show_keys=True  — first repeat: all key columns shown.
    show_keys=False — subsequent repeats: key columns blank.
    idx             — repeat index string ("1", "2", …) or "avg".
    All cells are pre-formatted as strings (numbers to 1 d.p.).
    """
    key_cells = [fmt(row[i]) for i in range(n_keys)] if show_keys else [''] * n_keys
    metric_cells = [fmt(row[i]) for i in range(n_keys, len(row))]
    return key_cells + [idx] + metric_cells


def _make_avg_row(grp: list, n_keys: int, n_metrics: int) -> list[str]:
    avg_cells = [''] * n_keys + ['<#>']
    for i in range(n_keys, n_keys + n_metrics):
        vals = [row[i] for _, row in grp]
        numeric = [float(v) for v in vals if isinstance(v, (int, float)) or (isinstance(v, str) and v)]
        avg_cells.append(fmt(mean(numeric)) if numeric else '')
    avg_cells.append(fmt(mean(r.walltime for r, _ in grp)))
    return avg_cells


def _infer_colalign(first_row: list[str], n_keys: int) -> list[str]:
    """Infer per-column alignment from the first data row."""
    result = []
    for i, cell in enumerate(first_row):
        if i == n_keys:
            result.append('center')
        elif _is_numeric_cell(cell):
            result.append('right')
        else:
            result.append('left')
    return result


def _apply_header_aliases(headers: list[str], aliases: dict[str, str]) -> list[str]:
    return [aliases.get(h, h) for h in headers]


def _print_table(col_headers: list[str], table_rows: list[list[str]], n_keys: int, console: Console) -> None:
    """Print the main report table using Rich's neutral ASCII table."""
    if not table_rows:
        return
    colalign = _infer_colalign(table_rows[0], n_keys)
    console.print(_make_result_table(col_headers, table_rows, colalign))


def _print_legend(
    col_headers: list[str],
    aliases: dict[str, str],
    notes: dict[str, str],
    console: Console,
) -> None:
    active = []
    for header in col_headers:
        alias = aliases.get(header)
        note = notes.get(alias) if alias else None
        if alias and note:
            active.append([alias, note])

    if not active:
        return

    console.print()
    console.print('[legend]', markup=False)
    console.print(_make_grid_table([['key', 'value'], *active], ['left', 'left']))


def _build_table(
    results: list,
    rows: list[list],
    key_headers: list[str],
    metric_headers: list[str],
) -> tuple[list[str], list[list[str]], int]:
    grouped: dict[str, list] = defaultdict(list)
    for r, row in zip(results, rows):
        grouped[r.output.rsplit('.', 1)[0]].append((r, row))

    n_keys = len(key_headers)
    n_metrics = len(metric_headers)
    col_headers = key_headers + ['#'] + metric_headers + ['walltime (s)']

    table_rows = []
    for _, grp in grouped.items():
        for i, (_, row) in enumerate(grp):
            table_rows.append(_make_data_row(row, n_keys, show_keys=(i == 0), idx=str(i + 1)))
        if len(grp) > 1:
            table_rows.append(_make_avg_row(grp, n_keys, n_metrics))

    return col_headers, table_rows, n_keys


def report(
    results: list,
    sort: bool = False,
    order: Literal['<', '>'] = '<',
) -> None:
    if not results:
        return

    config_ref = results[0].config
    key_headers = config_ref.key_headers
    metric_headers = config_ref.metric_headers
    header_aliases = getattr(config_ref, 'header_aliases', {}) or {}
    header_notes = getattr(config_ref, 'header_notes', {}) or {}

    all_headers = key_headers + metric_headers
    rows = [[_resolve(r, h) for h in all_headers] + [r.walltime] for r in results]

    if sort:
        rows = sorted(rows, key=lambda x: float(x[-1]), reverse=(order == '>'))

    outdir = os.path.dirname(results[0].output)
    report_path = os.path.join(outdir, 'report.txt')

    with open(report_path, 'w') as f:
        out = _Tee(sys.stdout, f)
        console = _make_console(file=out)

        _print_info(
            outdir=outdir,
            cpu=config_ref._cpu,
            gpu=config_ref._gpu,
            console=console,
        )

        col_headers, table_rows, n_keys = _build_table(results, rows, key_headers, metric_headers)
        display_headers = _apply_header_aliases(col_headers, header_aliases)

        console.print('[result]', markup=False)
        _print_table(display_headers, table_rows, n_keys, console=console)
        _print_legend(col_headers, header_aliases, header_notes, console=console)
        console.print()
