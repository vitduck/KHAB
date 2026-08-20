#!/usr/bin/env bash

_KHAB_SCRIPT="${BASH_SOURCE[0]}"

_KHAB_ROOT="$(cd "$(dirname "${_KHAB_SCRIPT}")" && pwd)"

export PYTHONPATH="${_KHAB_ROOT}/khab:${_KHAB_ROOT}/khab/benchmarks:${PYTHONPATH}"

export PYTHONDONTWRITEBYTECODE=1

echo "khab: PYTHONPATH updated for this shell session."
echo "  ${_KHAB_ROOT}/khab"
echo "  ${_KHAB_ROOT}/khab/benchmarks"

unset _KHAB_SCRIPT _KHAB_ROOT
