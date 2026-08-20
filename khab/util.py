#!/usr/bin/env python3
from __future__ import annotations

import os
import shlex
import subprocess

from typing import Optional

CommandToken = str | list[str]
CommandBlock = list[CommandToken]
Command = list[CommandBlock]


def _is_env_assignment(token: str) -> bool:
    key, sep, _ = token.partition('=')
    return bool(sep and key and key.replace('_', '').isalnum() and not key[0].isdigit())


def _flatten_block(block: CommandBlock) -> list[str]:
    flat = []
    for token in block:
        if isinstance(token, list):
            flat.extend(str(item) for item in token)
        else:
            flat.append(str(token))
    return flat


def _flatten_cmd(cmd: Command) -> list[str]:
    flat = []
    for block in cmd:
        flat.extend(_flatten_block(block))
    return flat


def _extract_env_prefix(flat: list[str]) -> tuple[dict[str, str], list[str]]:
    env = {}
    i = 0
    while i < len(flat) and _is_env_assignment(flat[i]):
        key, value = flat[i].split('=', 1)
        env[key] = value
        i += 1
    return env, flat[i:]


def sys_cmd(cmd: Command, output: Optional[str] = None) -> int:
    flat = _flatten_cmd(cmd)
    env_prefix, flat = _extract_env_prefix(flat)
    env = {**os.environ, **env_prefix} if env_prefix else None

    if output is None:
        process = subprocess.Popen(
            flat,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            env=env,
        )
        return process.wait()

    stderr_tmp = f'{output}.err.tmp'
    stderr_path = f'{output}.err'

    try:
        with open(output, 'w', buffering=1) as out, \
             open(stderr_tmp, 'w', buffering=1) as err:
            process = subprocess.Popen(
                flat,
                stdout=out,
                stderr=err,
                text=True,
                env=env,
            )
            rc = process.wait()
    except KeyboardInterrupt:
        try:
            process.terminate()
            process.wait()
        except UnboundLocalError:
            pass
        if os.path.exists(stderr_tmp):
            os.replace(stderr_tmp, stderr_path)
        raise

    if rc == 0:
        if os.path.exists(stderr_tmp):
            os.remove(stderr_tmp)
    else:
        os.replace(stderr_tmp, stderr_path)

    return rc


def _quote(token: str) -> str:
    return shlex.quote(str(token))


def _is_option(token: str) -> bool:
    return token.startswith('-') and token != '-'


def _segment(token: CommandToken) -> str:
    if isinstance(token, list):
        return ' '.join(_quote(item) for item in token)
    return _quote(token)

def _split_block(tokens: CommandBlock) -> list[str]:
    """Split one structured argv block into display lines.

    Rules:
      - First token (command scalar): own line, inherits block indent level.
      - Remaining tokens (key-value lists or scalar options): each on own line,
        one indent level deeper.
    """
    if not tokens:
        return []
    return [_segment(token) for token in tokens]

def fmt_cmd(cmd: Command, indent: int = 4) -> str:
    if not cmd:
        return ''

    lines: list[str] = []

    for level, block in enumerate(cmd):
        segments = _split_block(block)
        for i, segment in enumerate(segments):
            # i==0: command token, indent = level
            # i >0: option token, indent = level + 1
            pad = ' ' * (indent * (level + (1 if i else 0)))
            lines.append(pad + segment)

    for i in range(len(lines) - 1):
        lines[i] += ' \\'

    return '\n'.join(lines)
