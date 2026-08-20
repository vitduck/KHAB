#!/usr/bin/env python3
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Launcher(ABC):
    bin: str = ''

    @abstractmethod
    def build_cmd(self, spec) -> list[list[str]]:
        """Return command as a nested list of token groups.

        Each inner list is one logical sub-command (env vars, mpirun,
        singularity + wrapper, etc.). The nesting drives fmt_cmd indentation
        and is flattened by sys_cmd for subprocess execution.
        """
