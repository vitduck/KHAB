#!/usr/bin/env python3
from __future__ import annotations

import os

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from socket import gethostname
from typing import Any, Optional

from ClusterShell.NodeSet import NodeSet


@dataclass
class MPI(ABC):
    hostlist: list[str] = field(
        default_factory=lambda: list(
            NodeSet(os.environ.get("SLURM_NODELIST", gethostname()))
        )
    )

    @abstractmethod
    def build_cmd(self, spec) -> list[str]:
        """Return mpirun command as a flat list of logical tokens.

        Each element is one logical token — flag+value may be embedded in a
        single string (e.g. '--host node1:4') so fmt_cmd renders them together.
        """


@dataclass
class OpenMPI(MPI):
    """OpenMPI launcher.

    bind  — passed to --bind-to  (e.g. 'socket', 'core', 'none')
    map   — passed to --map-by   (e.g. 'socket', 'node')
    mca   — dict of MCA key/value pairs
    verbose — if true, adds --report-bindings
    """

    bind:    Optional[str] = None
    map:     Optional[str] = None
    mca:     dict[str, Any] = field(default_factory=dict)
    verbose: bool = False

    def build_cmd(self, spec) -> list[str]:
        cmd = ['mpirun']

        # Host
        hosts = self.hostlist[:spec.nnodes]
        cmd += [['--host', ','.join(f'{h}:{spec.ntasks}' for h in hosts)]]

        # Bind
        if self.bind:
            cmd += [['--bind-to', self.bind]]

        # Map
        if self.map:
            if spec.nthreads > 1:
                cmd += [['--map-by', f'{self.map}:PE={spec.nthreads}']]
            else:
                cmd += [['--map-by', self.map]]

        # MCA parameters
        for key, value in self.mca.items():
            cmd += [['--mca', key, str(value)]]

        # Environment
        for key, value in spec.env.items():
            cmd += [['-x', f'{key}={value}']]

        # Verbosity
        if self.verbose:
            cmd += ['--report-bindings']

        return cmd


@dataclass
class IntelMPI(MPI):
    """Intel MPI launcher — stub for future implementation."""
    ppn: int = 1

    def build_cmd(self, spec) -> list[str]:
        raise NotImplementedError("IntelMPI launcher not yet implemented")


@dataclass
class CrayMPICH(MPI):
    """Cray MPICH / srun launcher — stub for future implementation."""

    def build_cmd(self, spec) -> list[str]:
        raise NotImplementedError("CrayMPICH launcher not yet implemented")
