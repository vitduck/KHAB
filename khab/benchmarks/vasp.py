#!/usr/bin/env python3

import itertools
import os
import re

from dataclasses import dataclass, field
from shutil import copy
from typing import Any, ClassVar

from bmt import BmtConfig, RunSpec, _as_list
from launcher import Launcher
from mpi import MPI, OpenMPI


# ---------------------------------------------------------------------------
# INCARHandler — internal utility, not exposed to user
# ---------------------------------------------------------------------------

@dataclass
class INCARHandler:
    """Reads the baseline ./INCAR and overlays per-spec sweep parameters.

    The user edits ./INCAR directly for all fixed tags (ENCUT, EDIFF, etc.).
    The framework calls set() for sweep axes (NCORE, KPAR, NSIM) and
    write() to flush ./INCAR before each run.
    """

    def __post_init__(self):
        if not os.path.exists('INCAR'):
            raise FileNotFoundError("INCAR not found in working directory")
        with open('INCAR') as f:
            self._baseline = f.read()
        self._content = self._baseline
        self._new_params: dict[str, str] = {}

    def reset(self):
        """Restore content to baseline — call before each spec."""
        self._content = self._baseline
        self._new_params = {}

    def set(self, key: str, value: Any) -> None:
        """Set or add a key in INCAR content."""
        m = re.search(rf'^{key}\s*=\s*(.+)$', self._content, flags=re.MULTILINE)
        if m:
            if m.group(1).strip() == str(value):
                return
            self._content = re.sub(
                rf'^{key}\s*=\s*(.+)$',
                f'{key} = {value}',
                self._content,
                count=1,
                flags=re.MULTILINE,
            )
        else:
            self._new_params[key] = str(value)

    def write(self) -> None:
        """Flush current content to ./INCAR."""
        out = self._content
        if self._new_params:
            out += '\n'
            for k, v in self._new_params.items():
                out += f'{k} = {v}\n'
        with open('INCAR', 'w') as f:
            f.write(out)


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VaspSpec(RunSpec):
    ncore: int = 1
    kpar: int = 1
    nsim: int = 4


# ---------------------------------------------------------------------------
# Launcher
# ---------------------------------------------------------------------------

@dataclass
class VaspLauncher(Launcher):
    """Composes MPI + bare VASP binary."""
    mpi: MPI = field(default_factory=OpenMPI)

    def build_cmd(self, spec: VaspSpec) -> list[list[str]]:
        return [self.mpi.build_cmd(spec), [self.bin]]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class VaspConfig(BmtConfig):
    name: ClassVar[str] = 'VASP'
    key_headers: list = field(default_factory=lambda: [
        'nnodes', 'ntasks', 'nthreads', 'ngpus',
        'ncore', 'kpar', 'nsim',
    ])
    metric_headers: list = field(default_factory=list)

    bin: str = 'vasp_std'
    outdir: str = './output'

    launcher: VaspLauncher = field(default_factory=VaspLauncher)

    # Parameter spaces — plural, sweepable
    ncores: int | list[int] = 1
    kpars: int | list[int] = 1
    nsims: int | list[int] = 4

    def __post_init__(self):
        super().__post_init__()
        self.variant = 'GPU' if any(n > 0 for n in _as_list(self.ngpus)) else 'CPU'
        self._incar = INCARHandler()

    def _validate_config(self) -> None:
        ntasks = _as_list(self.ntasks)
        ngpus  = _as_list(self.ngpus)
        if len(ntasks) != len(ngpus):
            raise ValueError(
                f"ntasks and ngpus must have the same length "
                f"(got {len(ntasks)} vs {len(ngpus)}) -- they are zipped, not crossed"
            )

    def __iter__(self):
        tasks_gpus = zip(_as_list(self.ntasks), _as_list(self.ngpus))
        for nnodes, (ntasks, ngpus), nthreads, ncore, kpar, nsim in itertools.product(
            _as_list(self.nnodes),
            tasks_gpus,
            _as_list(self.nthreads),
            _as_list(self.ncores),
            _as_list(self.kpars),
            _as_list(self.nsims),
        ):
            if ngpus > 0 and kpar > ngpus:
                continue

            spec = VaspSpec(
                nnodes=nnodes,
                ntasks=ntasks,
                ngpus=ngpus,
                nthreads=nthreads,
                ncore=ncore,
                kpar=kpar,
                nsim=nsim,
            )
            self._validate(spec)

            yield spec

    def _output_path(self, spec: VaspSpec, n: int, outdir: str) -> str:
        tag = (
            f"VASP-{self.host}"
            f"-n{spec.nnodes}"
            f"-t{spec.ntasks}"
            f"-o{spec.nthreads}"
            f"-g{spec.ngpus}"
            f"-NCORE{spec.ncore}"
            f"-KPAR{spec.kpar}"
            f"-NSIM{spec.nsim}"
        )
        return os.path.join(outdir, f"{tag}.out.{n}")

    def _pre_run(self, spec: VaspSpec, n: int, output_path: str) -> None:
        """Write INCAR with spec parameters just before each run."""
        stem = output_path.replace(f'.out.{n}', '')
        self._incar.reset()
        self._incar.set('NCORE', spec.ncore)
        self._incar.set('KPAR',  spec.kpar)
        self._incar.set('NSIM',  spec.nsim)
        self._incar.write()
        copy('INCAR', f'{stem}.incar')

    def _post_run(self, spec: VaspSpec, n: int, output_path: str) -> None:
        """Archive OUTCAR and OSZICAR with spec stem after each repeat."""
        stem = output_path.replace(f'.out.{n}', '')
        copy('OUTCAR',  f'{stem}.outcar.{n}')
        copy('OSZICAR', f'{stem}.oszicar.{n}')

    def parse(self, output_path: str) -> dict[str, Any]:
        return {}
