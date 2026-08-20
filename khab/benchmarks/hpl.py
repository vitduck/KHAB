#!/usr/bin/env python3
from __future__ import annotations

import datetime
import math
import os
import re

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

from bmt import BmtConfig, RunSpec, _as_list, benchmark
from cpu import CPU
from gpu import GPU
from launcher import Launcher
from mpi import MPI, OpenMPI


# ---------------------------------------------------------------------------
# HplSpec -- outer sweep point, frozen into each BmtResult
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HplSpec(RunSpec):
    """A single HPL run point -- fully specified, one HPL.dat per instance.

    All fields are frozen so each result carries the exact parameters run.
    nnodes, ntasks, ngpus, threads inherited from RunSpec.
    """
    n:     int = 0
    nb:    int = 0
    p:     int = 0
    q:     int = 0
    bcast: int = 0


# ---------------------------------------------------------------------------
# Backend ABC -- hardware-specific launcher fragment + parameter defaults
# ---------------------------------------------------------------------------

@dataclass
class Backend(ABC):
    """Encapsulates everything that differs between hardware targets.

    Subclasses declare whichever path field they need:
      - bin  -- bare binary path (CpuHPL, AmdCPU, IntelCPU, ...)
      - ngc  -- NGC/ROCm container image path (NvidiaLegacy, Nvidia, AmdGPU, ...)

    Device discovery objects (cpu, gpu) are injected by HplConfig after init.
    """

    bin: str = ''  # bare binary path; empty for container-based backends

    # Injected by HplConfig._resolve_defaults() after hardware discovery
    cpu: Any = field(init=False, default=None, repr=False)
    gpu: Any = field(init=False, default=None, repr=False)

    @abstractmethod
    def hpl_cmd(self, spec: HplSpec, dat_path: str) -> list[list[str]]:
        """Return command blocks as nested token lists."""

    @abstractmethod
    def _build_blocksize(self) -> int:
        """Return the hardware-tuned default NB."""

    def _validate_config(self) -> None:
        """Config-level validation hook -- called by HplConfig._validate_config().

        Override in subclasses to check backend-specific invariants.
        """


# ---------------------------------------------------------------------------
# CpuHPL -- bare xhpl binary, no container
# ---------------------------------------------------------------------------

@dataclass
class CpuHPL(Backend):
    """CPU-only HPL -- invokes the xhpl binary directly.

    Uses inherited bin field.
    No container, no GPU affinity. blocksize default is conservative;
    tune per architecture (e.g. 232 for Intel, 256 for AMD).
    """

    def _validate_config(self) -> None:
        if not self.bin:
            raise ValueError("CpuHPL requires bin= path to the xhpl binary")

    def hpl_cmd(self, spec: HplSpec, dat_path: str) -> list[list[str]]:
        # TODO: pass --dat dat_path once CpuHPL supports it
        return [[self.bin]]

    def _build_blocksize(self) -> int:
        return 232


# ---------------------------------------------------------------------------
# NvidiaLegacy -- NGC v21.04  (V100)
# ---------------------------------------------------------------------------

@dataclass
class NvidiaLegacy(Backend):
    """NVIDIA NGC v21.04 HPL container -- targets V100 GPUs.

    --cpu-cores-per-rank is always emitted, sourced from spec.nthreads.
    ngc is the path to the NGC container image (.sif).
    bin is unused (empty).
    """

    ngc: str = ''

    def _validate_config(self) -> None:
        if not self.ngc:
            raise ValueError("NvidiaLegacy requires ngc= container image path")

    def hpl_cmd(self, spec: HplSpec, dat_path: str) -> list[list[str]]:
        selected     = list(range(spec.ngpus))
        cpu_affinity = ':'.join(str(n) for n in self.gpu.affinity(gpu_ids=selected))
        gpu_affinity = ':'.join(str(i) for i in selected)

        singularity = ['singularity', 'run', '--nv', self.ngc]

        wrapper = [
            '/workspace/hpl.sh',
            ['--dat', dat_path],
            ['--cpu-affinity', cpu_affinity],
            ['--mem-affinity', cpu_affinity],
            ['--gpu-affinity', gpu_affinity],
            ['--cpu-cores-per-rank', str(spec.nthreads)],
        ]

        return [singularity, wrapper]

    def _build_blocksize(self) -> int:
        return 288


# ---------------------------------------------------------------------------
# Nvidia -- NGC v24.09  (A100, H100, H200, GH200)
# ---------------------------------------------------------------------------

@dataclass
class Nvidia(Backend):
    """NVIDIA NGC v24.09 HPL container -- targets A/H-generation GPUs.

    ngc is the path to the NGC container image (.sif).
    bin is unused (empty).
    """

    ngc: str = ''

    def _validate_config(self) -> None:
        if not self.ngc:
            raise ValueError("NvidiaLegacy requires ngc= container image path")

    def hpl_cmd(self, spec: HplSpec, dat_path: str) -> list[list[str]]:
        selected     = list(range(spec.ngpus))
        cpu_affinity = ':'.join(str(n) for n in self.gpu.affinity(gpu_ids=selected))
        gpu_affinity = ':'.join(str(i) for i in selected)

        singularity = ['singularity', 'run', '--nv', self.ngc]

        wrapper = [
            '/workspace/hpl.sh',
            ['--dat', dat_path],
            ['--cpu-affinity', cpu_affinity],
            ['--mem-affinity', cpu_affinity],
            ['--gpu-affinity', gpu_affinity],
        ]

        if spec.nnodes == 1:
            wrapper += ['--no-multinode']

        return [singularity, wrapper]

    def _build_blocksize(self) -> int:
        name = self.gpu.name
        if re.search(r'A100', name):
            return 384
        elif re.search(r'H100|H200|GH200', name):
            return 1024
        else:
            return 256


# ---------------------------------------------------------------------------
# Backend factory -- select from discovered hardware
# ---------------------------------------------------------------------------

def make_backend(gpu, ngc: str = "") -> Backend:
    """Select Backend from discovered GPU name.

    For CPU-only runs, pass an explicit backend=CpuHPL(ngc=...) to HplConfig
    rather than relying on auto-detection.
    """
    name = gpu.name
    if re.search(r'V100', name):
        return NvidiaLegacy(ngc=ngc)
    elif re.search(r'[AH]100|H200|GH200', name):
        return Nvidia(ngc=ngc)
    else:
        raise ValueError(
            f"No HPL backend for GPU '{name}'. "
            "Pass an explicit backend= to HplConfig."
        )


# ---------------------------------------------------------------------------
# HplLauncher -- composes MPI + Backend, writes HPL.dat before each run
# ---------------------------------------------------------------------------

@dataclass
class HplLauncher(Launcher):
    """Builds the full HPL command for one outer sweep point.

    Sequence per build_cmd(spec) call:
      1. Write HPL.dat  (N from spec; NB, grids, algo params from config)
      2. mpi.build_cmd(spec)   -> mpirun tokens
      3. backend.hpl_cmd(spec) -> singularity + hpl.sh tokens
    """

    mpi: MPI = field(default_factory=OpenMPI)
    backend: Backend = field(default=None)

    # Reference back to config -- set by HplConfig.__post_init__
    _config: Any = field(init=False, default=None, repr=False)

    def build_cmd(self, spec: HplSpec) -> list[list[str]]:
        dat_path = self._write_dat(spec)
        mpi_block = self.mpi.build_cmd(spec)
        return [mpi_block] + self.backend.hpl_cmd(spec, dat_path)

    def _write_dat(self, spec: HplSpec) -> str:
        """Write HPL.dat to outdir with the same stem as the output file.

        Returns the dat path so build_cmd can forward it to hpl_cmd.
        CpuHPL still writes HPL.dat in the working directory (deferred).
        """
        cfg      = self._config
        out_path = cfg._output_path(spec, 1, cfg.outdir)                    # stem only -- repeat index irrelevant
        dat_path = re.sub(r'\.out\.\d+$', '.dat', out_path)    # replace .out.N suffix

        dat = f"""\
HPLinpack
Innovative Computing Laboratory
{'HPL.out':<20} output file name (if any)
{6:<20} device out (6=stdout,7=stderr,file)
{1:<20} # of problems sizes (N)
{spec.n:<20} Ns
{1:<20} # of NBs
{spec.nb:<20} NBs
{cfg.pmap:<20} PMAP process mapping (0=Row-,1=Column-major)
{1:<20} # of process grids (P x Q)
{spec.p:<20} Ps
{spec.q:<20} Qs
{16.0:<20} threshold
{1:<20} # of panel fact
{cfg.pfact:<20} PFACTs (0=left, 1=Crout, 2=Right)
{1:<20} # of recursive stopping criterium
{cfg.nbmin:<20} NBMINs (>= 1)
{1:<20} # of panels in recursion
{cfg.ndiv:<20} NDIVs
{1:<20} # of recursive panel fact.
{cfg.rfact:<20} RFACTs (0=left, 1=Crout, 2=Right)
{1:<20} # of broadcast
{spec.bcast:<20} BCASTs (0=1rg,1=1rM,2=2rg,3=2rM,4=Lng,5=LnM)
{1:<20} # of lookahead depth
{1:<20} DEPTHs (>=0)
{1:<20} SWAP (0=bin-exch,1=long,2=mix)
{192:<20} swapping threshold
{1:<20} L1 in (0=transposed,1=no-transposed) form
{0:<20} U  in (0=transposed,1=no-transposed) form
{0:<20} Equilibration (0=no,1=yes)
{8:<20} memory alignment in double (> 0)
"""
        with open(dat_path, 'w') as f:
            f.write(dat)
        return dat_path


# ---------------------------------------------------------------------------
# Helpers -- pure functions, no class state
# ---------------------------------------------------------------------------

def _build_grid(nnodes: int, ngpus: int) -> tuple[int, int]:
    """Return the most-square (P, Q) grid with P >= Q for GPU runs."""
    n = nnodes * ngpus
    factors = [(i, n // i) for i in range(1, int(math.sqrt(n)) + 1) if n % i == 0]
    p, q = min(factors, key=lambda x: abs(x[0] - x[1]))
    # P >= Q convention for GPU layouts
    if p < q:
        p, q = q, p
    return (p, q)


def _build_size(nnodes: int, ngpus: int, blocksize: int,
                mem_per_gpu: int, mem_frac: float) -> int:
    """Compute N from total GPU memory across all ranks."""
    total = mem_frac * mem_per_gpu * nnodes * ngpus
    n = int((total / 8.0) ** 0.5)
    # Round down to nearest multiple of blocksize
    return (n // blocksize) * blocksize


def _parse_mem(mem: str, mem_per_gpu: int) -> tuple[int, float]:
    """Return (mem_per_gpu_bytes, fraction) from a mem spec like '80%' or '32GiB'."""
    if mem.endswith('%'):
        frac = float(mem.rstrip('%')) / 100.0
        return mem_per_gpu, frac

    match = re.match(r'([\d.]+)\s*(GiB|GB|MiB|MB)', mem, re.IGNORECASE)
    if not match:
        raise ValueError(f"Invalid mem format: '{mem}'. Use '80%', '32GiB', etc.")

    value, unit = float(match.group(1)), match.group(2).upper()
    multipliers = {'GIB': 1024**3, 'GB': 1000**3, 'MIB': 1024**2, 'MB': 1000**2}
    fixed_bytes = int(value * multipliers[unit])
    # Express as fraction of one GPU's memory so _build_size stays uniform
    frac = fixed_bytes / mem_per_gpu if mem_per_gpu else 1.0
    return mem_per_gpu, frac


# ---------------------------------------------------------------------------
# HplConfig -- BmtConfig subclass
# ---------------------------------------------------------------------------

@dataclass
class HplConfig(BmtConfig):
    """HPL benchmark configuration.

    Outer sweep axes (one mpirun per combination):
        nnodes x zip(ntasks, ngpus)

    Inner sweep axes (encoded in HPL.dat, one output file per outer point):
        blocksize x grid x bcast

    N is computed per outer point from nnodes x ngpus x GPU memory.

    The binary or container image path is owned by the Backend (backend.bin),
    not by HplConfig directly. HplConfig.bin is left empty and unused.

    Example -- GPU run, N from memory fraction
    ------------------------------------------
    >>> config = HplConfig(
    ...     outdir='./output',
    ...     nnodes=[1, 2],
    ...     ntasks=[4, 8],            # zipped with ngpus
    ...     ngpus=[4, 8],
    ...     mem='80%',                # required: one of mem or sizes
    ...     launcher=HplLauncher(
    ...         mpi=OpenMPI(hostlist=['node1', 'node2']),
    ...         backend=Nvidia(ngc='hpc-benchmarks_24.09.sif'),
    ...     ),
    ... )

    Example -- GPU run, explicit N sweep
    ------------------------------------
    >>> config = HplConfig(
    ...     outdir='./output',
    ...     nnodes=1,
    ...     ntasks=4,
    ...     ngpus=4,
    ...     sizes=[130944, 185000],   # required: one of mem or sizes
    ...     launcher=HplLauncher(
    ...         mpi=OpenMPI(hostlist=['node1']),
    ...         backend=Nvidia(ngc='hpc-benchmarks_24.09.sif'),
    ...     ),
    ... )

    Example -- CPU run
    -----------------
    >>> config = HplConfig(
    ...     outdir='./output',
    ...     nnodes=1,
    ...     ntasks=64,
    ...     mem='80%',                # required: one of mem or sizes
    ...     launcher=HplLauncher(
    ...         mpi=OpenMPI(hostlist=['node1']),
    ...         backend=CpuHPL(bin='/opt/hpl/bin/xhpl'),
    ...     ),
    ... )
    """

    name: ClassVar[str] = 'HPL'
    key_headers: list = field(default_factory=lambda: ['nnodes', 'ntasks', 'nthreads', 'ngpus', 'n', 'nb', 'p', 'q', 'bcast'])
    metric_headers: list = field(default_factory=lambda: ['perf (Tflops)'])

    # Both bin and outdir override BmtConfig required fields with defaults.
    # They must appear in the same order as BmtConfig (bin first, outdir second)
    # to satisfy Python's dataclass inheritance rule: no non-default field may
    # follow a default field.
    bin:    str = field(default='')           # unused -- path lives on Backend.bin
    outdir: str = field(default='./output')

    # Exactly one of mem or sizes must be supplied -- they are mutually exclusive.
    # mem  -- fraction or absolute memory per GPU driving N; supports list for sweep
    #         e.g. '80%', '32GiB', or ['60%', '80%']
    # sizes -- explicit N value(s); e.g. 130944 or [65536, 130944]
    mem:   str | list[str] | None = field(default=None)
    sizes: int | list[int] | None = field(default=None)

    # Inner sweep -- all accept scalar or list
    blocksizes: int | list[int] = field(default=None)   # None -> from backend
    grids: list[tuple[int, int]] = field(default=None)  # None -> auto per point
    broadcast: int | list[int] = 1

    # Algorithm parameters (scalar only -- not swept by default)
    pmap:  int = 1
    pfact: int = 1
    nbmin: int = 2
    ndiv:  int = 2
    rfact: int = 1

    def _resolve_defaults(self) -> None:
        super()._resolve_defaults()

        # NvidiaLegacy passes nthreads to --cpu-cores-per-rank so it must be
        # non-zero. Default to 1 (single-threaded) so the flag is always valid;
        # user should set nthreads explicitly for best performance.
        if not self.nthreads and isinstance(self.launcher.backend, NvidiaLegacy):
            self.nthreads = 1

        launcher = self.launcher
        if not isinstance(launcher, HplLauncher):
            raise TypeError("HplConfig requires an HplLauncher")

        backend = launcher.backend
        if backend is None:
            launcher.backend = make_backend(self._gpu)
            backend = launcher.backend

        # Inject discovered hardware into backend
        backend.cpu = self._cpu
        backend.gpu = self._gpu

        self.variant = type(backend).__name__

        # Wire config reference into launcher for HPL.dat writing
        launcher._config = self

        # Blocksize default from backend if not user-supplied
        if self.blocksizes is None:
            self.blocksizes = backend._build_blocksize()


    def _validate_bin(self) -> None:
        pass  # bin unused -- container path lives on Backend.ngc / Backend.bin

    def _validate_config(self) -> None:
        """Delegate config validation to the backend."""
        if self.mem is None and self.sizes is None:
            raise ValueError("HplConfig: exactly one of 'mem' or 'sizes' must be supplied")
        if self.mem is not None and self.sizes is not None:
            raise ValueError("HplConfig: 'mem' and 'sizes' are mutually exclusive -- supply exactly one")
        self.launcher.backend._validate_config()

    def __iter__(self):
        """Full sweep: nnodes x zip(ntasks, ngpus) x blocksizes x sizes x grids x broadcast.

        Each yielded HplSpec is a fully-specified single run point.
        One HPL.dat is written per spec -- single WC result line per output file.
        """
        mem_per_gpu = self._gpu.memory(0) or 0

        for nnodes in _as_list(self.nnodes):
            for ntasks, ngpus in zip(_as_list(self.ntasks), _as_list(self.ngpus)):
                grids = self.grids if self.grids else [_build_grid(nnodes, ngpus)]
                for nb in _as_list(self.blocksizes):
                    if self.sizes is not None:
                        ns = _as_list(self.sizes)
                    else:
                        ns = [
                            _build_size(nnodes, ngpus, nb, mem_per_gpu, _parse_mem(m, mem_per_gpu)[1])
                            for m in _as_list(self.mem)
                        ]
                    for n in ns:
                        for (p, q) in grids:
                            for bcast in _as_list(self.broadcast):
                                spec = HplSpec(
                                    nnodes=nnodes,
                                    ntasks=ntasks,
                                    ngpus=ngpus,
                                    nthreads=self.nthreads,
                                    n=n,
                                    nb=nb,
                                    p=p,
                                    q=q,
                                    bcast=bcast,
                                )
                                self._validate(spec)
                                yield spec

    def _validate(self, spec: HplSpec) -> None:
        """Per-spec validation -- GPU constraint only for now.

        TODO: CPU validation depends on backend type:
          CpuHPL source-build: ntasks == nprocs
          IntelCPU / AmdCPU:   ntasks == nsockets
        """
        if spec.ngpus > self._gpu.count:
            raise ValueError(
                f"Requested {spec.ngpus} GPUs but only {self._gpu.count} available"
            )
        if isinstance(self.launcher.backend, (NvidiaLegacy, Nvidia)):
            if spec.ntasks != spec.ngpus:
                raise ValueError(
                    f"HPL GPU run requires ntasks == ngpus per node "
                    f"(got ntasks={spec.ntasks}, ngpus={spec.ngpus})"
                )

    def parse(self, output_path: str) -> dict[str, Any]:
        """Parse the single WC result line from one HPL output file."""
        pattern = (
            r'WC\S*\s+'
            r'(\d+)\s+'                    # N
            r'(\d+)\s+'                    # NB
            r'(\d+)\s+'                    # P
            r'(\d+)\s+'                    # Q
            r'[\d.]+\s+'                   # time
            r'([\d.]+(?:e[+-]\d+)?)'       # Gflops
        )
        try:
            with open(output_path) as fh:
                for line in fh:
                    m = re.search(pattern, line)
                    if m:
                        return {'perf (Tflops)': float(m.group(5)) / 1000}
        except FileNotFoundError:
            pass
        return {}

    def _output_path(self, spec: HplSpec, n: int, outdir: str) -> str:
        tag = (
            f"HPL-{self.host}"
            f"-n{spec.nnodes}"
            f"-t{spec.ntasks}"
            f"-o{spec.nthreads}"
            f"-g{spec.ngpus}"
            f"-N{spec.n}"
            f"-NB{spec.nb}"
            f"-P{spec.p}"
            f"-Q{spec.q}"
            f"-BCAST{spec.bcast}"
        )
        return os.path.join(outdir, f"{tag}.out.{n}")

    def _yaml_extra(self) -> dict:
        return {
            'mem':       self.mem,
            'blocksize': _as_list(self.blocksizes),
            'bcast':     _as_list(self.broadcast),
            'pmap':      self.pmap,
            'pfact':     self.pfact,
            'nbmin':     self.nbmin,
            'ndiv':      self.ndiv,
            'rfact':     self.rfact,
        }
