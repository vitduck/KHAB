#!/usr/bin/env python3

import itertools
import logging
import os

from dataclasses import dataclass, field
from typing import Any, ClassVar

from bmt import BmtConfig, Launcher, RunSpec, _as_list

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class BabelStreamSpec(RunSpec):
    model: str = ''
    device: int = 0
    size: int = 2**25
    ntimes: int = 100


@dataclass
class BabelStreamLauncher(Launcher):
    def build_cmd(self, spec: BabelStreamSpec) -> list[list[str]]:
        return [[
            self.bin,
            ['--device', str(spec.device)],
            ['--arraysize', str(spec.size)],
            ['--numtimes', str(spec.ntimes)],
        ]]


@dataclass
class BabelStreamConfig(BmtConfig):
    name: ClassVar[str] = 'BABELSTREAM'
    key_headers: list = field(default_factory=lambda: ['model', 'device', 'size', 'ntimes'])
    metric_headers: list = field(default_factory=lambda: [
        'copy (GB/s)', 'mul (GB/s)', 'add (GB/s)', 'triad (GB/s)', 'dot (GB/s)',
    ])

    model: str = ''
    device: int | list[int] = 0
    size: int | list[int] = 2**25
    ntimes: int | list[int] = 100

    launcher: Launcher = field(default_factory=BabelStreamLauncher)

    def _resolve_defaults(self) -> None:
        super()._resolve_defaults()
        self.variant = self.model.upper()

    def _validate_config(self) -> None:
        if not self.model:
            raise ValueError(
                "BabelStreamConfig: 'model' is required (e.g. model='cuda', model='omp')"
            )

    def _validate(self, spec: BabelStreamSpec) -> None:
        if spec.device >= self._gpu.count:
            raise ValueError(
                f"Requested device {spec.device} but only {self._gpu.count} GPUs available"
            )

    def __iter__(self):
        for nnodes, ntasks, ngpus, nthreads, device, size, ntimes in itertools.product(
            _as_list(self.nnodes),
            _as_list(self.ntasks),
            _as_list(self.ngpus),
            _as_list(self.nthreads),
            _as_list(self.device),
            _as_list(self.size),
            _as_list(self.ntimes),
        ):
            spec = BabelStreamSpec(
                nnodes=nnodes,
                ntasks=ntasks,
                ngpus=ngpus,
                nthreads=nthreads,
                model=self.model,
                device=device,
                size=size,
                ntimes=ntimes,
            )
            self._validate(spec)
            yield spec

    def _output_path(self, spec: BabelStreamSpec, n: int, outdir: str) -> str:
        partition = os.environ.get('SLURM_JOB_PARTITION', self.host)
        model = self.model.upper() if self.model else self._gpu.name.upper()
        tag = f"{self.name}-{partition}-{model}-device_{spec.device}-size_{spec.size}"
        return os.path.join(outdir, f"{tag}.out.{n}")

    def parse(self, output_path: str) -> dict[str, Any]:
        # output format: "Function    MBytes/sec  Min (sec) ..."
        # data rows:     "Copy        792858.880  0.00068 ..."
        kernels = {'Copy', 'Mul', 'Add', 'Triad', 'Dot'}
        bw: dict[str, float] = {}
        try:
            with open(output_path) as fh:
                for line in fh:
                    parts = line.split()
                    if len(parts) >= 2 and parts[0] in kernels:
                        try:
                            bw[f"{parts[0].lower()} (GB/s)"] = float(parts[1]) / 1000
                        except ValueError:
                            pass
        except FileNotFoundError:
            log.warning("Output file not found: %s", output_path)
        return bw

    def _yaml_extra(self) -> dict:
        return {
            'model': self.model,
            'device': _as_list(self.device),
            'size': _as_list(self.size),
            'ntimes': _as_list(self.ntimes),
        }
