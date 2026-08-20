#!/usr/bin/env python3

import itertools
import os

from dataclasses import dataclass, field
from typing import Any, ClassVar

from affinity import Affinity, GNU
from bmt import BmtConfig, Launcher, RunSpec, _as_list


@dataclass(frozen=True)
class StreamSpec(RunSpec):
    omp: str = ''
    place: str = ''
    bind: str = ''


@dataclass
class StreamLauncher(Launcher):
    def build_cmd(self, spec: StreamSpec) -> list[list[str]]:
        os.environ.update(spec.env)
        env = [f'{k}={v}' for k, v in spec.env.items()]
        return [[*env, self.bin]]


@dataclass
class StreamConfig(BmtConfig):
    name: ClassVar[str] = "STREAM"
    key_headers: list = field(default_factory=lambda: ["omp", "place", "bind", "nthreads"])
    metric_headers: list = field(default_factory=lambda: ["copy (GB/s)", "scale (GB/s)", "add (GB/s)", "triad (GB/s)"])

    omp: Affinity = field(default_factory=GNU)
    bind: str | list[str] = field(default_factory=list)
    place: str | list[str] = field(default_factory=list)
    launcher: Launcher = field(default_factory=StreamLauncher)

    def _resolve_defaults(self) -> None:
        super()._resolve_defaults()
        self.ngpus = 0
        if not self.nthreads:
            self.nthreads = self._cpu.threads
        self.variant = type(self.omp).__name__

        if not self.bind:
            self.bind = [self.omp.bind]
        if not self.place:
            self.place = [self.omp.place]

    def _yaml_extra(self) -> dict:
        return {
            'bind': _as_list(self.bind),
            'place': _as_list(self.place),
        }

    def _validate(self, spec: RunSpec) -> None:
        pass

    def __iter__(self):
        for nnodes, ntasks, ngpus, place, bind, nthreads in itertools.product(
            _as_list(self.nnodes),
            _as_list(self.ntasks),
            _as_list(self.ngpus),
            _as_list(self.place),
            _as_list(self.bind),
            _as_list(self.nthreads),
        ):
            omp = type(self.omp)(place=place, bind=bind)
            spec = StreamSpec(
                nnodes=nnodes,
                ntasks=ntasks,
                ngpus=ngpus,
                nthreads=nthreads,
                omp=omp.name,
                place=omp.place,
                bind=omp.bind,
                extra_env=omp.env(),
            )
            yield spec

    def _output_path(self, spec: StreamSpec, n: int, outdir: str) -> str:
        partition = os.environ.get("SLURM_JOB_PARTITION", self.host)
        tag = f"{self.name}-{partition}-place_{spec.place}-bind_{spec.bind}-omp_{spec.nthreads}"
        return os.path.join(outdir, f"{tag}.out.{n}")

    def parse(self, output_path: str) -> dict[str, Any]:
        kernels = ["Copy", "Scale", "Add", "Triad"]
        bw: dict[str, float] = {}

        try:
            with open(output_path) as fh:
                for line in fh:
                    parts = line.split()
                    for k in kernels:
                        if parts and parts[0] == f"{k}:":
                            try:
                                bw[f"{k.lower()} (GB/s)"] = float(parts[1]) / 1000
                            except (IndexError, ValueError):
                                pass
        except FileNotFoundError:
            pass

        return bw
