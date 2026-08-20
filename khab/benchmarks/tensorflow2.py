#!/usr/bin/env python3

import itertools
import logging
import os
import re
import subprocess

from dataclasses import dataclass, field
from typing import Any, ClassVar

from bmt import BmtConfig, RunSpec, _as_list
from launcher import Launcher
from mpi import MPI, OpenMPI

log = logging.getLogger(__name__)

# Verified-good container and commit — only these are supported
_TARGET_NGC = 'nvcr.io/nvidia/tensorflow:23.07-tf2-py3'
_TARGET_COMMIT = 'c8e97df0d4d3d0c1020b98391c526df12371fc30'

# Default batch size per GPU architecture
_BATCH_DEFAULTS: dict[str, int] = {
    'v100': 256,
    'a100': 512,
    'h100': 768,
    'h200': 768,
}


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TensorFlow2Spec(RunSpec):
    model: str = ''
    batch_size: int = 0


# ---------------------------------------------------------------------------
# Launcher
# ---------------------------------------------------------------------------

@dataclass
class TensorFlow2Launcher(Launcher):
    """Composes MPI + singularity + tf_cnn_benchmarks.

    build_cmd returns:
        [mpi_block, singularity_block, tf2_block]
    """
    mpi: MPI = field(default_factory=OpenMPI)
    ngc: str    = ''   # singularity image path
    tf_cnn_benchmark: str    = ''   # path to tf_cnn_benchmarks repo

    # Optional tf_cnn_benchmarks flags — scalars, not sweep axes
    imagenet_dir: str = ''
    optimizer: str = 'sgd'
    data_format: str = 'NCHW'
    num_warmup_batches: int = 10
    num_batches: int = 100
    variable_update: str = 'horovod'

    # Injected by TensorFlow2Config._resolve_defaults()
    _cpu: Any = field(init=False, default=None, repr=False)
    _tf_log_level: int = field(init=False, default=3, repr=False)

    def build_cmd(self, spec: TensorFlow2Spec) -> list[list[str]]:
        if self._cpu:
            self.mpi.map = self._build_numa(spec)

        mpi_block = self.mpi.build_cmd(spec)

        singularity_block = [
            'singularity', 'run',
            ['--env', f'TF_CPP_MIN_LOG_LEVEL={self._tf_log_level}'],
            '--nv', self.ngc,
            'python3',
        ]

        tf2_block = [
            f'{self.tf_cnn_benchmark}/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py',
            ['--model', spec.model],
            ['--data_format', self.data_format],
            ['--optimizer', self.optimizer],
            ['--num_warmup_batches', str(self.num_warmup_batches)],
            ['--num_batches', str(self.num_batches)],
            ['--batch_size', str(spec.batch_size)],
            ['--variable_update', self.variable_update],
            ['--datasets_num_private_threads', str(spec.nthreads)],
            '--compute_lr_on_cpu=True',
            '--allow_growth=True',
        ]

        if self.imagenet_dir:
            tf2_block += [
                '--data_name=imagenet',
                ['--data_dir', self.imagenet_dir],
            ]

        return [mpi_block, singularity_block, tf2_block]

    def _build_numa(self, spec: TensorFlow2Spec) -> str:
        """Map MPI ranks by node or numa depending on private_threads load."""
        if self._cpu.numa and spec.nthreads > self._cpu.cores / self._cpu.numa:
            return 'node'
        return 'numa'

    def _validate(self) -> None:
        """Validate container version and benchmark commit."""
        self._verify_commit()
        self._verify_ngc_version()

    def _verify_commit(self) -> None:
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                cwd=self.tf_cnn_benchmark,
                capture_output=True,
                text=True,
                check=True,
            )
            current = result.stdout.strip()
            if current != _TARGET_COMMIT:
                raise RuntimeError(
                    f"Incorrect tf_cnn_benchmark commit.\n"
                    f"Expected: {_TARGET_COMMIT}\n"
                    f"Current:  {current}\n"
                    f"Fix: cd {self.tf_cnn_benchmark} && git checkout {_TARGET_COMMIT}"
                )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Failed to check git commit in {self.tf_cnn_benchmark}: {e.stderr}"
            )

    def _verify_ngc_version(self) -> None:
        try:
            result = subprocess.run(
                ['singularity', 'inspect', self.ngc],
                capture_output=True,
                text=True,
                check=True,
            )
            m = re.search(
                r'org\.label-schema\.usage\.singularity\.deffile\.from:\s*(.+)',
                result.stdout,
            )
            current = m.group(1).strip() if m else ''
            if current != _TARGET_NGC:
                raise RuntimeError(
                    f"Incorrect TensorFlow NGC container.\n"
                    f"Expected: {_TARGET_NGC}\n"
                    f"Actual:   {current}\n"
                    f"Fix: singularity pull docker://{_TARGET_NGC}"
                )
        except FileNotFoundError:
            raise RuntimeError(
                "singularity not found — ensure Singularity is installed and on PATH"
            )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TensorFlow2Config(BmtConfig):
    name: ClassVar[str] = 'TENSORFLOW2'
    key_headers: list = field(default_factory=lambda: ['nnodes', 'ntasks', 'nthreads', 'ngpus', 'model', 'batch_size'])
    metric_headers: list = field(default_factory=lambda: ['throughput (images/sec)'])

    # No bare binary — container-based
    bin:    str = ''
    outdir: str = './output'

    launcher: TensorFlow2Launcher = field(default_factory=TensorFlow2Launcher)

    # Parameter spaces — plural, sweepable
    models: str | list[str] = 'resnet50'
    batch_sizes: int | list[int] = 0   # 0 → auto per GPU type
    tf_log_level: int = 3              # TF_CPP_MIN_LOG_LEVEL: 0=all 1=no INFO 2=no WARNING 3=no ERROR

    def _validate_bin(self) -> None:
        pass   # container-based — no bare binary

    def _validate_config(self) -> None:
        if self.ntasks != self.ngpus:
            raise ValueError(
                f"ntasks ({self.ntasks}) must equal ngpus ({self.ngpus}) for TensorFlow2"
            )
        self.launcher._validate()

    def _resolve_defaults(self) -> None:
        super()._resolve_defaults()

        if not self.nthreads:
            self.nthreads = 4

        # Inject cpu and log level into launcher
        self.launcher._cpu = self._cpu
        self.launcher._tf_log_level = self.tf_log_level
        self.variant = os.path.basename(self.launcher.ngc).replace('.sif', '')

        # Auto batch_size per GPU type if not set
        if not self.batch_sizes:
            self.batch_sizes = self._auto_batch_size()

    def _auto_batch_size(self) -> int:
        m = re.search(r'(v100|a100|h100|h200)', self._gpu.name, re.IGNORECASE)
        if m:
            return _BATCH_DEFAULTS.get(m.group(1).lower(), 64)
        return 64

    def __iter__(self):
        for nnodes, ntasks, ngpus, nthreads, model, batch_size in itertools.product(
            _as_list(self.nnodes),
            _as_list(self.ntasks),
            _as_list(self.ngpus),
            _as_list(self.nthreads),
            _as_list(self.models),
            _as_list(self.batch_sizes),
        ):
            spec = TensorFlow2Spec(
                nnodes=nnodes,
                ntasks=ntasks,
                ngpus=ngpus,
                nthreads=nthreads,
                model=model,
                batch_size=batch_size,

            )
            self._validate(spec)
            yield spec

    def _validate(self, spec: TensorFlow2Spec) -> None:
        super()._validate(spec)

    def _output_path(self, spec: TensorFlow2Spec, n: int, outdir: str) -> str:
        tag = (
            f"TF2-{self.host}"
            f"-n{spec.nnodes}"
            f"-t{spec.ntasks}"
            f"-o{spec.nthreads}"
            f"-g{spec.ngpus}"
            f"-{spec.model}"
            f"-bs{spec.batch_size}"
        )
        return os.path.join(outdir, f"{tag}.out.{n}")

    def parse(self, output_path: str) -> dict[str, Any]:
        throughputs: list[float] = []

        try:
            with open(output_path) as fh:
                for line in fh:
                    m = re.search(r'total images/sec:\s+([\d.]+)', line)
                    if m:
                        throughputs.append(float(m.group(1)))
        except FileNotFoundError:
            log.warning("Output file not found: %s", output_path)

        if throughputs:
            return {'throughput (images/sec)': sum(throughputs) / len(throughputs)}
        return {}
