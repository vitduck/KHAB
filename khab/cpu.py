#!/usr/bin/env python3

import os
import logging
import subprocess
import cpuinfo

from dataclasses import dataclass
from device import Device

log = logging.getLogger(__name__)

@dataclass
class CPU(Device):
    name: str = 'Unknown'
    vendor: str = 'Unknown'
    cores: int = 0
    threads: int = 0
    sockets: int = 0
    numa: int = 0

    def __post_init__(self):
        try:
            info = cpuinfo.get_cpu_info()

            self.name = info.get('brand_raw', self.name)
            self.vendor = info.get('vendor_id_raw', self.vendor)
            self.cores = info.get('count', self.cores)
            self.threads = os.cpu_count() or 0
        except Exception as e:
            log.warning("CPU info detection failed: %s", e)

        try:
            topo = self._topology()
            self.sockets = topo['sockets']
            self.numa = topo['numa']
        except Exception as e:
            log.warning("CPU topology detection failed: %s", e)

    def _topology(self) -> dict:
        numa = 0
        sockets = 0
        
        out = subprocess.run(['lscpu'], capture_output=True, text=True).stdout

        for line in out.splitlines():
            if 'Socket(s):' in line:
                sockets = int(line.split(':')[1].strip())
            elif 'NUMA node(s):' in line:
                numa = int(line.split(':')[1].strip())

        return {'sockets': sockets, 'numa': numa}

    def info(self) -> None:
        log.info(
            "CPU: %s / Vendor: %s / Cores: %d / Threads: %d / Sockets: %d / NUMA: %d",
            self.name, self.vendor, self.cores, self.threads, self.sockets, self.numa
        )
