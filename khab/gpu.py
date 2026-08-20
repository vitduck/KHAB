#!/usr/bin/env python3

import glob
import logging
import pynvml

from dataclasses import dataclass, field
from typing import Optional
from device import Device

log = logging.getLogger(__name__)


@dataclass
class GPU(Device):
    vendor: str = 'Unknown'
    count: int = 0
    devices: list[dict] = field(default_factory=list)

    @property
    def name(self) -> str:
        raw = self.devices[0]['name'] if self.devices else ''
        return raw.replace(' ', '_')

    def __post_init__(self):
        try:
            pynvml.nvmlInit()
            self.count = pynvml.nvmlDeviceGetCount()

            if self.count > 0:
                self.vendor = 'nvidia'
                numa = self._topology()

                for i in range(self.count):
                    h = pynvml.nvmlDeviceGetHandleByIndex(i)
                    raw_name = pynvml.nvmlDeviceGetName(h)
                    raw_uuid = pynvml.nvmlDeviceGetUUID(h)

                    self.devices.append({
                        'index': i,
                        'name': raw_name.decode() if isinstance(raw_name, bytes) else raw_name,
                        'uuid': raw_uuid.decode() if isinstance(raw_uuid, bytes) else raw_uuid,
                        'memory': pynvml.nvmlDeviceGetMemoryInfo(h).total,
                        'affinity': numa[i] if i < len(numa) else None,
                    })
        except Exception as e:
            log.warning("GPU detection failed: %s", e)
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    def _topology(self) -> list[int]:
        """Return NUMA node per GPU via sysfs PCI bus ID."""
        numa = []
        for i in range(self.count):
            try:
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                bus_id = pynvml.nvmlDeviceGetPciInfo(h).busId
                bus_id = bus_id.decode() if isinstance(bus_id, bytes) else bus_id
                bus_id_short = bus_id.split(':', 1)[-1].lower()

                matches = glob.glob(f"/sys/bus/pci/devices/*{bus_id_short}/numa_node")
                if matches:
                    with open(matches[0]) as f:
                        numa.append(int(f.read().strip()))
                else:
                    log.warning("No sysfs numa_node found for GPU %d (bus %s)", i, bus_id_short)
                    numa.append(None)
            except Exception as e:
                log.warning("NUMA detection failed for GPU %d: %s", i, e)
                numa.append(None)
        return numa

    def memory(self, gpu_id: int = 0) -> Optional[int]:
        return self.devices[gpu_id]['memory'] if 0 <= gpu_id < self.count else None

    def affinity(self, gpu_ids: Optional[list[int]] = None) -> list:
        targets = range(self.count) if gpu_ids is None else gpu_ids
        return [
            self.devices[i]['affinity'] if 0 <= i < self.count else None
            for i in targets
        ]

    def info(self) -> None:
        for d in self.devices:
            log.info(
                "GPU %d: %s / UUID: %s / VRAM: %d GB / Affinity: %s",
                d['index'], d['name'], d['uuid'],
                int(d['memory'] / 1024**3), d['affinity'],
            )
