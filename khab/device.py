#!/usr/bin/env python3

from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class Device(ABC):

    @abstractmethod
    def info(self) -> None:
        pass
