#!/usr/bin/env python3

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar


@dataclass
class Affinity(ABC):
    name: ClassVar[str] = ''

    mapped_place: dict = field(default_factory=dict)
    mapped_bind: dict = field(default_factory=dict)

    place: str = ''
    bind: str = ''
    verbose: bool = False

    def __setattr__(self, name, value):
        if name == 'place' and self.__dict__.get('mapped_place'):
            value = self.mapped_place.get(value, value)
        if name == 'bind' and self.__dict__.get('mapped_bind'):
            value = self.mapped_bind.get(value, value)
        super().__setattr__(name, value)

    @abstractmethod
    def env(self) -> dict[str, str]:
        pass

@dataclass
class GNU(Affinity):
    name: ClassVar[str] = 'GNU'

    mapped_place: dict = field(default_factory=lambda: {
        'core': 'cores', 'cores': 'cores',
        'thread': 'threads', 'threads': 'threads',
    })
    mapped_bind: dict = field(default_factory=lambda: {
        'close': 'close', 'spread': 'spread',
        'none': 'false', 'false': 'false',
    })

    place: str = 'cores'
    bind: str = 'true'

    def env(self) -> dict[str, str]:
        return {
            'OMP_PLACES': self.place,
            'OMP_PROC_BIND': self.bind,
        }

@dataclass
class Intel(Affinity):
    name: ClassVar[str] = 'Intel'

    mapped_place: dict = field(default_factory=lambda: {
        'core': 'core', 'cores': 'core',
        'thread': 'thread', 'threads': 'thread',
    })
    mapped_bind: dict = field(default_factory=lambda: {
        'close': 'compact', 'spread': 'scatter',
        'compact': 'compact', 'scatter': 'scatter',
        'none': 'none', 'false': 'none',
    })

    place: str = 'core'
    bind: str = 'none'

    def env(self) -> dict[str, str]:
        return {
            'KMP_AFFINITY': f'granularity={self.place},{self.bind}',
        }
