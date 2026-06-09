from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LaneDepartureState:
    active: bool = False

    def threshold(self, *, enter: float, exit: float) -> float:
        return exit if self.active else enter

    def update(self, triggered: bool) -> None:
        self.active = triggered
