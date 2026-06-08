from __future__ import annotations

import copy
import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import numpy as np

from decision import PlanningDecision


@dataclass(frozen=True)
class WarmStartHit:
    payload: dict[str, Any]
    match_type: str
    key: tuple[Any, ...]


class WarmStartManager:
    def __init__(
        self,
        enabled: bool = True,
        capacity_bucket_kw: float = 100.0,
        neighbor_bucket_radius: int = 1,
        max_entries: int = 512,
        td_fingerprint: str = "",
        config_fp: str = "",
    ) -> None:
        self.enabled = bool(enabled)
        self.capacity_bucket_kw = max(float(capacity_bucket_kw), 1.0e-9)
        self.neighbor_bucket_radius = max(0, int(neighbor_bucket_radius))
        self.max_entries = max(1, int(max_entries))
        self.td_fingerprint = str(td_fingerprint)
        self.config_fp = str(config_fp)
        self._entries: OrderedDict[tuple[Any, ...], dict[str, Any]] = OrderedDict()
        self.exact_hits = 0
        self.neighbor_hits = 0
        self.misses = 0

    @classmethod
    def from_config(cls, config: dict[str, Any], td_fingerprint: str = "") -> "WarmStartManager":
        inner = config.get("solver", {}).get("inner", {})
        warm = inner.get("warm_start", {}) if isinstance(inner, dict) else {}
        if not isinstance(warm, dict):
            warm = {}
        return cls(
            enabled=bool(warm.get("enabled", True)),
            capacity_bucket_kw=float(warm.get("capacity_bucket_kw", 100.0)),
            neighbor_bucket_radius=int(warm.get("neighbor_bucket_radius", 1)),
            max_entries=int(warm.get("max_entries", 512)),
            td_fingerprint=td_fingerprint,
            config_fp=_config_fingerprint(config),
        )

    def remember(self, decision: PlanningDecision, payload: dict[str, Any] | None) -> None:
        if not self.enabled or not payload:
            return
        key = self.key_for(decision)
        if key in self._entries:
            self._entries.move_to_end(key)
        self._entries[key] = _copy_payload(payload)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)

    def lookup(self, decision: PlanningDecision) -> WarmStartHit | None:
        if not self.enabled:
            return None
        key = self.key_for(decision)
        exact = self._entries.get(key)
        if exact is not None:
            self._entries.move_to_end(key)
            self.exact_hits += 1
            return WarmStartHit(payload=exact, match_type="exact", key=key)

        topology = key[:3]
        buckets = key[3]
        for candidate_key in reversed(self._entries.keys()):
            if candidate_key[:3] != topology:
                continue
            candidate_buckets = candidate_key[3]
            if len(candidate_buckets) != len(buckets):
                continue
            if all(
                abs(int(candidate) - int(current)) <= self.neighbor_bucket_radius
                for candidate, current in zip(candidate_buckets, buckets)
            ):
                self._entries.move_to_end(candidate_key)
                self.neighbor_hits += 1
                return WarmStartHit(
                    payload=self._entries[candidate_key],
                    match_type="neighbor",
                    key=candidate_key,
                )
        self.misses += 1
        return None

    def key_for(self, decision: PlanningDecision) -> tuple[Any, ...]:
        zones = tuple(decision.zone_ids)
        topology = tuple(
            (
                zone,
                int(decision.s_z[zone]),
                int(decision.o_z.get(zone, 0)),
                int(decision.n_z.get(zone, 0)),
                int(decision.p_z.get(zone, 0)),
                int(decision.r_z.get(zone, 0)),
                int(decision.a_z.get(zone, 0)),
                int(decision.w_z.get(zone, 0)),
            )
            for zone in zones
        )
        capacity_values: list[float] = []
        for attr_name in ("cap_ac_new", "cap_rdhx", "cap_cdu"):
            values = getattr(decision, attr_name)
            capacity_values.extend(float(values.get(zone, 0.0)) for zone in zones)
        for attr_name in (
            "cap_ashp",
            "cap_wshp",
            "cap_bess",
            "cap_tes",
            "delta_cap_chiller",
            "delta_cap_tower",
        ):
            capacity_values.append(float(getattr(decision, attr_name)))
        buckets = tuple(int(round(value / self.capacity_bucket_kw)) for value in capacity_values)
        return (self.config_fp, self.td_fingerprint, topology, buckets)

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    @property
    def stats(self) -> dict[str, int]:
        return {
            "exact_hits": self.exact_hits,
            "neighbor_hits": self.neighbor_hits,
            "misses": self.misses,
            "entries": self.entry_count,
        }


def _config_fingerprint(config: dict[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, default=str, ensure_ascii=True).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:16]


def _copy_payload(payload: dict[str, Any]) -> dict[str, Any]:
    def copy_value(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.copy()
        if isinstance(value, dict):
            return {key: copy_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [copy_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(copy_value(item) for item in value)
        return copy.deepcopy(value)

    return {key: copy_value(value) for key, value in payload.items()}
