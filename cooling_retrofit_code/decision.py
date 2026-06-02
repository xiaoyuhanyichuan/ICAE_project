from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PlanningDecision:
    zone_ids: list[str]
    s_z: dict[str, int]
    cap_ac_new: dict[str, float]
    cap_rdhx: dict[str, float]
    cap_cdu: dict[str, float]
    cap_ashp: float
    cap_wshp: float
    cap_bess: float
    cap_tes: float
    delta_cap_chiller: float
    delta_cap_tower: float
    o_z: dict[str, int]
    n_z: dict[str, int]
    p_z: dict[str, int]
    r_z: dict[str, int]
    a_z: dict[str, int]
    w_z: dict[str, int]


class DecisionSchema:
    zone_capacity_names = ("cap_ac_new", "cap_rdhx", "cap_cdu")
    system_capacity_names = (
        "cap_ashp",
        "cap_wshp",
        "cap_bess",
        "cap_tes",
        "delta_cap_chiller",
        "delta_cap_tower",
    )
    active_config_values = (0, 1, 2, 3, 4, 5, 6)

    def __init__(
        self,
        zone_ids: list[str],
        zone_cap_upper_kw: float | dict[str, float] = 1000.0,
        system_cap_upper_kw: float | dict[str, float] = 5000.0,
    ) -> None:
        if len(set(zone_ids)) != len(zone_ids):
            raise ValueError("Duplicate zone_ids are not allowed.")
        self.zone_ids = list(zone_ids)
        self.zone_cap_upper_kw = _upper_by_name(
            self.zone_capacity_names,
            zone_cap_upper_kw,
            default=1000.0,
        )
        self.system_cap_upper_kw = _upper_by_name(
            self.system_capacity_names,
            system_cap_upper_kw,
            default=5000.0,
        )

    @classmethod
    def from_config(cls, zone_ids: list[str], config: dict) -> "DecisionSchema":
        bounds = config.get("decision_bounds", {})
        default_zone = float(bounds.get("default_zone_cap_upper_kw", 1000.0))
        default_system = float(bounds.get("default_system_cap_upper_kw", 5000.0))
        zone_upper = _bounds_from_config(
            cls.zone_capacity_names,
            bounds.get("zone_cap_upper_kw", default_zone),
            default_zone,
        )
        system_upper = _bounds_from_config(
            cls.system_capacity_names,
            bounds.get("system_cap_upper_kw", default_system),
            default_system,
        )
        return cls(zone_ids, zone_upper, system_upper)

    @property
    def vector_length(self) -> int:
        return (
            len(self.zone_ids)
            + len(self.zone_capacity_names) * len(self.zone_ids)
            + len(self.system_capacity_names)
        )

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lower = np.zeros(self.vector_length, dtype=float)
        upper = np.concatenate(
            [
                np.full(len(self.zone_ids), 6.0),
                np.concatenate(
                    [
                        np.full(len(self.zone_ids), self.zone_cap_upper_kw[name])
                        for name in self.zone_capacity_names
                    ]
                ),
                np.array(
                    [self.system_cap_upper_kw[name] for name in self.system_capacity_names],
                    dtype=float,
                ),
            ]
        )
        return lower, upper

    def encode_default(self) -> np.ndarray:
        return np.zeros(self.vector_length, dtype=float)

    def random_vector(self, rng: np.random.Generator) -> np.ndarray:
        lower, upper = self.bounds()
        vector = rng.uniform(lower, upper)
        vector[: len(self.zone_ids)] = rng.choice(self.active_config_values, size=len(self.zone_ids))
        return vector

    def decode(self, vector: np.ndarray) -> PlanningDecision:
        values = np.asarray(vector, dtype=float)
        if values.shape != (self.vector_length,):
            raise ValueError(f"Expected vector length {self.vector_length}, got {values.size}.")
        if not np.isfinite(values).all():
            raise ValueError("Decision vector must contain only finite values.")

        zone_count = len(self.zone_ids)
        config_values = np.clip(np.rint(values[:zone_count]), 0, 6).astype(int)
        s_z = dict(zip(self.zone_ids, config_values.tolist()))

        cursor = zone_count
        zone_capacities: dict[str, dict[str, float]] = {}
        for name in self.zone_capacity_names:
            caps = np.maximum(values[cursor : cursor + zone_count], 0.0)
            zone_capacities[name] = dict(zip(self.zone_ids, caps.astype(float).tolist()))
            cursor += zone_count

        system_values = {
            name: float(max(values[cursor + idx], 0.0))
            for idx, name in enumerate(self.system_capacity_names)
        }

        return PlanningDecision(
            zone_ids=list(self.zone_ids),
            s_z=s_z,
            cap_ac_new=zone_capacities["cap_ac_new"],
            cap_rdhx=zone_capacities["cap_rdhx"],
            cap_cdu=zone_capacities["cap_cdu"],
            cap_ashp=system_values["cap_ashp"],
            cap_wshp=system_values["cap_wshp"],
            cap_bess=system_values["cap_bess"],
            cap_tes=system_values["cap_tes"],
            delta_cap_chiller=system_values["delta_cap_chiller"],
            delta_cap_tower=system_values["delta_cap_tower"],
            o_z={zone: int(k in {0, 4}) for zone, k in s_z.items()},
            n_z={zone: int(k in {1, 2, 3, 6}) for zone, k in s_z.items()},
            p_z={zone: int(k in {4, 6}) for zone, k in s_z.items()},
            r_z={zone: int(k == 5) for zone, k in s_z.items()},
            a_z={zone: int(k in {2, 4, 6}) for zone, k in s_z.items()},
            w_z={zone: int(k in {4, 5, 6}) for zone, k in s_z.items()},
        )


def _upper_by_name(
    names: tuple[str, ...],
    value: float | dict[str, float],
    default: float,
) -> dict[str, float]:
    if isinstance(value, dict):
        return {name: _nonnegative_float(value.get(name, default), name) for name in names}
    number = _nonnegative_float(value, "capacity upper bound")
    return {name: number for name in names}


def _bounds_from_config(
    names: tuple[str, ...],
    value: float | dict[str, float],
    default: float,
) -> dict[str, float]:
    if isinstance(value, dict):
        return {name: _nonnegative_float(value.get(name, default), name) for name in names}
    return _upper_by_name(names, value, default)


def _nonnegative_float(value: float, label: str) -> float:
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{label} must be finite.")
    if number < 0.0:
        raise ValueError(f"{label} must be non-negative.")
    return number
