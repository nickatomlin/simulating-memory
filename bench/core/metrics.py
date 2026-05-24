from __future__ import annotations
from typing import Dict, List

def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0

def macro_avg(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    keys = sorted({k for m in metrics_list for k in m.keys()})
    out={}
    for k in keys:
        vals=[m[k] for m in metrics_list if k in m]
        if vals:
            out[k]=mean(vals)
    return out
