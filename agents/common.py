import typing
import numpy as np

import torch


def returns_from(rewards: typing.List[float], discount_factor: float) -> typing.List[float]:
    R = 0.0
    returns = []
    for r in rewards[::-1]:
        R = R * discount_factor + r
        returns.append(R)
    returns.reverse()
    return normalize(returns)


def normalize(values: typing.List[float]) -> typing.List[float]:
    values = torch.tensor(values)
    values = (values - values.mean()) / (values.std() + 1e-8)
    return values.tolist()