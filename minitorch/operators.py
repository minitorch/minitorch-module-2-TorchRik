"""
Collection of the core mathematical operators used throughout the code base.
"""

import math
import typing as tp

# ## Task 0.1
#
# Implementation of a prelude of elementary functions.
EPS = 1e-6


def mul(a: float, b: float) -> float:
    return float(a * b)


def id(a: float) -> float:
    return a


def add(a: float, b: float) -> float:
    return a + b


def neg(a: float) -> float:
    return float(-a)


def lt(a: float, b: float) -> float:
    return float(a < b)


def eq(a: float, b: float) -> float:
    return float(a == b)


def is_close(a: float, b: float) -> float:
    return float(abs(a - b) <= EPS)


def sigmoid(a: float) -> float:
    return 1 / (1 + math.exp(-a))


def relu(a: float) -> float:
    return a if a > 0 else 0.0


def max(x: float, y: float) -> float:
    return x if x > y else y


def log(x: float) -> float:
    "$f(x) = log(x)$"
    return float(math.log(x + EPS))


def exp(x: float) -> float:
    "$f(x) = e^{x}$"
    return float(math.exp(x))


def log_back(a: float, b: float) -> float:
    return b / a


def inv(a: float) -> float:
    return 1 / a


def inv_back(a: float, b: float) -> float:
    return -b / (a * a)


def relu_back(a: float, b: float) -> float:
    return b if a > 0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.


def map(
    fn: tp.Callable[[float], float]
) -> tp.Callable[[tp.Iterable[float]], tp.Iterable[float]]:
    """
    Higher-order map.

    See https://en.wikipedia.org/wiki/Map_(higher-order_function)

    Args:
        fn: Function from one value to one value.

    Returns:
        A function that takes a list, applies `fn` to each element, and returns a
         new list
    """
    return lambda values: (fn(row) for row in values)


def zipWith(
    fn: tp.Callable[[float, float], float]
) -> tp.Callable[[tp.Iterable[float], tp.Iterable[float]], tp.Iterable[float]]:
    return lambda left_value, right_values: (
        fn(left_value, right_values)
        for left_value, right_values in zip(left_value, right_values)
    )


def reduce(
    func: tp.Callable[[float, float], float]
) -> tp.Callable[[tp.Iterable[float]], float]:
    def foo(values: tp.Iterable[float]) -> float:
        prev_val = None
        for val in values:
            if prev_val is None:
                prev_val = val
            else:
                prev_val = func(prev_val, val)
        if prev_val is None:
            return 0
        return prev_val

    return foo


def negList(values: tp.Iterable[float]) -> tp.Iterable[float]:
    return map(neg)(values)


def addLists(
    left_values: tp.Iterable[float], right_values: tp.Iterable[float]
) -> tp.Iterable[float]:
    return zipWith(add)(left_values, right_values)


def sum(values: tp.Iterable[float]) -> float:
    return reduce(add)(values)


def prod(values: tp.Iterable[float]) -> float:
    return reduce(mul)(values)
