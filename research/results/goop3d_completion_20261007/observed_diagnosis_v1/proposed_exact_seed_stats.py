"""Proposal only: independent three-seed SD with exact sample-variance arithmetic.

No source/model/output data are read. The frozen checker remains unchanged.
"""
from decimal import Decimal, localcontext
from fractions import Fraction
import math


def valid(value):
    return type(value) in (int, float) and math.isfinite(value)


def seed_values_exact(items):
    if len(items) != 3:
        raise ValueError('three required seed values')
    complete = all(valid(x) for x in items)
    mean = math.fsum(items) / 3 if complete else None
    sd = None
    if complete:
        exact = [Fraction(x) for x in items]
        # For n=3, unbiased sample variance = sum_{i<j}(xi-xj)^2 / 6.
        variance = sum((exact[i] - exact[j]) ** 2 for i in range(3) for j in range(i + 1, 3)) / 6
        # Avoid float variance overflow/underflow before the square root. Only
        # the final result is rounded to float; this is independent of stdev.
        with localcontext() as context:
            context.prec = 60
            sd = float((Decimal(variance.numerator) / Decimal(variance.denominator)).sqrt())
    return {'seed_values': {str(s): items[s] for s in range(3)}, 'required_seed_pairs': 3,
            'defined_seed_pairs': sum(valid(x) for x in items), 'mean': mean, 'sample_sd': sd}
