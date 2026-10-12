#!/usr/bin/env python3
"""Independent 80-digit Decimal references; Python standard library only.

No native extension/nalgebra/NumPy is used. Inputs are interpreted from their
decimal spelling. Test cases use exactly representable inputs except the
explicit tiny/large scale checks, which use relative tolerances.
"""
import json
import platform
from decimal import Decimal, getcontext
from pathlib import Path

getcontext().prec = 80
D = Decimal


def matrix(x):
    return [[D(str(v)) for v in row] for row in x]


def transpose(x):
    return list(map(list, zip(*x)))


def multiply(a, b):
    return [[sum(x * y for x, y in zip(row, col)) for col in transpose(b)] for row in a]


def moments(x, weights=None, normalization="population"):
    x = matrix(x)
    w = [D(1)] * len(x) if weights is None else [D(str(v)) for v in weights]
    total = sum(w)
    w = [v / total for v in w]
    mean = [sum(v * row[j] for v, row in zip(w, x)) for j in range(len(x[0]))]
    denom = D(1) if normalization == "population" else D(1) - sum(v * v for v in w)
    cov = [[sum(v * (row[j] - mean[j]) * (row[k] - mean[k]) for v, row in zip(w, x)) / denom
            for k in range(len(mean))] for j in range(len(mean))]
    return {"mean": mean, "covariance": cov}


def cholesky(x):
    p = len(x)
    lower = [[D(0)] * p for _ in x]
    for i in range(p):
        for j in range(i + 1):
            remainder = x[i][j] - sum(lower[i][k] * lower[j][k] for k in range(j))
            lower[i][j] = remainder.sqrt() if i == j else remainder / lower[j][j]
    return lower


def covariance_map(source, target):
    ls, lt = cholesky(matrix(source)), cholesky(matrix(target))
    # Solve each row of A*Ls = Lt by backward substitution.
    p = len(ls)
    a = [[D(0)] * p for _ in ls]
    for i in range(p):
        for j in reversed(range(p)):
            a[i][j] = (lt[i][j] - sum(a[i][k] * ls[k][j] for k in range(j + 1, p))) / ls[j][j]
    return {"matrix": a, "log_abs_det": sum(lt[j][j].ln() - ls[j][j].ln() for j in range(p))}


def inverse_and_logdet(x):
    # Independent pivoted Gauss-Jordan, used only to construct round-trip fixtures.
    a = matrix(x)
    p = len(a)
    inv = [[D(i == j) for j in range(p)] for i in range(p)]
    logdet = D(0)
    for j in range(p):
        pivot = max(range(j, p), key=lambda i: abs(a[i][j]))
        a[j], a[pivot] = a[pivot], a[j]
        inv[j], inv[pivot] = inv[pivot], inv[j]
        value = a[j][j]
        logdet += abs(value).ln()
        a[j] = [v / value for v in a[j]]
        inv[j] = [v / value for v in inv[j]]
        for i in range(p):
            if i != j:
                factor = a[i][j]
                a[i] = [v - factor * u for v, u in zip(a[i], a[j])]
                inv[i] = [v - factor * u for v, u in zip(inv[i], inv[j])]
    return inv, logdet


def affine(x, a, shift):
    inv, logdet = inverse_and_logdet(a)
    transformed = multiply(matrix(x), transpose(matrix(a)))
    transformed = [[v + D(str(b)) for v, b in zip(row, shift)] for row in transformed]
    back_shift = [-row[0] for row in multiply(inv, [[D(str(b))] for b in shift])]
    return {"draws": transformed, "log_abs_det": logdet}, inv, back_shift


def floats(value):
    if isinstance(value, D):
        return float(value)
    if isinstance(value, dict):
        return {k: floats(v) for k, v in value.items()}
    if isinstance(value, list):
        return [floats(v) for v in value]
    return value


def main():
    cases = []
    correlated = [[-3, -5, 2], [0, 1, -1], [2, 3, 4], [5, 8, 0], [1, -2, 3]]
    for name, x, w in [
        ("uniform", correlated, None),
        ("explicit_uniform", correlated, [7] * 5),
        ("nonuniform_zero", correlated, [1, 0, 2, 4, 3]),
        ("singular", [[1, 2], [2, 4], [4, 8]], None),
        ("large_offset", [[10**12 + i, -10**12 + 2*i] for i in [0, 2, 5, 9]], [1, 2, 0, 1]),
        ("one_parameter", [[-2], [1], [7]], [0, 2, 1]),
    ]:
        for norm in ["population", "unbiased"]:
            cases.append(dict(name=f"{name}_{norm}", operation="moments", draws=x, weights=w,
                              normalization=norm, expected=moments(x, w, norm)))
    for name, x, w in [
        ("one_draw", [[2, -3]], None),
        ("one_effective", [[2, -3], [100, 200], [5, 6]], [9, 0, 0]),
        ("huge_weights", [[-2], [4], [7]], [1e308, 1e308, 0]),
    ]:
        cases.append(dict(name=name, operation="moments", draws=x, weights=w,
                          normalization="population", expected=moments(x, w)))
    for name, source, target in [
        ("correlated_map", [[4, 1, -2], [1, 3, 0.5], [-2, 0.5, 5]],
         [[9, -3, 1], [-3, 4, 0.5], [1, 0.5, 2]]),
        ("identity_map", [[2, 1], [1, 3]], [[2, 1], [1, 3]]),
        ("one_parameter_map", [[4]], [[9]]),
        ("mixed_units_map", [[1e-100, 0], [0, 1e100]], [[4e-100, 0], [0, 9e100]]),
    ]:
        cases.append(dict(name=name, operation="covariance_map", source_covariance=source,
                          target_covariance=target, expected=covariance_map(source, target)))
    for name, x, a, b in [
        ("shear_rotation_reflection", correlated, [[0, 2, 1], [1, -1, 0.5], [0, 0, 3]], [4, -2, 7]),
        ("pivot_negative_determinant", [[1, 2], [-3, 5], [0, 0]], [[0, 2], [3, 1]], [-1, 4]),
        ("one_parameter_affine", [[1], [4]], [[-2]], [3]),
        ("tiny_logdet", [[1, 0], [0, 1]], [[1e-200, 0], [0, 2e-200]], [0, 0]),
        ("huge_logdet", [[1, 0], [0, 1]], [[1e200, 0], [0, 2e200]], [0, 0]),
        ("mixed_units_affine", [[1, 2], [-3, 4]], [[1e-100, 0], [0, -1e100]], [0, 0]),
    ]:
        expected, inv, back_shift = affine(x, a, b)
        cases.append(dict(name=name, operation="affine", draws=x, matrix=a, shift=b,
                          inverse=inv, inverse_shift=back_shift, expected=expected))
    fixture = dict(generator="tests/generate-draw-fixtures.py", python=platform.python_version(),
                   backend="Python decimal, precision=80; independent scalar loops, Cholesky, backward solve and pivoted Gauss-Jordan",
                   tolerance="1e-11 + 1e-11*abs(reference); extreme scale outputs use relative 1e-12",
                   cases=cases)
    path = Path(__file__).resolve().parent / "fixtures/draws-python.json"
    path.write_text(json.dumps(floats(fixture), indent=2, allow_nan=False) + "\n")
    print(f"Generated {len(cases)} independent draw cases: {path}")


if __name__ == "__main__":
    main()
