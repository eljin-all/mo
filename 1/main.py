"""
Практическая работа №1. Вариант 1.
Методы: равномерный поиск, квадратичная интерполяция (метод парабол).
"""
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = Path("results")
OUT_DIR.mkdir(exist_ok=True)


@dataclass
class Result1D:
    x: float
    fx: float
    evaluations: int


class Counter:
    """Считает обращения к целевой функции."""

    def __init__(self, function: Callable):
        self.function = function
        self.count = 0

    def __call__(self, x):
        self.count += 1
        return self.function(x)


def uniform_search(function: Callable[[float], float], a: float, b: float, n: int) -> Result1D:
    """Равномерно проверяет n + 1 точку отрезка [a, b]."""
    if n < 1:
        raise ValueError("Количество частей n должно быть положительным")
    f = Counter(function)
    best_x = a
    best_value = f(a)
    step = (b - a) / n
    for i in range(1, n + 1):
        x = a + i * step
        value = f(x)
        if value < best_value:
            best_x, best_value = x, value
    return Result1D(best_x, best_value, f.count)


def quadratic_interpolation(function: Callable[[float], float], a: float, b: float,
                             eps: float, max_iter: int = 500) -> Result1D:
    """Метод квадратичной интерполяции (метод парабол).

    Через три точки x1 < x2 < x3 (f(x2) не больше значений на краях)
    строится парабола, вычисляется абсцисса её вершины x, после чего
    тройка точек сужается так, чтобы минимум снова оказался внутри
    интервала. Итерации останавливаются, когда очередная вершина
    параболы смещается меньше, чем на eps.
    """
    f = Counter(function)
    x1, x3 = a, b
    x2 = (a + b) / 2
    f1, f2, f3 = f(x1), f(x2), f(x3)

    for _ in range(max_iter):
        denom = (x2 - x1) * (f2 - f3) - (x2 - x3) * (f2 - f1)
        if abs(denom) < 1e-14:
            break
        x_bar = x2 - 0.5 * ((x2 - x1) ** 2 * (f2 - f3) - (x2 - x3) ** 2 * (f2 - f1)) / denom
        x_bar = min(max(x_bar, x1), x3)
        f_bar = f(x_bar)

        if abs(x_bar - x2) < eps:
            if f_bar < f2:
                x2, f2 = x_bar, f_bar
            break

        if x_bar < x2:
            if f_bar < f2:
                x3, f3 = x2, f2
                x2, f2 = x_bar, f_bar
            else:
                x1, f1 = x_bar, f_bar
        else:
            if f_bar < f2:
                x1, f1 = x2, f2
                x2, f2 = x_bar, f_bar
            else:
                x3, f3 = x_bar, f_bar

    return Result1D(x2, f2, f.count)


def f1(x: float) -> float:
    return (2 * x + 3) ** 2 - 8 * x - 10


def f2(x: float) -> float:
    return 4 * (x - 5) ** 2 + math.exp(x) * (x - 6) ** 2


FUNCTIONS_1D = {
    "f1": (f1, -6.0, 6.0, -0.5, -2.0),
    "f2": (f2, 0.0, 6.0, 5.990035844581383, 3.9603409528734455),
}


def run_control_test() -> list[str]:
    """
    Проверка методов на функции f(x) = 2x^2 - 12x из примеров 5.2-5.3
    учебника Пантелеева и Летовой. Точный минимум на отрезке [0; 10]:
    x* = 3, f* = -18.
    """
    test = lambda x: 2 * x ** 2 - 12 * x
    uniform = uniform_search(test, 0, 10, 10)
    quad = quadratic_interpolation(test, 0, 10, 1e-4)

    assert abs(uniform.x - 3) <= 0.01
    assert abs(uniform.fx - (-18)) <= 0.01
    assert abs(quad.x - 3) <= 1e-4
    assert abs(quad.fx - (-18)) <= 1e-4

    return [
        f"Равномерный поиск: x*={uniform.x:.6f}, f*={uniform.fx:.6f}, "
        f"вычислений={uniform.evaluations}",
        f"Квадратичная интерполяция: x*={quad.x:.6f}, f*={quad.fx:.6f}, "
        f"вычислений={quad.evaluations}",
    ]


def one_dimensional_experiments() -> pd.DataFrame:
    rows: list[dict] = []
    for name, (function, a, b, true_x, true_value) in FUNCTIONS_1D.items():
        for n in range(20, 1001, 20):
            result = uniform_search(function, a, b, n)
            rows.append({
                "method": "Равномерный поиск", "function": name,
                "parameter": "n", "value": n,
                "x": result.x, "fx": result.fx,
                "error": abs(result.fx - true_value),
                "evaluations": result.evaluations,
            })

        eps = 0.005
        while eps <= 0.5 + 1e-9:
            result = quadratic_interpolation(function, a, b, eps)
            rows.append({
                "method": "Квадратичная интерполяция", "function": name,
                "parameter": "eps", "value": eps,
                "x": result.x, "fx": result.fx,
                "error": abs(result.fx - true_value),
                "evaluations": result.evaluations,
            })
            eps += 0.005

    return pd.DataFrame(rows)


def plot_metric(frame: pd.DataFrame, method: str, parameter: str, metric: str, filename: str) -> None:
    subset = frame[(frame["method"] == method) & (frame["parameter"] == parameter)]
    grouped = subset.groupby(["function", "value"], as_index=False)[metric].mean()

    fig, ax = plt.subplots(figsize=(8.2, 4.7))
    for function, data in grouped.groupby("function"):
        ax.plot(data["value"], data[metric], marker="o", markersize=3, linewidth=1.4, label=function)

    ax.set_xlabel({"n": "Количество частей n", "eps": "Точность eps"}[parameter])
    ax.set_ylabel("Погрешность" if metric == "error" else "Количество вычислений функции")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=180)
    plt.close(fig)


def build_summary(all_results: pd.DataFrame) -> pd.DataFrame:
    descriptions = {
        ("Равномерный поиск", "n"): "n = 20...1000, шаг 20",
        ("Квадратичная интерполяция", "eps"): "eps = 0,005...0,500, шаг 0,005",
    }
    rows = []
    for (method, function, parameter), group in all_results.groupby(["method", "function", "parameter"]):
        rows.append({
            "Метод": method,
            "Функция": function,
            "Параметры": descriptions[(method, parameter)],
            "Погрешность средняя": group["error"].mean(),
            "Погрешность лучшая": group["error"].min(),
            "Погрешность худшая": group["error"].max(),
            "Вычисления средние": group["evaluations"].mean(),
            "Вычисления лучшие": group["evaluations"].min(),
            "Вычисления худшие": group["evaluations"].max(),
        })
    return pd.DataFrame(rows)


def save_table_image(summary: pd.DataFrame) -> None:
    shown = summary.copy()
    for column in ["Погрешность средняя", "Погрешность лучшая", "Погрешность худшая"]:
        shown[column] = shown[column].map(lambda v: f"{v:.3e}")
    for column in ["Вычисления средние", "Вычисления лучшие", "Вычисления худшие"]:
        shown[column] = shown[column].map(lambda v: f"{v:.1f}")

    fig, ax = plt.subplots(figsize=(16, 3.2))
    ax.axis("off")
    table = ax.table(cellText=shown.values, colLabels=shown.columns, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2.0)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "summary_table.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    control_lines = run_control_test()
    all_results = one_dimensional_experiments()
    all_results.to_csv(OUT_DIR / "all_results.csv", index=False)

    plot_metric(all_results, "Равномерный поиск", "n", "error", "uniform_error.png")
    plot_metric(all_results, "Равномерный поиск", "n", "evaluations", "uniform_speed.png")
    plot_metric(all_results, "Квадратичная интерполяция", "eps", "error", "quad_error.png")
    plot_metric(all_results, "Квадратичная интерполяция", "eps", "evaluations", "quad_speed.png")

    summary = build_summary(all_results)
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    save_table_image(summary)

    lines = ["КОНТРОЛЬНЫЙ ТЕСТ", *control_lines, "", "ТОЧНЫЕ МИНИМУМЫ"]
    for name, (_, _, _, true_x, true_value) in FUNCTIONS_1D.items():
        lines.append(f"{name}: x*={true_x:.9f}, f*={true_value:.9f}")
    lines += ["", "СВОДНАЯ ТАБЛИЦА", summary.to_string(index=False)]
    print("\n".join(lines))


if __name__ == "__main__":
    main()