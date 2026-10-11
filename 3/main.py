from pathlib import Path
from typing import Callable
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
OUT_DIR = Path("results")
OUT_DIR.mkdir(exist_ok=True)
 
 
class Counter:
    """Считает обращения к целевой функции."""
 
    def __init__(self, function: Callable):
        self.function = function
        self.count = 0
 
    def __call__(self, x):
        self.count += 1
        return self.function(x)
 
 
def newton_method(function, gradient, hessian, x0, eps: float = 1e-6,
                   max_iter: int = 200_000):
    """Метод Ньютона.
    """
    f = Counter(function)
    x = np.array(x0, dtype=float)
    f(x)
    for _ in range(max_iter):
        g = gradient(x)
        if np.linalg.norm(g) < eps:
            break
        step = np.linalg.solve(hessian, g)
        x = x - step
        f(x)
    return x, function(x), f.count
 
 
def newton_raphson(function, gradient, hessian, x0, t: float, eps: float = 1e-6,
                    max_iter: int = 200_000):
    """Метод Ньютона-Рафсона (демпфированный метод Ньютона).
    """
    f = Counter(function)
    x = np.array(x0, dtype=float)
    f(x)
    for _ in range(max_iter):
        g = gradient(x)
        if np.linalg.norm(g) < eps:
            break
        step = np.linalg.solve(hessian, g)
        x = x - t * step
        f(x)
    return x, function(x), f.count
 
 
def f1(x):
    return (x[0] + x[1] - 1) ** 2 + 2 * (x[0] - 2) ** 2
 
 
def grad1(x):
    return np.array([
        2 * (x[0] + x[1] - 1) + 4 * (x[0] - 2),
        2 * (x[0] + x[1] - 1),
    ])
 
 
H1 = np.array([[6.0, 2.0], [2.0, 2.0]])
 
 
def f2(x):
    return x[1] ** 2 + 6 * x[0] ** 2 + (5 * x[0] + 2 * x[1]) ** 2 + 1
 
 
def grad2(x):
    return np.array([
        62 * x[0] + 20 * x[1],
        20 * x[0] + 10 * x[1],
    ])
 
 
H2 = np.array([[62.0, 20.0], [20.0, 10.0]])
 
FUNCTIONS_2D = {
    "f1": (f1, grad1, H1, np.array([0.0, 0.0]), np.array([2.0, -1.0]), 0.0),
    "f2": (f2, grad2, H2, np.array([5.0, 5.0]), np.array([0.0, 0.0]), 1.0),
}
 
 
def run_control_test() -> list[str]:
    """
    Проверка реализации на простой квадратичной функции
    f(x1, x2) = x1^2 + x2^2, минимум которой заведомо известен: x* = (0, 0),
    f(x*) = 0. Такая функция аналогична квадратичным примерам раздела 2
    учебника Пантелеева и Летовой и удобна для контроля правильности
    методов второго порядка.
    """
    test_f = lambda x: x[0] ** 2 + x[1] ** 2
    test_grad = lambda x: np.array([2 * x[0], 2 * x[1]])
    test_H = np.array([[2.0, 0.0], [0.0, 2.0]])
    x0 = np.array([3.0, 4.0])
 
    newton = newton_method(test_f, test_grad, test_H, x0)
    nr = newton_raphson(test_f, test_grad, test_H, x0, t=1.0)
 
    assert np.allclose(newton[0], [0, 0], atol=1e-9)
    assert np.allclose(nr[0], [0, 0], atol=1e-9)
 
    return [
        f"Метод Ньютона: x*=({newton[0][0]:.6f}, {newton[0][1]:.6f}), "
        f"f*={newton[1]:.6e}, вычислений={newton[2]}",
        f"Метод Ньютона-Рафсона (t=1): x*=({nr[0][0]:.6f}, {nr[0][1]:.6f}), "
        f"f*={nr[1]:.6e}, вычислений={nr[2]}",
    ]
 
 
def two_dimensional_experiments() -> pd.DataFrame:
    rows: list[dict] = []
    for name, (function, gradient, hessian, x0, true_x, true_value) in FUNCTIONS_2D.items():
        eps = 0.0001
        while eps <= 0.01 + 1e-12:
            x, fx, cnt = newton_method(function, gradient, hessian, x0, eps=eps)
            rows.append({
                "method": "Метод Ньютона", "function": name,
                "parameter": "eps", "value": eps,
                "x1": x[0], "x2": x[1], "fx": fx,
                "error": abs(fx - true_value), "evaluations": cnt,
            })
            eps += 0.0001
 
        t = 0.05
        while t <= 1.0 + 1e-9:
            x, fx, cnt = newton_raphson(function, gradient, hessian, x0, t)
            rows.append({
                "method": "Метод Ньютона-Рафсона", "function": name,
                "parameter": "t", "value": t,
                "x1": x[0], "x2": x[1], "fx": fx,
                "error": abs(fx - true_value), "evaluations": cnt,
            })
            t += 0.05
 
    return pd.DataFrame(rows)
 
 
def plot_metric(frame: pd.DataFrame, method: str, parameter: str, metric: str, filename: str) -> None:
    subset = frame[(frame["method"] == method) & (frame["parameter"] == parameter)]
 
    fig, ax = plt.subplots(figsize=(8.2, 4.7))
    for function, data in subset.groupby("function"):
        ax.plot(data["value"], data[metric], marker="o", markersize=3, linewidth=1.2, label=function)
 
    ax.set_xlabel({"eps": "Точность eps", "t": "Коэффициент демпфирования t"}[parameter])
    ax.set_ylabel("Погрешность" if metric == "error" else "Количество вычислений функции")
    if metric == "error":
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=180)
    plt.close(fig)
 
 
def build_summary(all_results: pd.DataFrame) -> pd.DataFrame:
    descriptions = {
        ("Метод Ньютона", "eps"): "eps = 0,0001...0,01, шаг 0,0001",
        ("Метод Ньютона-Рафсона", "t"): "t = 0,05...1,00, шаг 0,05",
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
    all_results = two_dimensional_experiments()
    all_results.to_csv(OUT_DIR / "all_results.csv", index=False)
 
    plot_metric(all_results, "Метод Ньютона", "eps", "error", "newton_error.png")
    plot_metric(all_results, "Метод Ньютона", "eps", "evaluations", "newton_speed.png")
    plot_metric(all_results, "Метод Ньютона-Рафсона", "t", "error", "nr_error.png")
    plot_metric(all_results, "Метод Ньютона-Рафсона", "t", "evaluations", "nr_speed.png")
 
    summary = build_summary(all_results)
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    save_table_image(summary)
 
    lines = ["КОНТРОЛЬНЫЙ ТЕСТ", *control_lines, "", "ТОЧНЫЕ МИНИМУМЫ"]
    for name, (_, _, _, _, true_x, true_value) in FUNCTIONS_2D.items():
        lines.append(f"{name}: x*=({true_x[0]:.6f}, {true_x[1]:.6f}), f*={true_value:.6f}")
    lines += ["", "СВОДНАЯ ТАБЛИЦА", summary.to_string(index=False)]
    print("\n".join(lines))
 
 
if __name__ == "__main__":
    main()
