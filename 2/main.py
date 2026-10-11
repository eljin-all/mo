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
 
 
def gradient_descent_const_step(function, gradient, x0, alpha: float,
                                 eps: float = 1e-6, max_iter: int = 200_000):
    """Метод градиентного спуска с постоянным шагом alpha.
 
    x_{k+1} = x_k - alpha * grad f(x_k).
    Останавливается, когда норма градиента становится меньше eps.
    """
    f = Counter(function)
    x = np.array(x0, dtype=float)
    f(x)
    for _ in range(max_iter):
        g = gradient(x)
        if np.linalg.norm(g) < eps:
            break
        x = x - alpha * g
        f(x)
    return x, function(x), f.count
 
 
def steepest_descent(function, gradient, hessian, x0, eps: float = 1e-6,
                      max_iter: int = 200_000):
    """Метод наискорейшего градиентного спуска.
 
    На каждой итерации длина шага alpha_k выбирается так, чтобы точно
    минимизировать f вдоль направления антиградиента. Так как обе
    рассматриваемые функции квадратичные (с постоянной матрицей Гессе),
    alpha_k находится в явном виде: alpha_k = (g^T g) / (g^T H g),
    где g = grad f(x_k), H - матрица Гессе функции f.
    """
    f = Counter(function)
    x = np.array(x0, dtype=float)
    f(x)
    for _ in range(max_iter):
        g = gradient(x)
        if np.linalg.norm(g) < eps:
            break
        hg = hessian @ g
        alpha = (g @ g) / (g @ hg)
        x = x - alpha * g
        f(x)
    return x, function(x), f.count
 
 
def f1(x):
    return (x[0] - 4) ** 2 + (x[1] - 1) ** 2
 
 
def grad1(x):
    return np.array([2 * (x[0] - 4), 2 * (x[1] - 1)])
 
 
H1 = np.array([[2.0, 0.0], [0.0, 2.0]])
 
 
def f2(x):
    return 4 * x[0] ** 2 + x[0] * x[1] + 2 * x[1] ** 2 - 2 * x[1]
 
 
def grad2(x):
    return np.array([8 * x[0] + x[1], x[0] + 4 * x[1] - 2])
 
 
H2 = np.array([[8.0, 1.0], [1.0, 4.0]])
 
FUNCTIONS_2D = {
    "f1": (f1, grad1, H1, np.array([0.0, 0.0]), np.array([4.0, 1.0]), 0.0),
    "f2": (f2, grad2, H2, np.array([5.0, 5.0]),
           np.array([-0.06451613, 0.51612903]), -0.5161290322580645),
}
 
 
def run_control_test() -> list[str]:
    """
    Проверка реализации на простой квадратичной функции
    f(x1, x2) = x1^2 + x2^2, минимум которой заведомо известен: x* = (0, 0),
    f(x*) = 0. Такая функция аналогична квадратичным примерам раздела 2
    учебника Пантелеева и Летовой и удобна для контроля правильности
    градиентных методов.
    """
    test_f = lambda x: x[0] ** 2 + x[1] ** 2
    test_grad = lambda x: np.array([2 * x[0], 2 * x[1]])
    test_H = np.array([[2.0, 0.0], [0.0, 2.0]])
    x0 = np.array([3.0, 4.0])
 
    gd = gradient_descent_const_step(test_f, test_grad, x0, alpha=0.3)
    sd = steepest_descent(test_f, test_grad, test_H, x0)
 
    assert np.allclose(gd[0], [0, 0], atol=1e-3)
    assert np.allclose(sd[0], [0, 0], atol=1e-6)
 
    return [
        f"Градиентный спуск с постоянным шагом: x*=({gd[0][0]:.6f}, {gd[0][1]:.6f}), "
        f"f*={gd[1]:.6e}, вычислений={gd[2]}",
        f"Наискорейший градиентный спуск: x*=({sd[0][0]:.6f}, {sd[0][1]:.6f}), "
        f"f*={sd[1]:.6e}, вычислений={sd[2]}",
    ]
 
 
def two_dimensional_experiments() -> pd.DataFrame:
    rows: list[dict] = []
    for name, (function, gradient, hessian, x0, true_x, true_value) in FUNCTIONS_2D.items():
        alpha = 0.01
        while alpha <= 0.22 + 1e-9:
            x, fx, cnt = gradient_descent_const_step(function, gradient, x0, alpha)
            rows.append({
                "method": "Градиентный спуск (пост. шаг)", "function": name,
                "parameter": "alpha", "value": alpha,
                "x1": x[0], "x2": x[1], "fx": fx,
                "error": abs(fx - true_value), "evaluations": cnt,
            })
            alpha += 0.01
 
        eps = 0.0001
        while eps <= 0.01 + 1e-12:
            x, fx, cnt = steepest_descent(function, gradient, hessian, x0, eps=eps)
            rows.append({
                "method": "Наискорейший градиентный спуск", "function": name,
                "parameter": "eps", "value": eps,
                "x1": x[0], "x2": x[1], "fx": fx,
                "error": abs(fx - true_value), "evaluations": cnt,
            })
            eps += 0.0001
 
    return pd.DataFrame(rows)
 
 
def plot_metric(frame: pd.DataFrame, method: str, parameter: str, metric: str, filename: str) -> None:
    subset = frame[(frame["method"] == method) & (frame["parameter"] == parameter)]
 
    fig, ax = plt.subplots(figsize=(8.2, 4.7))
    for function, data in subset.groupby("function"):
        ax.plot(data["value"], data[metric], marker="o", markersize=3, linewidth=1.2, label=function)
 
    ax.set_xlabel({"alpha": "Шаг alpha", "eps": "Точность eps"}[parameter])
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
        ("Градиентный спуск (пост. шаг)", "alpha"): "alpha = 0,01...0,22, шаг 0,01",
        ("Наискорейший градиентный спуск", "eps"): "eps = 0,0001...0,01, шаг 0,0001",
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
 
    plot_metric(all_results, "Градиентный спуск (пост. шаг)", "alpha", "error", "gd_error.png")
    plot_metric(all_results, "Градиентный спуск (пост. шаг)", "alpha", "evaluations", "gd_speed.png")
    plot_metric(all_results, "Наискорейший градиентный спуск", "eps", "error", "sd_error.png")
    plot_metric(all_results, "Наискорейший градиентный спуск", "eps", "evaluations", "sd_speed.png")
 
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
 
