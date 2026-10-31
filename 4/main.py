
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = Path("results")
OUT_DIR.mkdir(exist_ok=True)
XOR_TABLE = np.array([
    [0.0, 0.0, 0.0],
    [0.0, 1.0, 1.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
])

BETA = 1.0


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-BETA * z))


def xor_network_output(w: np.ndarray, u1: float, u2: float) -> float:
    """Прямой проход сети рис. 4.2: 2 входа - 2 скрытых нейрона - 1 выход."""
    w11, w12, w21, w22, w31, w32, w10, w20, w30 = w
    h1 = sigmoid(w11 * u1 + w12 * u2 + w10)
    h2 = sigmoid(w21 * u1 + w22 * u2 + w20)
    y = sigmoid(w31 * h1 + w32 * h2 + w30)
    return y


def error_Q(w: np.ndarray) -> float:
    """Среднеквадратичная погрешность сети на всех 4 образах XOR."""
    errors = []
    for u1, u2, d in XOR_TABLE:
        y = xor_network_output(w, u1, u2)
        errors.append((d - y) ** 2)
    return float(np.mean(errors))


def fitness_from_error(q: float) -> float:
    """Функция приспособленности: неотрицательна, максимизируется, равна 1 при Q=0."""
    return 1.0 / (1.0 + q)


@dataclass
class Encoding:
    """Двоичное кодирование n_vars вещественных переменных в интервале [lo, hi]."""
    n_vars: int
    lo: float
    hi: float
    bits_per_var: int = 16

    @property
    def chrom_len(self) -> int:
        return self.n_vars * self.bits_per_var

    def decode(self, bits: np.ndarray) -> np.ndarray:
        bits = bits.reshape(self.n_vars, self.bits_per_var)
        powers = 2 ** np.arange(self.bits_per_var - 1, -1, -1)
        ints = bits @ powers
        max_int = 2 ** self.bits_per_var - 1
        return self.lo + (ints / max_int) * (self.hi - self.lo)


@dataclass
class GAResult:
    best_weights: np.ndarray
    best_error: float
    best_fitness: float
    generations_run: int
    evaluations: int
    best_history: list = field(default_factory=list)
    worst_history: list = field(default_factory=list)
    mean_history: list = field(default_factory=list)


def genetic_algorithm(
    objective_error: Callable[[np.ndarray], float],
    encoding: Encoding,
    pop_size: int = 100,
    generations: int = 150,
    pc: float = 0.8,
    pm: float = 0.02,
    tournament_k: int = 2,
    elitism: bool = True,
    target_error: float = 1e-5,
    rng: np.random.Generator | None = None,
) -> GAResult:
    """Классический генетический алгоритм с двоичным кодированием.
    """
    if rng is None:
        rng = np.random.default_rng()

    L = encoding.chrom_len
    population = rng.integers(0, 2, size=(pop_size, L))

    evaluations = 0
    best_history, worst_history, mean_history = [], [], []
    best_weights_overall = None
    best_error_overall = np.inf

    def evaluate(pop: np.ndarray):
        nonlocal evaluations
        errors = np.empty(pop_size)
        for idx in range(pop_size):
            weights = encoding.decode(pop[idx])
            errors[idx] = objective_error(weights)
        evaluations += pop_size
        fitness = 1.0 / (1.0 + errors)
        return errors, fitness

    def tournament_selection(fitness: np.ndarray) -> np.ndarray:
        new_pop = np.empty_like(population)
        for i in range(pop_size):
            contenders = rng.integers(0, pop_size, size=tournament_k)
            winner = contenders[np.argmax(fitness[contenders])]
            new_pop[i] = population[winner]
        return new_pop

    generations_run = 0
    for gen in range(generations):
        errors, fitness = evaluate(population)

        best_idx = int(np.argmax(fitness))
        worst_idx = int(np.argmin(fitness))
        best_history.append(fitness[best_idx])
        worst_history.append(fitness[worst_idx])
        mean_history.append(float(np.mean(fitness)))

        if errors[best_idx] < best_error_overall:
            best_error_overall = errors[best_idx]
            best_weights_overall = encoding.decode(population[best_idx])

        generations_run = gen + 1
        if best_error_overall <= target_error:
            break

        elite_chrom = population[best_idx].copy()

        parents = tournament_selection(fitness)

        children = parents.copy()
        for i in range(0, pop_size - 1, 2):
            if rng.random() < pc:
                point = rng.integers(1, L)
                children[i, point:], children[i + 1, point:] = (
                    parents[i + 1, point:].copy(),
                    parents[i, point:].copy(),
                )
        mutation_mask = rng.random((pop_size, L)) < pm
        children[mutation_mask] = 1 - children[mutation_mask]

        if elitism:
            children[worst_idx] = elite_chrom

        population = children

    return GAResult(
        best_weights=best_weights_overall,
        best_error=best_error_overall,
        best_fitness=fitness_from_error(best_error_overall),
        generations_run=generations_run,
        evaluations=evaluations,
        best_history=best_history,
        worst_history=worst_history,
        mean_history=mean_history,
    )


def run_control_test() -> list[str]:
    def control_error(w: np.ndarray) -> float:
        x = w[0]
        f_max = 100.0
        return f_max - x ** 2

    enc = Encoding(n_vars=1, lo=-10.0, hi=10.0, bits_per_var=16)
    rng = np.random.default_rng(0)
    result = genetic_algorithm(
        control_error, enc, pop_size=40, generations=60,
        pc=0.8, pm=0.02, elitism=True, target_error=1e-6, rng=rng,
    )
    x_best = result.best_weights[0]
    f_best = x_best ** 2
    assert abs(x_best - 10.0) < 0.01, "ГА не нашёл известный максимум x*=10"
    return [
        f"Контрольная задача: max f(x)=x^2, x in [-10,10], f* = 100 при x* = 10",
        f"Найдено генетическим алгоритмом: x = {x_best:.6f}, f(x) = {f_best:.6f}, "
        f"поколений = {result.generations_run}, вычислений = {result.evaluations}",
    ]


ENCODING_XOR = Encoding(n_vars=9, lo=-10.0, hi=10.0, bits_per_var=16)

DEFAULT_PARAMS = dict(pop_size=100, generations=200, pc=0.8, pm=0.02,
                       tournament_k=2, elitism=True, target_error=1e-5)


def run_xor_training(seed: int = 1, **overrides) -> GAResult:
    params = {**DEFAULT_PARAMS, **overrides}
    rng = np.random.default_rng(seed)
    return genetic_algorithm(error_Q, ENCODING_XOR, rng=rng, **params)


N_REPEATS = 8


def sweep_population_size() -> pd.DataFrame:
    values = [20, 40, 60, 80, 100, 140, 180, 220]
    rows = []
    for n in values:
        for rep in range(N_REPEATS):
            rng = np.random.default_rng(1000 * n + rep)
            result = genetic_algorithm(
                error_Q, ENCODING_XOR, pop_size=n, generations=200,
                pc=0.8, pm=0.02, elitism=True, target_error=1e-5, rng=rng,
            )
            rows.append({
                "parameter": "pop_size", "value": n,
                "error": result.best_error, "evaluations": result.evaluations,
                "generations": result.generations_run,
            })
    return pd.DataFrame(rows)


def sweep_mutation_prob() -> pd.DataFrame:
    values = [0.0, 0.005, 0.01, 0.02, 0.04, 0.08, 0.15, 0.30]
    rows = []
    for pm in values:
        for rep in range(N_REPEATS):
            rng = np.random.default_rng(int(2000 + pm * 10000) + rep)
            result = genetic_algorithm(
                error_Q, ENCODING_XOR, pop_size=100, generations=200,
                pc=0.8, pm=pm, elitism=True, target_error=1e-5, rng=rng,
            )
            rows.append({
                "parameter": "pm", "value": pm,
                "error": result.best_error, "evaluations": result.evaluations,
                "generations": result.generations_run,
            })
    return pd.DataFrame(rows)


def plot_history(result: GAResult, filename: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.7))
    gens = np.arange(1, len(result.best_history) + 1)
    ax.plot(gens, result.best_history, label="лучший индивид", linewidth=1.4)
    ax.plot(gens, result.worst_history, label="худший индивид", linewidth=1.0, alpha=0.7)
    ax.plot(gens, result.mean_history, label="средняя по популяции", linewidth=1.2)
    ax.set_xlabel("Поколение")
    ax.set_ylabel("Приспособленность (1 / (1 + Q))")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=180)
    plt.close(fig)


def plot_sweep(frame: pd.DataFrame, parameter: str, xlabel: str, metric: str,
               filename: str, logy: bool = False) -> None:
    grouped = frame.groupby("value")[metric].agg(["mean", "min", "max"])
    fig, ax = plt.subplots(figsize=(8.2, 4.7))
    ax.plot(grouped.index, grouped["mean"], marker="o", markersize=4, label="среднее")
    ax.fill_between(grouped.index, grouped["min"], grouped["max"], alpha=0.2,
                     label="диапазон (мин-макс)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Погрешность Q" if metric == "error" else "Количество вычислений функции")
    if logy:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=180)
    plt.close(fig)


def build_summary(pop_frame: pd.DataFrame, pm_frame: pd.DataFrame) -> pd.DataFrame:
    descriptions = {
        "pop_size": "N = 20...220 (переменная), G=200, pc=0,8, pm=0,02",
        "pm": "pm = 0...0,30 (переменная), G=200, pc=0,8, N=100",
    }
    rows = []
    for param, frame in (("pop_size", pop_frame), ("pm", pm_frame)):
        for value, group in frame.groupby("value"):
            rows.append({
                "Параметр": param,
                "Значение": value,
                "Условия": descriptions[param],
                "Погрешность средняя": group["error"].mean(),
                "Погрешность лучшая": group["error"].min(),
                "Погрешность худшая": group["error"].max(),
                "Вычисления средние": group["evaluations"].mean(),
                "Вычисления лучшие": group["evaluations"].min(),
                "Вычисления худшие": group["evaluations"].max(),
                "Доля успешных запусков": float(np.mean(group["error"] <= 1e-4)),
            })
    return pd.DataFrame(rows)


def save_table_image(summary: pd.DataFrame, filename: str, max_rows: int | None = None) -> None:
    shown = summary.copy()
    if max_rows:
        shown = shown.head(max_rows)
    for column in ["Погрешность средняя", "Погрешность лучшая", "Погрешность худшая"]:
        shown[column] = shown[column].map(lambda v: f"{v:.3e}")
    for column in ["Вычисления средние", "Вычисления лучшие", "Вычисления худшие"]:
        shown[column] = shown[column].map(lambda v: f"{v:.1f}")
    shown["Доля успешных запусков"] = shown["Доля успешных запусков"].map(lambda v: f"{v:.2f}")

    fig, ax = plt.subplots(figsize=(18, 0.5 * len(shown) + 1.2))
    ax.axis("off")
    table = ax.table(cellText=shown.values, colLabels=shown.columns, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    control_lines = run_control_test()

    base_result = run_xor_training(seed=1)
    plot_history(base_result, "xor_fitness_history.png",
                 "Обучение весов сети XOR генетическим алгоритмом "
                 "(N=100, pc=0.8, pm=0.02)")

    pop_frame = sweep_population_size()
    pop_frame.to_csv(OUT_DIR / "sweep_pop_size.csv", index=False)
    plot_sweep(pop_frame, "pop_size", "Размер популяции N", "error",
               "pop_size_error.png", logy=True)
    plot_sweep(pop_frame, "pop_size", "Размер популяции N", "evaluations",
               "pop_size_evaluations.png")

    pm_frame = sweep_mutation_prob()
    pm_frame.to_csv(OUT_DIR / "sweep_pm.csv", index=False)
    plot_sweep(pm_frame, "pm", "Вероятность мутации pm", "error",
               "pm_error.png", logy=True)
    plot_sweep(pm_frame, "pm", "Вероятность мутации pm", "evaluations",
               "pm_evaluations.png")

    summary = build_summary(pop_frame, pm_frame)
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    save_table_image(summary[summary["Параметр"] == "pop_size"],
                      "summary_table_pop_size.png")
    save_table_image(summary[summary["Параметр"] == "pm"],
                      "summary_table_pm.png")

    w = base_result.best_weights
    lines = ["КОНТРОЛЬНЫЙ ТЕСТ", *control_lines, "",
             "ОБУЧЕНИЕ СЕТИ XOR (базовый запуск, seed=1)",
             f"Поколений выполнено: {base_result.generations_run}",
             f"Вычислений функции погрешности: {base_result.evaluations}",
             f"Итоговая погрешность Q = {base_result.best_error:.6e}",
             f"Итоговая приспособленность = {base_result.best_fitness:.6f}",
             "Найденные веса:",
             f"  w11={w[0]:.4f} w12={w[1]:.4f} w21={w[2]:.4f} w22={w[3]:.4f}",
             f"  w31={w[4]:.4f} w32={w[5]:.4f}",
             f"  w10={w[6]:.4f} w20={w[7]:.4f} w30={w[8]:.4f}",
             "",
             "Проверка на всех образах XOR:"]
    for u1, u2, d in XOR_TABLE:
        y = xor_network_output(w, u1, u2)
        lines.append(f"  u1={u1:.0f} u2={u2:.0f} -> d={d:.0f}, y_сети={y:.4f}")

    lines += ["", "СВОДНАЯ ТАБЛИЦА (фрагмент)", summary.to_string(index=False)]
    print("\n".join(lines))


if __name__ == "__main__":
    main()
