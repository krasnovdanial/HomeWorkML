import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.nonparametric.smoothers_lowess import lowess


def plot_box_comparison(
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        x_col: str,
        y_col: str,
        plot_title: str
):
    """
        Строит два boxplot-графика (Train и Test) рядом с общей осью Y.

        Аргументы:
            df_train (pd.DataFrame): Обучающий набор данных.
            df_test (pd.DataFrame): Тестовый набор данных.
            x_col (str): Категориальный столбец для оси X.
            y_col (str): Числовой столбец для оси Y.
            plot_title (str): Текст, который добавляется в заголовок графика.
        """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    sns.boxplot(
        data=df_train,
        x=x_col,
        y=y_col,
        ax=axes[0],
        color="lightblue"
    )
    axes[0].set_title(f"Train: {plot_title} {x_col}", fontsize=14)
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)

    sns.boxplot(
        data=df_test,
        x=x_col,
        y=y_col,
        ax=axes[1],
        color="orange"
    )
    axes[1].set_title(f"Test: {plot_title} {x_col}", fontsize=14)
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    return fig


def plot_scatter_comparison(
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        x_col: str,
        y_col: str,
        plot_title: str = ""
):
    """
    Строит два scatterplot-графика (Train и Test) рядом.

    Аргументы:
        df_train (pd.DataFrame): Обучающий набор.
        df_test (pd.DataFrame): Тестовый набор.
        x_col (str): Столбец по оси X.
        y_col (str): Столбец по оси Y.
        plot_title (str): Префикс к заголовку графика.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    sns.scatterplot(
        data=df_train, x=x_col, y=y_col, ax=axes[0],
        color="blue", alpha=0.6
    )
    axes[0].set_title(f"Train: {plot_title} {x_col}", fontsize=14)
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    sns.scatterplot(
        data=df_test, x=x_col, y=y_col, ax=axes[1],
        color="orange", alpha=0.6
    )
    axes[1].set_title(f"Test: {plot_title} {x_col}", fontsize=14)
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    return fig


def plot_train_test_pairplot(
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        columns: list[str]
):
    """
    Строит PairPlot для указанных признаков, сравнивая распределения Train и Test.
    """

    train_tmp = df_train[columns].copy()
    test_tmp = df_test[columns].copy()

    train_tmp["Dataset"] = "Train"
    test_tmp["Dataset"] = "Test"

    df_combined = pd.concat([train_tmp, test_tmp], ignore_index=True)

    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.0)

    grid = sns.pairplot(
        df_combined,
        vars=columns,
        hue="Dataset",
        palette={"Train": "blue", "Test": "orange"},
        diag_kind="hist",
        corner=True,
        height=2.5,
        plot_kws={"alpha": 0.3, "s": 15},
    )

    grid.figure.suptitle(
        "Сравнение признаков Train vs Test",
        y=1.02,
        fontsize=15
    )

    return grid.figure


def plot_feature_target_relationships_overlay(
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        features: list[str],
        target: str = "log_selling_price",
        smooth_frac: float = 0.3
):
    """
    Строит графики зависимости признаков от таргета,
    накладывая Train и Test друг на друга.

    Синие точки — Train
    Оранжевые точки — Test
    Красная линия — тренд Train
    Зеленая линия — тренд Test
    """

    n_cols = 3
    n_rows = int(np.ceil(len(features) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, n_rows * 6))
    axes = axes.flatten()

    for idx, feature in enumerate(features):

        ax = axes[idx]

        ax.scatter(df_train[feature], df_train[target],
                   color="blue", alpha=0.25, s=12, label="Train")

        ax.scatter(df_test[feature], df_test[target],
                   color="orange", alpha=0.25, s=12, label="Test")

        try:
            sm_train = lowess(df_train[target], df_train[feature],
                              frac=smooth_frac, return_sorted=True)
            ax.plot(sm_train[:, 0], sm_train[:, 1], color="red", linewidth=2, label="Train trend")
        except Exception:
            pass

        try:
            sm_test = lowess(df_test[target], df_test[feature],
                             frac=smooth_frac, return_sorted=True)
            ax.plot(sm_test[:, 0], sm_test[:, 1], color="green", linewidth=2, label="Test trend")
        except Exception:
            pass

        ax.set_title(f"{feature} и {target}")
        ax.set_xlabel(feature)
        ax.set_ylabel(target)
        ax.grid(alpha=0.3)

        if idx == 0:
            ax.legend()

    plt.tight_layout()
    return fig


def plot_target_distribution(
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        target_col: str
):
    """
    Сравнивает распределение целевой переменной в Train и Test
    в оригинальном масштабе и в логарифмическом.

    Аргументы:
        df_train (pd.DataFrame): Обучающий набор.
        df_test (pd.DataFrame): Тестовый набор.
        target_col (str): Название таргета.
    """
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    sns.kdeplot(
        df_train[target_col], fill=True, ax=ax[0],
        label="Train", color="blue", alpha=0.3
    )
    sns.kdeplot(
        df_test[target_col], fill=True, ax=ax[0],
        label="Test", color="orange", alpha=0.3
    )
    ax[0].set_title("Исходное распределение")
    ax[0].legend()
    ax[0].grid(linestyle="--", alpha=0.5)

    sns.kdeplot(
        np.log1p(df_train[target_col]), fill=True, ax=ax[1],
        label="Train", color="blue", alpha=0.3
    )
    sns.kdeplot(
        np.log1p(df_test[target_col]), fill=True, ax=ax[1],
        label="Test", color="orange", alpha=0.3
    )
    ax[1].set_title("Логарифмированное распределение")
    ax[1].legend()
    ax[1].grid(linestyle="--", alpha=0.5)

    plt.tight_layout()
    return fig


def compare_corr_heatmap(
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        plot_title: str = ""
):
    """
    Строит две матрицы корреляций для Train и Test
    с единой цветовой шкалой.

    Аргументы:
        df_train (pd.DataFrame): Обучающий набор.
        df_test (pd.DataFrame): Тестовый набор.
        plot_title (str): Заголовок всей визуализации.
    """
    corr_train = df_train.corr(numeric_only=True)
    corr_test = df_test.corr(numeric_only=True)

    vmin = min(corr_train.min().min(), corr_test.min().min())
    vmax = max(corr_train.max().max(), corr_test.max().max())

    mask_train = np.triu(np.ones_like(corr_train, dtype=bool))
    mask_test = np.triu(np.ones_like(corr_test, dtype=bool))

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    sns.heatmap(
        corr_train,
        mask=mask_train,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=vmin,
        vmax=vmax,
        linewidths=.5,
        ax=axes[0],
        cbar=False
    )
    axes[0].set_title("Обучающая выборка Train")

    sns.heatmap(
        corr_test,
        mask=mask_test,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=vmin,
        vmax=vmax,
        linewidths=.5,
        ax=axes[1],
        cbar=True
    )
    axes[1].set_title("Тестовая выборка Test")

    fig.suptitle(plot_title, fontsize=18)
    plt.tight_layout()
    return fig


def box_plot_for_categories(df_train: pd.DataFrame, df_test: pd.DataFrame, x_col: str, y_col: str,
                            plot_title: str = ""):
    """
    Строит сравнительные boxplot'ы специально для признака 'seats' или других дискретных.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    sns.boxplot(data=df_train, x=x_col, y=y_col, ax=axes[0], palette='Blues')
    axes[0].set_title(f"Train: {plot_title} {x_col}", fontsize=14)
    axes[0].tick_params(axis='x', rotation=0)
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)

    sns.boxplot(data=df_test, x=x_col, y=y_col, ax=axes[1], palette='Oranges')
    axes[1].set_title(f"Test: {plot_title} {x_col}", fontsize=14)
    axes[1].tick_params(axis='x', rotation=0)
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    return fig
