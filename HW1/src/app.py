import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent

sys.path.append(str(PROJECT_ROOT))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st

from HW1.utils.eda_plots import plot_box_comparison, plot_scatter_comparison, plot_train_test_pairplot, \
    compare_corr_heatmap, plot_feature_target_relationships_overlay, plot_target_distribution


@st.cache_resource
def load_model_bundle():
    with open(Path("HW1/models/best_model_bundle.pkl"), "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle["scaler"], bundle["columns_after_ohe"]


@st.cache_data
def load_eda_data():
    return pd.read_csv(Path("HW1/data/cleaned_train_before_ohe.csv"))


def generate_summary(df):
    summary = []

    if "selling_price" in df.columns:
        numeric = df.select_dtypes(include=[np.number])
        corr = numeric.corr()["selling_price"].drop("selling_price")
        corr_abs = corr.abs().sort_values(ascending=False)

        top_corr = corr_abs.head(5)

        corr_text = "### Корреляции с ценой\n"
        for feat, val in top_corr.items():
            direction = "прямая" if corr[feat] > 0 else "обратная"
            corr_text += f"- **{feat}**: корреляция {val:.2f} ({direction})\n"
        summary.append(corr_text)

    outlier_text = "### Признаки с выбросами\n"
    for col in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        low = Q1 - 1.5 * IQR
        high = Q3 + 1.5 * IQR
        outliers = ((df[col] < low) | (df[col] > high)).sum()
        if outliers > 0:
            outlier_text += f"- **{col}**: выбросов ≈ {outliers} значений\n"
    summary.append(outlier_text)

    cats = df.select_dtypes(exclude=[np.number]).columns
    if len(cats) > 0:
        cat_text = "### Категориальные признаки\n"
        for col in cats:
            top = df[col].value_counts().head(3).index.tolist()
            cat_text += f"- **{col}**: топ категории: {', '.join(top)}\n"
        summary.append(cat_text)

    return "\n\n".join(summary)


def extract_brand(row):
    brand_cols = [c for c in row.index if c.startswith("brand_")]
    for c in brand_cols:
        if row[c] == 1:
            return c.replace("brand_", "")
    return "Unknown"


def predict_from_ohe(df, model, scaler, columns_after_ohe):
    missing = set(columns_after_ohe) - set(df.columns)
    if missing:
        st.error(f"Не хватает колонок после OHE: {missing}")
        return None

    df = df[columns_after_ohe]
    df_scaled = scaler.transform(df)
    pred_log = model.predict(df_scaled)
    pred_price = np.expm1(pred_log)
    return pred_price


def plot_feature_hist(df, col):
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(7, 3))

    sns.histplot(df[col], bins=40, kde=True, ax=ax, color="#4C72B0")
    ax.set_title(f"Распределение {col}", fontsize=12, pad=8)
    ax.tick_params(labelsize=10)

    fig.tight_layout()
    return fig


def plot_corr(df):
    numeric = df.select_dtypes(include=[np.number])
    corr = numeric.corr()

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.heatmap(
        corr,
        cmap="coolwarm",
        annot=True,
        fmt=".2f",
        annot_kws={"size": 7},
        linewidths=0.3,
        linecolor="white",
        cbar_kws={"shrink": 0.7}
    )

    ax.set_title("Корреляционная матрица", fontsize=14, pad=10)

    fig.tight_layout()
    return fig


def plot_boxplot(df, col):
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(7, 3))

    sns.boxplot(
        y=df[col],
        ax=ax,
        color="#4C72B0",
        width=0.4,
        fliersize=3
    )

    ax.set_title(f"Boxplot: {col}", fontsize=12, pad=8)
    ax.tick_params(labelsize=10)

    q_low = df[col].quantile(0.01)
    q_high = df[col].quantile(0.99)
    ax.set_ylim(q_low, q_high)

    fig.tight_layout()
    return fig


def plot_cat_bar(df, col, top_n=10):
    vc = df[col].value_counts().head(top_n).reset_index()
    vc.columns = [col, "count"]

    fig = px.bar(
        vc,
        x=col,
        y="count",
        title=f"Топ-{top_n} категорий в '{col}'",
        height=350
    )

    fig.update_layout(
        xaxis_tickangle=-30,
        margin=dict(l=20, r=20, t=40, b=20),
        font=dict(size=11),
    )

    return fig


def plot_scatter_vs_price(df, feature, target="selling_price"):
    fig = px.scatter(
        df,
        x=feature,
        y=target,
        trendline="ols",
        opacity=0.5,
        title=f"{feature} vs {target}",
        height=350
    )

    fig.update_traces(marker=dict(size=5))
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))

    return fig


def main():
    st.set_page_config(page_title="Car Price Prediction", page_icon="🚗", layout="wide")
    st.title("ML-приложение для предсказания стоимости автомобиля")

    model, scaler, columns_after_ohe = load_model_bundle()
    df_eda = load_eda_data()

    menu = st.sidebar.radio(
        "Навигация",
        ["EDA", "Веса модели", "Предсказание цены"]
    )

    if menu == "EDA":
        st.header("Exploratory Data Analysis (до OHE)")
        st.markdown("Используем обучающий датасет до OHE для интерпретируемых визуализаций.")

        st.subheader("Train датасет")
        st.write("Размер:", df_eda.shape)
        st.dataframe(df_eda.head(), width='stretch')

        st.subheader("Загрузите тестовый датасет (до OHE)")
        test_file = st.file_uploader("Test CSV (до OHE)", type=["csv"], key="test_upload")

        df_test_eda = None
        if test_file:
            df_test_eda = pd.read_csv(test_file)
            st.success("Тестовый датасет загружен ✓")
            st.dataframe(df_test_eda.head(), width='stretch')

        numeric_cols = df_eda.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df_eda.select_dtypes(exclude=[np.number]).columns.tolist()

        st.subheader("1. Распределения числовых признаков")
        selected = st.multiselect(
            "Выберите признаки",
            numeric_cols,
            default=numeric_cols[:5]
        )
        for col in selected:
            fig = plot_feature_hist(df_eda, col)
            st.pyplot(fig)

        st.subheader("2. Boxplot (Train)")
        box_selected = st.multiselect(
            "Выберите признаки",
            numeric_cols,
            default=[c for c in ["selling_price", "km_driven", "max_power"] if c in numeric_cols],
            key="train_box"
        )
        for col in box_selected:
            fig = plot_boxplot(df_eda, col)
            st.pyplot(fig)

        if cat_cols:
            st.subheader("3. Категориальные признаки")
            cat_col = st.selectbox("Выберите категориальный признак", cat_cols)
            fig = plot_cat_bar(df_eda, cat_col, top_n=10)
            st.plotly_chart(fig, width='stretch')

        st.subheader("4. Корреляции Train")
        fig = plot_corr(df_eda)
        st.pyplot(fig)

        st.subheader("5.Авто генерация выводов по Train")
        summary_text = generate_summary(df_eda)
        st.markdown(summary_text)

        if df_test_eda is not None:
            st.header("Сравнение Train vs Test")

            compare_mode = st.selectbox(
                "Выберите тип сравнения",
                [
                    "Boxplot Train vs Test",
                    "Scatter Train vs Test",
                    "PairPlot Train vs Test",
                    "Heatmap корреляций Train vs Test",
                    "Зависимость признаков от таргета",
                    "Распределение таргета Train vs Test",
                    "Анализ дубликатов (Train & Test)",
                ]
            )

            if compare_mode == "Boxplot Train vs Test":
                num_cols = df_eda.select_dtypes(include=[np.number]).columns.tolist()
                all_cat = df_eda.select_dtypes(exclude=[np.number]).columns.tolist()

                x_col = st.selectbox("Категория (X)", all_cat)
                y_col = st.selectbox("Числовой признак (Y)", num_cols)

                fig = plot_box_comparison(df_eda, df_test_eda, x_col, y_col, "Boxplot:")
                st.pyplot(fig)

            if compare_mode == "Scatter Train vs Test":
                num_cols = df_eda.select_dtypes(include=[np.number]).columns.tolist()

                x_col = st.selectbox("X признак", num_cols)
                y_col = st.selectbox("Y признак", num_cols)

                fig = plot_scatter_comparison(df_eda, df_test_eda, x_col, y_col)
                st.pyplot(fig)

            if compare_mode == "PairPlot Train vs Test":
                num_cols = df_eda.select_dtypes(include=[np.number]).columns.tolist()

                cols = st.multiselect("Выберите признаки", num_cols, default=num_cols[:4])
                if cols:
                    plot_train_test_pairplot(df_eda, df_test_eda, cols)
                    st.pyplot(plt.gcf())

            if compare_mode == "Heatmap корреляций Train vs Test":
                compare_corr_heatmap(df_eda, df_test_eda)
                st.pyplot(plt.gcf())

            if compare_mode == "Зависимость признаков от таргета":
                num_cols = df_eda.select_dtypes(include=[np.number]).columns.tolist()

                features = st.multiselect("Выберите признаки", num_cols, default=num_cols[:5])
                target = st.selectbox("Таргет", num_cols)

                plot_feature_target_relationships_overlay(df_eda, df_test_eda, features, target)
                st.pyplot(plt.gcf())

            if compare_mode == "Распределение таргета Train vs Test":
                possible_tg = df_eda.select_dtypes(include=[np.number]).columns.tolist()
                target = st.selectbox("Выберите таргет", possible_tg)

                plot_target_distribution(df_eda, df_test_eda, target)
                st.pyplot(plt.gcf())

    elif menu == "Веса модели":
        st.header("Веса Ridge-модели")

        coefs = pd.DataFrame({
            "feature": columns_after_ohe,
            "coef": model.coef_
        })
        coefs["abs"] = coefs["coef"].abs()
        top20 = coefs.sort_values("abs", ascending=False).head(20)

        top20["feature_short"] = top20["feature"].apply(
            lambda x: x if len(x) < 20 else x[:20] + "..."
        )

        fig = px.bar(
            top20.sort_values("coef"),
            x="coef",
            y="feature_short",
            orientation="h",
            title="Top-20 самых важных признаков",
            height=600,
            width=900
        )

        fig.update_layout(
            xaxis_title="Вес (коэффициент модели)",
            yaxis_title="Признак",
            title_x=0.35,
            margin=dict(l=80, r=40, t=60, b=40),
            font=dict(size=14),
        )
        fig.update_xaxes(showgrid=True, gridcolor="lightgray")
        st.plotly_chart(fig, width='stretch')

    elif menu == "Предсказание цены":
        st.header("Предсказание стоимости автомобиля")

        st.markdown("""
            Загрузите CSV **после OHE**.

            Если вы загружаете несколько строк — покажем:
            - индивидуальные предсказания
            - распределение цен
            - scatter-графики
            """)

        file = st.file_uploader("Загрузите CSV-файл", type=["csv"])

        if file:
            df_user = pd.read_csv(file)
            st.subheader("Загруженные данные")
            st.dataframe(df_user.head(), width='stretch')

            preds = predict_from_ohe(df_user, model, scaler, columns_after_ohe)

            if preds is not None:
                df_user["predicted_price"] = preds.round().astype(int)

                df_user["predicted_price_fmt"] = df_user["predicted_price"].round().astype(int).apply(
                    lambda x: f"{x:,}".replace(",", " ")
                )
                df_user["brand"] = df_user.apply(extract_brand, axis=1)

                st.subheader("Предсказанные цены")

                for i, row in df_user.head(10).iterrows():
                    st.markdown(f"""
                        <div style='padding:18px; border-radius:10px; 
                                    background:#ffffff; margin-bottom:15px;
                                    border-left:6px solid #4CAF50;
                                    box-shadow: 0 2px 6px rgba(0,0,0,0.07);'>
                            <div style='font-size:20px; font-weight:600; margin-bottom:6px;'>
                                {row['brand']}
                            </div>
                            <b>Предсказанная цена:</b>
                            <span style='color:#4CAF50; font-size:20px; font-weight:700;'>
                                {row['predicted_price_fmt']} ₹
                            </span>
                            <br><br>
                            <b>Пробег:</b> {row.get('km_driven', '—')} км<br>
                            <b>Мощность:</b> {row.get('max_power', '—')} hp<br>
                            <b>Возраст автомобиля:</b> {row.get('car_age', '—')} лет<br>
                        </div>
                    """, unsafe_allow_html=True)
                st.subheader("Таблица с предсказаниями")

                st.dataframe(df_user.drop('predicted_price', axis=1), width='stretch')

                st.subheader("Распределение предсказанных цен")
                fig = px.histogram(
                    df_user,
                    x="predicted_price",
                    nbins=40,
                    title="Гистограмма предсказанных цен",
                    color_discrete_sequence=["#4CAF50"]
                )
                st.plotly_chart(fig, width='stretch')

                numeric_cols_user = [c for c in df_user.columns if df_user[c].dtype != "object"]

                if "engine_power_estimated" in numeric_cols_user:
                    st.subheader("Мощность двигателя к Цене")

                    def clean_power(x):
                        if isinstance(x, str):
                            x = x.lower().replace("hp", "").replace("bhp", "").strip()
                        return pd.to_numeric(x, errors="coerce")

                    df_user["engine_power_estimated"] = df_user["engine_power_estimated"].apply(clean_power)
                    df_user["predicted_price"] = pd.to_numeric(df_user["predicted_price"], errors="coerce")

                    fig = px.scatter(
                        df_user,
                        x="engine_power_estimated",
                        y="predicted_price",
                        trendline="ols",
                        title="Влияние мощности на цену"
                    )
                    st.plotly_chart(fig, width='stretch')

                if "car_age" in numeric_cols_user:
                    st.subheader("Возраст автомобиля к Цене")
                    fig = px.scatter(
                        df_user,
                        x="car_age",
                        y="predicted_price",
                        trendline="ols",
                        title="Влияние возраста автомобиля на цену"
                    )
                    st.plotly_chart(fig, width='stretch')


if __name__ == "__main__":
    main()
