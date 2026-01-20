# ============================================================
# STREAMLIT APP : Time Series Forecasting (ARIMA / SARIMA)
# ============================================================

import io
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ------------------------------------------------------------
# CONFIG STREAMLIT
# ------------------------------------------------------------
st.set_page_config(page_title="Projet Série Temporelle (ARIMA/SARIMA)", layout="wide")

st.title("📈 Application Streamlit - Série temporelle (ARIMA / SARIMA)")
st.write(
    """
Chargement d'un **CSV/XLSX**, visualisation de la série + STL, teste de la stationnarité (ADF),
entraînement d'un modèle **ARIMA** ou **SARIMA**, puis prévision.
"""
)
st.write(
    """
Par Julient GOUAULT
"""
)

# ------------------------------------------------------------
# FONCTIONS
# ------------------------------------------------------------

def read_file(uploaded_file) -> pd.DataFrame:
    """Lecture robuste CSV/XLSX."""
    filename = uploaded_file.name.lower()

    if filename.endswith(".xlsx") or filename.endswith(".xls"):
        return pd.read_excel(uploaded_file)

    if not filename.endswith(".csv"):
        raise ValueError("Format non supporté : merci d'uploader un CSV ou un Excel.")

    raw = uploaded_file.getvalue()

    # Détection encodage (utile pour latin-1/cp1252)
    enc = "utf-8"
    try:
        import chardet
        detected = chardet.detect(raw)
        if detected and detected.get("encoding"):
            enc = detected["encoding"]
    except Exception:
        pass

    # Autodétection séparateur
    try:
        return pd.read_csv(io.BytesIO(raw), encoding=enc, sep=None, engine="python")
    except Exception:
        # Fallback séparateurs courants
        for sep in [",", ";", "\t", "|"]:
            try:
                return pd.read_csv(io.BytesIO(raw), encoding=enc, sep=sep)
            except Exception:
                continue

    raise ValueError("Impossible de lire le CSV (séparateur/encodage).")


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie noms de colonnes + rend uniques si doublons."""
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.replace("\u00A0", " ", regex=False)  # NBSP -> espace
        .str.strip()
    )

    if df.columns.duplicated().any():
        new_cols, counts = [], {}
        for c in df.columns:
            if c not in counts:
                counts[c] = 0
                new_cols.append(c)
            else:
                counts[c] += 1
                new_cols.append(f"{c}.{counts[c]}")
        df.columns = new_cols

    return df


def parse_datetime_column(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Convertit la colonne date en datetime, drop invalides, trie."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df = df.sort_values(by=date_col)
    return df


def to_timeseries(df: pd.DataFrame, date_col: str, value_col: str, freq: str) -> pd.Series:
    """Transforme en série temporelle régulière + interpolation."""
    df = df.copy()

    if value_col not in df.columns:
        raise KeyError(f"Colonne valeur '{value_col}' introuvable. Colonnes : {list(df.columns)}")

    # conversion numérique robuste (gère virgules)
    df[value_col] = df[value_col].astype(str).str.replace(",", ".", regex=False)
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[value_col])

    df = df.set_index(date_col)

    ts = df[value_col].resample(freq).mean()
    ts = ts.interpolate(method="time")
    return ts


def plot_series(ts: pd.Series, title: str):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ts.index, ts.values)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Valeur")
    ax.grid(True)
    st.pyplot(fig)


def adf_test(ts: pd.Series) -> dict:
    """Test ADF : H0 = non stationnaire."""
    ts_clean = ts.dropna()
    stat, pval, usedlag, nobs, crit, icbest = adfuller(ts_clean, autolag="AIC")
    return {
        "statistic": stat,
        "pvalue": pval,
        "usedlag": usedlag,
        "nobs": nobs,
        "critical_values": crit,
        "icbest": icbest,
    }


def stl_decompose(ts: pd.Series, period: int):
    ts_clean = ts.dropna()
    return STL(ts_clean, period=period, robust=True).fit()


def fit_model(ts: pd.Series, model_type: str, order: tuple, seasonal_order: tuple):
    """Entraîne ARIMA/SARIMA via SARIMAX."""
    ts_clean = ts.dropna()

    if model_type == "ARIMA":
        seasonal_order = (0, 0, 0, 0)

    model = SARIMAX(
        ts_clean,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False)


def forecast_and_plot(ts: pd.Series, fitted_model, steps: int, show_last_n: int):
    """Prévision future + graphs + retourne forecast + IC."""
    ts_clean = ts.dropna()

    forecast_res = fitted_model.get_forecast(steps=steps)
    forecast_mean = forecast_res.predicted_mean
    forecast_ci = forecast_res.conf_int()

    # Graph 1 : Observé + forecast + IC
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(ts_clean.index, ts_clean.values, label="Observé")
    ax1.plot(forecast_mean.index, forecast_mean.values, label="Prévision")
    ax1.fill_between(
        forecast_ci.index,
        forecast_ci.iloc[:, 0],
        forecast_ci.iloc[:, 1],
        alpha=0.2,
        label="IC",
    )
    ax1.set_title("Prévision future")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Valeur")
    ax1.grid(True)
    ax1.legend()
    st.pyplot(fig1)

    # Graph 2 : Observé vs fitted (récent)
    fitted_values = fitted_model.fittedvalues
    common_idx = ts_clean.index.intersection(fitted_values.index)

    ts_recent = ts_clean.loc[common_idx].tail(show_last_n)
    fv_recent = fitted_values.loc[common_idx].tail(show_last_n)

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(ts_recent.index, ts_recent.values, label="Observé (récent)")
    ax2.plot(fv_recent.index, fv_recent.values, label="Prédit in-sample (récent)")
    ax2.set_title(f"Comparaison Observé vs Prédit (sur {show_last_n} derniers points)")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Valeur")
    ax2.grid(True)
    ax2.legend()
    st.pyplot(fig2)

    return forecast_mean, forecast_ci


# ------------------------------------------------------------
# SIDEBAR : UPLOAD
# ------------------------------------------------------------
st.sidebar.header("⚙️ Paramètres")
uploaded_file = st.sidebar.file_uploader("1) Upload CSV ou Excel", type=["csv", "xlsx", "xls"])

if uploaded_file is None:
    st.info("⬅️ Upload un fichier pour commencer.")
    st.stop()

# Lecture + nettoyage colonnes
try:
    df = read_file(uploaded_file)
    df = clean_columns(df)
except Exception as e:
    st.error(f"Erreur lecture fichier : {e}")
    st.stop()

st.subheader("Aperçu des données")
st.dataframe(df.head(20))

# ------------------------------------------------------------
# SELECTION COLONNES (anti-date 2 fois)
# ------------------------------------------------------------
st.sidebar.subheader("2) Sélection des colonnes")
columns = df.columns.tolist()

date_col = st.sidebar.selectbox("Colonne Date/Temps", options=columns, key="date_col")

value_options = [c for c in columns if c != date_col]
if not value_options:
    st.error("Il faut au moins 2 colonnes : une date + une valeur.")
    st.stop()

# Reset si Streamlit avait gardé value_col = date_col
if "value_col" in st.session_state and st.session_state["value_col"] == date_col:
    st.session_state["value_col"] = value_options[0]

value_col = st.sidebar.selectbox("Colonne Valeur (numérique)", options=value_options, key="value_col")

if date_col == value_col:
    st.error("La colonne Date/Temps et la colonne Valeur doivent être différentes.")
    st.stop()

# ------------------------------------------------------------
# FREQUENCE + TS
# ------------------------------------------------------------
st.sidebar.subheader("3) Fréquence de la série")
freq = st.sidebar.selectbox("Resampling", options=["D", "W", "M", "H"], index=0, key="freq")

try:
    df2 = parse_datetime_column(df, date_col)
    ts = to_timeseries(df2, date_col, value_col, freq=freq)
except Exception as e:
    st.error(f"Erreur préparation série temporelle : {e}")
    st.stop()

st.subheader("📌 Série temporelle")
plot_series(ts, "Série originale (après resampling)")

# ------------------------------------------------------------
# STL
# ------------------------------------------------------------
st.subheader("🧩 Décomposition STL")

default_period_map = {"D": 7, "W": 52, "M": 12, "H": 24}
default_period = default_period_map.get(freq, 7)

period = st.number_input(
    "Période saisonnière STL",
    min_value=2,
    value=int(default_period),
    step=1
)

try:
    stl_res = stl_decompose(ts, period=period)

    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(stl_res.observed); axes[0].set_title("Observed"); axes[0].grid(True)
    axes[1].plot(stl_res.trend); axes[1].set_title("Trend"); axes[1].grid(True)
    axes[2].plot(stl_res.seasonal); axes[2].set_title("Seasonal"); axes[2].grid(True)
    axes[3].plot(stl_res.resid); axes[3].set_title("Residual"); axes[3].grid(True)
    plt.tight_layout()
    st.pyplot(fig)
except Exception as e:
    st.warning(f"STL impossible : {e}")

# ------------------------------------------------------------
# ADF
# ------------------------------------------------------------
st.subheader("🧪 Test ADF (stationnarité)")

try:
    adf_res = adf_test(ts)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("ADF Statistic", f"{adf_res['statistic']:.4f}")
        st.metric("p-value", f"{adf_res['pvalue']:.6f}")
    with col2:
        st.write("Valeurs critiques :")
        st.json(adf_res["critical_values"])

    if adf_res["pvalue"] < 0.05:
        st.success("✅ Série probablement stationnaire (p < 0.05).")
    else:
        st.warning("⚠️ Série probablement NON stationnaire (p ≥ 0.05). Ajuste d (et D si SARIMA).")

except Exception as e:
    st.error(f"Erreur test ADF : {e}")

# ------------------------------------------------------------
# MODELING : ARIMA / SARIMA + PREDICTIONS
# ------------------------------------------------------------
st.subheader("🤖 Modèle ARIMA / SARIMA + Prévisions")

st.sidebar.subheader("4) Choix du modèle")
model_type = st.sidebar.selectbox("Modèle", options=["ARIMA", "SARIMA"], key="model_type")

st.sidebar.subheader("5) Paramètres ARIMA (p,d,q)")
p = st.sidebar.number_input("p", min_value=0, max_value=10, value=1, step=1, key="p")
d = st.sidebar.number_input("d", min_value=0, max_value=3, value=1, step=1, key="d")
q = st.sidebar.number_input("q", min_value=0, max_value=10, value=1, step=1, key="q")
order = (int(p), int(d), int(q))

if model_type == "SARIMA":
    st.sidebar.subheader("Paramètres saisonniers (P,D,Q,s)")
    P = st.sidebar.number_input("P", min_value=0, max_value=10, value=1, step=1, key="P")
    D = st.sidebar.number_input("D", min_value=0, max_value=3, value=1, step=1, key="D")
    Q = st.sidebar.number_input("Q", min_value=0, max_value=10, value=1, step=1, key="Q")
    s = st.sidebar.number_input("s", min_value=1, value=int(default_period), step=1, key="s")
    seasonal_order = (int(P), int(D), int(Q), int(s))
else:
    seasonal_order = (0, 0, 0, 0)

st.sidebar.subheader("6) Prévision")
steps = st.sidebar.number_input("Nombre de pas dans le futur", min_value=1, max_value=500, value=30, step=1, key="steps")
show_last_n = st.sidebar.number_input("Zoom sur N derniers points", min_value=10, max_value=500, value=60, step=10, key="show_last_n")

run = st.button("🚀 Entraîner et prédire", key="run_btn")

if run:
    with st.spinner("Entraînement du modèle..."):
        try:
            fitted = fit_model(ts, model_type=model_type, order=order, seasonal_order=seasonal_order)
            st.success("✅ Modèle entraîné !")

            st.subheader("Résumé du modèle")
            st.text(fitted.summary())

            st.subheader("📊 Prévisions")
            forecast_mean, forecast_ci = forecast_and_plot(
                ts, fitted_model=fitted, steps=int(steps), show_last_n=int(show_last_n)
            )

            st.subheader("Table des prévisions")
            out_df = pd.DataFrame(
                {
                    "forecast": forecast_mean,
                    "lower_ci": forecast_ci.iloc[:, 0],
                    "upper_ci": forecast_ci.iloc[:, 1],
                }
            )
            st.dataframe(out_df)

            csv_data = out_df.to_csv(index=True).encode("utf-8")
            st.download_button(
                label="⬇️ Télécharger les prévisions (CSV)",
                data=csv_data,
                file_name="forecast.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error(f"❌ Erreur entraînement/prévision : {e}")
