# ============================================================
# STREAMLIT APP : Time Series Forecasting (ARIMA / SARIMA)
# - Upload CSV / Excel
# - Visualisation : série, STL decomposition, test ADF
# - Modèles : ARIMA / SARIMA (via SARIMAX)
# - Prévision + graphiques + export CSV
#
# Dépendances :
# pip install streamlit pandas numpy matplotlib statsmodels openpyxl chardet
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
Cette application permet de charger un fichier **CSV ou Excel** contenant une série temporelle,
de la visualiser (courbe + STL), de tester la stationnarité (ADF), puis d'entraîner un modèle
**ARIMA ou SARIMA** pour produire une prévision.
"""
)

# ------------------------------------------------------------
# FONCTIONS UTILES
# ------------------------------------------------------------

def read_file(uploaded_file) -> pd.DataFrame:
    """
    Lecture robuste CSV/XLSX :
    - XLSX/XLS : lecture via pandas.read_excel
    - CSV : tente autodétection séparateur + encodage, puis fallback séparateurs courants
    """
    filename = uploaded_file.name.lower()

    # ----- Excel -----
    if filename.endswith(".xlsx") or filename.endswith(".xls"):
        df = pd.read_excel(uploaded_file)
        return df

    # ----- CSV -----
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

    # Tentative autodétection séparateur
    try:
        df = pd.read_csv(io.BytesIO(raw), encoding=enc, sep=None, engine="python")
        return df
    except Exception:
        # Fallback sur séparateurs courants
        for sep in [",", ";", "\t", "|"]:
            try:
                df = pd.read_csv(io.BytesIO(raw), encoding=enc, sep=sep)
                return df
            except Exception:
                continue

    raise ValueError("Impossible de lire le CSV : vérifie séparateur/encodage.")


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie les noms de colonnes :
    - enlève espaces au début/fin
    - remplace espaces insécables (Excel) par espaces normaux
    - si doublons, rend les noms uniques (Date, Date.1, etc.)
    """
    df = df.copy()

    # Normaliser les noms
    df.columns = (
        df.columns.astype(str)
        .str.replace("\u00A0", " ", regex=False)  # NBSP -> espace
        .str.strip()
    )

    # Rendre uniques si doublons
    if df.columns.duplicated().any():
        # Méthode robuste : ajoute suffixes .1, .2...
        new_cols = []
        counts = {}
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
    """Convertit la colonne date en datetime, supprime invalides, trie."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df = df.sort_values(by=date_col)
    return df


def to_timeseries(df: pd.DataFrame, date_col: str, value_col: str, freq: str) -> pd.Series:
    """
    Transforme le DataFrame en Series temporelle régulière :
    - index = dates
    - resample à freq (moyenne)
    - interpolation des trous
    """
    df = df.copy()

    # Sécurité : si value_col n'existe pas
    if value_col not in df.columns:
        raise KeyError(f"Colonne valeur '{value_col}' introuvable. Colonnes : {list(df.columns)}")

    # Conversion numérique robuste :
    # - gère virgule décimale "12,3" -> "12.3"
    df[value_col] = df[value_col].astype(str).str.replace(",", ".", regex=False)
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[value_col])

    # Passage en index temporel
    df = df.set_index(date_col)

    # Resampling à une fréquence régulière (moyenne)
    ts = df[value_col].resample(freq).mean()

    # Interpolation des valeurs manquantes
    ts = ts.interpolate(method="time")

    return ts


def plot_series(ts: pd.Series, title: str):
    """Affiche une série temporelle."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ts.index, ts.values)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Valeur")
    ax.grid(True)
    st.pyplot(fig)


def adf_test(ts: pd.Series) -> dict:
    """
    Test ADF :
    H0 = non stationnaire.
    p-value < 0.05 => stationnaire (souvent).
    """
    ts_clean = ts.dropna()
    result = adfuller(ts_clean, autolag="AIC")
    return {
        "statistic": result[0],
        "pvalue": result[1],
        "usedlag": result[2],
        "nobs": result[3],
        "critical_values": result[4],
        "icbest": result[5],
    }


def stl_decompose(ts: pd.Series, period: int):
    """STL decomposition (robust=True pour mieux gérer les outliers)."""
    ts_clean = ts.dropna()
    stl = STL(ts_clean, period=period, robust=True)
    return stl.fit()


def fit_model(ts: pd.Series, model_type: str, order: tuple, seasonal_order: tuple):
    """
    Entraîne ARIMA/SARIMA via SARIMAX.
    - ARIMA => seasonal_order forcé à (0,0,0,0)
    """
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
    fitted = model.fit(disp=False)
    return fitted


def forecast_and_plot(ts: pd.Series, fitted_model, steps: int, show_last_n: int):
    """
    Forecast futur + 2 graphiques :
    1) Observé + prévision + intervalle de confiance
    2) Observé vs fitted (in-sample) sur N derniers points
    """
    ts_clean = ts.dropna()

    # Prévision future
    forecast_res = fitted_model.get_forecast(steps=steps)
    forecast_mean = forecast_res.predicted_mean
    forecast_ci = forecast_res.conf_int()

    # Graph 1 : Observé + forecast
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

    # Graph 2 : Observé vs fitted sur une fenêtre récente
    fitted_values = fitted_model.fittedvalues

    common_idx = ts_clean.index.intersection(fitted_values.index)
    ts_aligned = ts_clean.loc[common_idx]
    fv_aligned = fitted_values.loc[common_idx]

    ts_recent = ts_aligned.tail(show_last_n)
    fv_recent = fv_aligned.tail(show_last_n)

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(ts_recent.index, ts_recent.values, label="Observé (récent)")
    ax2.plot(fv_recent.index, fv_recent.values, label="Prédit in-sample (récent)")
    ax2.set_title(f"Comparaison Observé vs Prédit (sur les {show_last_n} derniers points)")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Valeur")
    ax2.grid(True)
    ax2.legend()
    st.pyplot(fig2)

    return forecast_mean, forecast_ci


# ------------------------------------------------------------
# SIDEBAR : UPLOAD + PARAMS
# ------------------------------------------------------------
st.sidebar.header("⚙️ Paramètres")
uploaded_file = st.sidebar.file_uploader("1) Upload CSV ou Excel", type=["csv", "xlsx", "xls"])

if uploaded_file is None:
    st.info("⬅️ Commence par uploader un fichier CSV ou Excel dans la barre latérale.")
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
# SELECTION COLONNES (avec blocage date == valeur)
# ------------------------------------------------------------
st.sidebar.subheader("2) Sélection des colonnes")
columns = df.columns.tolist()

# IMPORTANT : utiliser des keys pour éviter les effets mémoire Streamlit
date_col = st.sidebar.selectbox("Colonne Date/Temps", options=columns, key="date_col")

# Exclure la colonne date de la liste des valeurs
value_options = [c for c in columns if c != date_col]
if not value_options:
    st.error("Il faut au moins 2 colonnes : une date + une valeur numérique.")
    st.stop()

# Reset automatique si Streamlit avait gardé une ancienne valeur = date
if "value_col" in st.session_state and st.session_state["value_col"] == date_col:
    st.session_state["value_col"] = value_options[0]

value_col = st.sidebar.selectbox(
    "Colonne Valeur (numérique)",
    options=value_options,
    key="value_col"
)

# Sécurité ultime
if date_col == value_col:
    st.error("La colonne Date/Temps et la colonne Valeur doivent être différentes.")
    st.stop()

# ------------------------------------------------------------
# FREQUENCE
# ------------------------------------------------------------
st.sidebar.subheader("3) Fréquence de la série")
freq = st.sidebar.selectbox(
    "Resampling (pandas freq)",
    options=["D", "W", "M", "H"],
    index=0,
    help="D=jour, W=semaine, M=mois, H=heure",
)

# Pré-traitement
try:
    df2 = parse_datetime_column(df, date_col)
    ts = to_timeseries(df2, date_col, value_col, freq=freq)
except Exception as e:
    st.error(f"Erreur préparation série temporelle : {e}")
    st.stop()

# ------------------------------------------------------------
# VISUALISATION
# ------------------------------------------------------------
st.subheader("📌 Série temporelle originale")
plot_series(ts, "Série originale (après mise en forme + resampling)")

# ------------------------------------------------------------
# STL
# ------------------------------------------------------------
st.subheader("🧩 Décomposition STL (tendance, saisonnalité, résidu)")

default_period_map = {"D": 7, "W": 52, "M": 12, "H": 24}
default_period = default_period_map.get(freq, 7)

period = st.number_input(
    "Période saisonnière STL (ex: 7, 12, 24, 52...)",
    min_value=2,
    value=int(default_period),
    step=1,
)

try:
    stl_res = stl_decompose(ts, period=period)
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(stl_res.observed); axes[0].set_title("Observed"); axes[0].grid(True)
    axes[1].plot(stl_res.trend); axes[1].set_title("Trend (tendance)"); axes[1].grid(True)
    axes[2].plot(stl_res.seasonal); axes[2].set_title("Seasonal (saisonnalité)"); axes[2].grid(True)
    axes[3].plot(stl_res.resid); axes[3].set_title("Residual (résidu)"); axes[3].grid(True)
    plt.tight_layout()
    st.pyplot(fig)
except Exception as e:
    st.warning(f"Impossible de faire la STL decomposition : {e}")

# ------------------------------------------------------------
# ADF
# ------------------------------------------------------------
st.subheader("🧪 Test de stationnarité (ADF)")
try:
    adf_res = adf_test(ts)
    col1, col2 = st.columns(2)
