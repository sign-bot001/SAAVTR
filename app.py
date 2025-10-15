# app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from io import StringIO

# ----------------- CONFIG -----------------
st.set_page_config(page_title="Signal Alert AVTR", layout="wide")
tz_choice = "Asia/Seoul"

# ----------------- CSS -----------------
st.markdown("""
<style>
body {background-color: #050608;}
.stApp { color: #cfeef8; }
.title {font-family: 'Courier New', monospace; color: #7afcff; font-size:28px;}
h3, h4 { color:#b7f7ff; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">⚡ Signal Alert AVTR — Version Optimisée</div>', unsafe_allow_html=True)
st.write("Clique sur le bouton pour entraîner le modèle et lancer les prédictions.")

# ----------------- SAMPLE HISTORY -----------------
SAMPLE_HISTORY = [1.3,1.23,1.56,2.25,1.15,13.09,20.91,2.05,10.17,3.82,
                  1,1.46,1.4,1.73,1.17,1.00,26.60,8.6,1.27,1.46,
                  1.36,1.76,3.61,2.74,1.47,3.7,1.05]

@st.cache_data
def load_sample():
    tz = pytz.timezone(tz_choice)
    last_ts = datetime.now(tz)
    timestamps = [last_ts - timedelta(minutes=(len(SAMPLE_HISTORY)-1-i)) for i in range(len(SAMPLE_HISTORY))]
    df = pd.DataFrame({"timestamp":timestamps, "multiplier":SAMPLE_HISTORY})
    return df

# ----------------- CSV LOADER -----------------
def _looks_like_one_column_numbers(lines):
    if not lines: return False
    for ln in lines:
        ln = ln.strip()
        if not ln: 
            continue
        # autorise chiffres, espaces, +/-, , et .
        if not all(ch.isdigit() or ch in " .,-+" for ch in ln):
            return False
        # doit contenir au moins un chiffre
        if not any(ch.isdigit() for ch in ln):
            return False
    return True

@st.cache_data
def load_csv(uploaded_bytes: bytes):
    text = uploaded_bytes.decode("utf-8", errors="ignore")
    lines = [ln for ln in text.splitlines() if ln.strip()]

    # Cas 1: une seule colonne de nombres (accepte virgule décimale)
    if _looks_like_one_column_numbers(lines) and all(line.count(",") <= 1 for line in lines):
        vals = []
        for ln in lines:
            ln = ln.strip().replace(",", ".")
            try:
                vals.append(float(ln))
            except:
                pass
        tz = pytz.timezone(tz_choice)
        last_ts = datetime.now(tz)
        timestamps = [last_ts - timedelta(minutes=(len(vals)-1-i)) for i in range(len(vals))]
        df = pd.DataFrame({"timestamp": timestamps, "multiplier": vals})
        return df

    # Cas 2: CSV classique (timestamp,multiplier) ou équivalent
    text = text.replace(";", ",")
    df = pd.read_csv(StringIO(text))
    # détecte colonnes
    ts_col, mult_col = None, None
    for c in df.columns:
        cl = c.lower()
        if any(k in cl for k in ["time","date","timestamp"]):
            ts_col = c
        if any(k in cl for k in ["multiplier","mult","cote","value","rate"]):
            mult_col = c
    if ts_col is None and df.shape[1] >= 2:
        ts_col = df.columns[0]
    if mult_col is None:
        mult_col = df.columns[1] if df.shape[1] >= 2 else df.columns[0]

    df = df[[ts_col, mult_col]].copy()
    df.columns = ["timestamp", "multiplier"]
    # virgule décimale → point
    df["multiplier"] = df["multiplier"].astype(str).str.replace(",", ".")
    df["multiplier"] = pd.to_numeric(df["multiplier"], errors="coerce")
    # parse date et convertit en KST
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna()
    df["timestamp"] = df["timestamp"].dt.tz_convert(pytz.timezone(tz_choice))
    # ré-échantillonne à la minute (utilise la dernière valeur de chaque minute)
    df = df.set_index("timestamp").resample("1T").last().ffill().reset_index()
    return df

# ----------------- FEATURE ENGINEERING -----------------
def make_features(series, lags=10):
    X, y = [], []
    for i in range(lags, len(series)):
        X.append(series[i-lags:i])
        y.append(series[i])
    return np.array(X), np.array(y)

# ----------------- MODEL -----------------
@st.cache_resource
def train_model(series, lags=10, n_estimators=150):
    X, y = make_features(series, lags)
    if len(X) < 5:
        raise ValueError("Historique trop court pour entraîner un modèle.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    return model, scaler, mae, rmse

def predict_future(model, scaler, series, lags=10, steps=10):
    preds = []
    window = series[-lags:].tolist()
    for _ in range(steps):
        X = scaler.transform([window])
        p = model.predict(X)[0]
        preds.append(p)
        window.append(p)
        window = window[-lags:]
    return preds

# ----------------- SIDEBAR CONTROLS -----------------
st.sidebar.header("Données")
uploaded = st.sidebar.file_uploader("Charger un CSV (timestamp,multiplier) ou une liste de nombres", type=["csv"])
lags = st.sidebar.slider("Taille de fenêtre (lags)", 5, 40, 10)
steps = st.sidebar.slider("Minutes à prédire", 5, 120, 20)
n_estimators = st.sidebar.slider("Arbres (RandomForest)", 50, 300, 150)

# ----------------- DATA SOURCE -----------------
if uploaded is not None:
    df = load_csv(uploaded.getvalue())
    if df is None or df.empty:
        st.warning("CSV non lisible. Utilisation de l'historique intégré.")
        df = load_sample()
else:
    df = load_sample()

st.markdown("### Aperçu de l'historique (KST)")
st.dataframe(df.tail())

# ----------------- ACTION BUTTON -----------------
go = st.button("⚙️ Entraîner et prédire")

if go:
    try:
        st.info("Entraînement du modèle…")
        model, scaler, mae, rmse = train_model(df["multiplier"].values, lags=lags, n_estimators=n_estimators)
        st.success(f"Modèle entraîné (MAE={mae:.3f}, RMSE={rmse:.3f})")

        preds = predict_future(model, scaler, df["multiplier"].values, lags=lags, steps=steps)
        tz = pytz.timezone(tz_choice)
        future_times = [df["timestamp"].iloc[-1] + timedelta(minutes=i+1) for i in range(len(preds))]
        out = pd.DataFrame({"timestamp_kst":future_times, "predicted_multiplier":preds})
        st.markdown("### Prédictions")
        st.dataframe(out)

        st.line_chart(out.set_index("timestamp_kst"))
        st.download_button("Télécharger CSV des prédictions",
                           out.to_csv(index=False),
                           file_name="predictions_signal_alert_avtr.csv")
    except Exception as e:
        st.error(f"Impossible d'entraîner/prédire: {e}")
