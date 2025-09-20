# ==================================================
# STREAMLIT APP - PFE HANIN
# Base Stock + Prévisions (SES / Croston / SBA)
# Sélection meilleure méthode + Simulation commandes
# + Analyse de sensibilité
# ==================================================

import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import nbinom

# ---------- PARAMÈTRES ----------
EXCEL_PATH = "PFE  HANIN (1).xlsx"
PRODUCT_CODES = ["EM0400","EM1499","EM1091","EM1523","EM0392","EM1526"]

LEAD_TIME = 10
LEAD_TIME_SUPPLIER = 3
SERVICE_LEVEL = 0.95
NB_SIM = 1000

ALPHAS = [0.1, 0.2, 0.3, 0.4]
WINDOW_RATIOS = [0.6, 0.7, 0.8]
RECALC_INTERVALS = [5, 10, 20]
SERVICE_LEVELS = [0.90, 0.92, 0.95, 0.98]

# ==================================================
# Caching Excel loading
# ==================================================
@st.cache_data
def load_excel(path):
    return pd.ExcelFile(path)

@st.cache_data
def load_matrix_timeseries(excel_path, sheet_name):
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    prod_col = df.columns[0]
    new_cols = [prod_col]
    for c in df.columns[1:]:
        try:
            new_cols.append(pd.to_datetime(c))
        except:
            new_cols.append(c)
    df.columns = new_cols
    return df, prod_col

xls = load_excel(EXCEL_PATH)
df_classification, prod_col = load_matrix_timeseries(EXCEL_PATH, "classification")

# ==================================================
# Forecasting methods
# ==================================================
def ses_forecast(x, alpha=0.2):
    if len(x) == 0:
        return 0.0
    l = x[0]
    for t in range(1, len(x)):
        l = alpha * x[t] + (1 - alpha) * l
    return l

def croston_or_sba_forecast(x, alpha=0.2, variant="croston"):
    x = np.array(x, dtype=float)
    if (x == 0).all():
        return 0.0
    nz_idx = np.where(x > 0)[0]
    if len(nz_idx) == 0:
        return 0.0
    z, p = x[nz_idx[0]], len(x)/len(nz_idx)
    psd = 0
    for t in range(nz_idx[0]+1, len(x)):
        psd += 1
        if x[t] > 0:
            I_t = psd
            z = alpha * x[t] + (1-alpha) * z
            p = alpha * I_t + (1-alpha) * p
            psd = 0
    f = z / p
    if variant == "sba":
        f *= (1 - alpha/2.0)
    return f

def rolling_forecast_with_metrics(df, prod_col, product_code,
                                  alpha, window_ratio, interval, method):
    row = df.loc[df[prod_col] == product_code]
    if row.empty:
        return pd.DataFrame()
    series = row.drop(columns=[prod_col]).T.squeeze()
    series.index = pd.to_datetime(series.index, errors="coerce")
    series = series.sort_index()
    full_idx = pd.date_range(series.index.min(), series.index.max(), freq="D")
    daily = series.reindex(full_idx, fill_value=0.0).astype(float)
    values = daily.values
    split_index = int(len(values) * window_ratio)
    if split_index < 2:
        return pd.DataFrame()
    out_rows = []
    for i in range(split_index, len(values)):
        if (i - split_index) % interval == 0:
            train = values[:i]
            real_demand = float(values[i])
            if method == "ses":
                f = ses_forecast(train, alpha)
            elif method == "croston":
                f = croston_or_sba_forecast(train, alpha, "croston")
            elif method == "sba":
                f = croston_or_sba_forecast(train, alpha, "sba")
            else:
                f = 0.0
            out_rows.append({
                "real_demand": real_demand,
                "forecast": f,
                "error": real_demand - f
            })
    return pd.DataFrame(out_rows)

def compute_metrics(df_run):
    if df_run.empty:
        return np.nan, np.nan, np.nan
    e = df_run["error"].astype(float)
    MSE = (e**2).mean()
    RMSE = np.sqrt(MSE)
    absME = e.abs().mean()
    return absME, MSE, RMSE

@st.cache_data
def grid_search_all_methods(df, prod_col):
    candidates = []
    for code in PRODUCT_CODES:
        metrics_rows = []
        for method in ["ses","croston","sba"]:
            for a in ALPHAS:
                for w in WINDOW_RATIOS:
                    for itv in RECALC_INTERVALS:
                        df_run = rolling_forecast_with_metrics(
                            df, prod_col, code,
                            a, w, itv, method
                        )
                        absME, MSE, RMSE = compute_metrics(df_run)
                        metrics_rows.append({
                            "code": code, "method": method,
                            "alpha": a, "window_ratio": w,
                            "recalc_interval": itv,
                            "absME": absME, "MSE": MSE, "RMSE": RMSE
                        })
        df_metrics = pd.DataFrame(metrics_rows)
        candidates.append(df_metrics)
    return pd.concat(candidates, ignore_index=True)

# ==================================================
# MAIN STREAMLIT APP
# ==================================================
st.title("📊 PFE HANIN - Base Stock + Prévisions")

if st.button("Run Grid Search"):
    all_candidates = grid_search_all_methods(df_classification, prod_col)
    idx = all_candidates.groupby("code")["RMSE"].idxmin()
    best_per_code = all_candidates.loc[idx].reset_index(drop=True)

    st.subheader("✅ Meilleure méthode par article (critère: RMSE)")
    st.dataframe(best_per_code)

    st.subheader("--- Résumé ---")
    for _, r in best_per_code.iterrows():
        st.write(f"• {r['code']}: {r['method'].upper()} | "
                 f"RMSE={r['RMSE']:.4g} | "
                 f"α={r['alpha']}, win={r['window_ratio']}, "
                 f"itv={int(r['recalc_interval'])}")
