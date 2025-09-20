# ============================================================
# STREAMLIT APP – Fusion des 2 codes
# Contient :
# - Croston
# - SES
# - SBA
# - Unified Final (Q*, sensibilité, politiques de commande, exports)
# ============================================================

import numpy as np
import pandas as pd
from scipy.stats import nbinom
import streamlit as st

# ============================================================
# CONFIGURATION STREAMLIT
# ============================================================
st.title("📊 Forecasting App – Croston, SES, SBA")
st.write("Upload your Excel file and run Grid Search + Final Evaluation")

uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])
SHEET_NAME = st.text_input("Excel sheet name", value="classification")

ALPHAS = [0.1, 0.2, 0.3, 0.4]
WINDOW_RATIOS = [0.6, 0.7, 0.8]
RECALC_INTERVALS = [5, 10, 20]

LEAD_TIME = 1
LEAD_TIME_SUPPLIER = 3
SERVICE_LEVEL = 0.95
NB_SIM = 1000
RNG_SEED = 42

SERVICE_LEVELS = [0.90, 0.92, 0.95, 0.98]

# ============================================================
# FONCTIONS UTILITAIRES
# ============================================================
def _disp(obj, n=None, title=None):
    if title:
        st.subheader(title)
    if isinstance(obj, pd.DataFrame):
        st.dataframe(obj.head(n) if n else obj)
    else:
        st.write(obj)

def load_matrix_timeseries(excel_file, sheet_name):
    df = pd.read_excel(excel_file, sheet_name=sheet_name)
    prod_col = df.columns[0]
    new_cols = [prod_col]
    for c in df.columns[1:]:
        try:
            new_cols.append(pd.to_datetime(c))
        except:
            new_cols.append(c)
    df.columns = new_cols
    return df, prod_col

def compute_metrics(df):
    if df.empty or "forecast_error" not in df:
        return np.nan, np.nan, np.nan, np.nan
    e = df["forecast_error"].astype(float)
    ME = e.mean()
    absME = e.abs().mean()
    MSE = (e**2).mean()
    RMSE = np.sqrt(MSE)
    return ME, absME, MSE, RMSE

# ============================================================
# MÉTHODES DE PRÉVISION
# ============================================================
def croston_forecast(x, alpha=0.2, variant="croston"):
    x = pd.Series(x).fillna(0.0).astype(float).values
    x = np.where(x < 0, 0.0, x)
    if (x == 0).all():
        return {"forecast_per_period": 0.0, "z_t": 0.0, "p_t": float("inf")}
    nz_idx = [i for i, v in enumerate(x) if v > 0]
    first = nz_idx[0]
    z = x[first]
    if len(nz_idx) >= 2:
        p = sum([j - i for i, j in zip(nz_idx[:-1], nz_idx[1:])]) / len(nz_idx)
    else:
        p = len(x) / len(nz_idx)
    psd = 0
    for t in range(first + 1, len(x)):
        psd += 1
        if x[t] > 0:
            I_t = psd
            z = alpha * x[t] + (1 - alpha) * z
            p = alpha * I_t + (1 - alpha) * p
            psd = 0
    f = z / p
    if variant == "sba":
        f *= (1 - alpha / 2.0)
    return {"forecast_per_period": float(f), "z_t": float(z), "p_t": float(p)}

def ses_forecast(x, alpha):
    x = pd.Series(x).fillna(0.0).astype(float).values
    if len(x) == 0:
        return {"forecast_per_period": 0.0, "z_t": 0.0, "p_t": 1.0}
    l = x[0]
    for t in range(1, len(x)):
        l = alpha * x[t] + (1 - alpha) * l
    return {"forecast_per_period": float(l), "z_t": float(l), "p_t": 1.0}

# ============================================================
# ROLLING + ROP
# ============================================================
def rolling_run(method, excel_file, product_code, sheet_name,
                alpha, window_ratio, interval,
                lead_time, lead_time_supplier,
                service_level, nb_sim, rng_seed):
    df, prod_col = load_matrix_timeseries(excel_file, sheet_name)
    row = df.loc[df[prod_col] == product_code]
    if row.empty:
        return pd.DataFrame()

    series = row.drop(columns=[prod_col]).T.squeeze()
    series.index = pd.to_datetime(series.index)
    series = series.sort_index()
    full_idx = pd.date_range(series.index.min(), series.index.max(), freq="D")
    daily = series.reindex(full_idx, fill_value=0.0).astype(float)
    values = daily.values

    split_index = int(len(values) * window_ratio)
    if split_index < 2:
        return pd.DataFrame()

    rng = np.random.default_rng(rng_seed)
    out_rows = []
    for i in range(split_index, len(values)):
        if (i - split_index) % interval == 0:
            train = values[:i]
            test_date = daily.index[i]
            real_demand = float(values[i])

            if method == "ses":
                fc = ses_forecast(train, alpha)
            elif method == "sba":
                fc = croston_forecast(train, alpha, "sba")
            else:
                fc = croston_forecast(train, alpha, "croston")

            f = float(fc["forecast_per_period"])
            sigma_period = float(pd.Series(train).std(ddof=1))

            # ROP usine
            X_Lt = lead_time * f
            sigma_Lt = sigma_period * np.sqrt(max(lead_time, 1e-9))
            var_u = sigma_Lt**2 if sigma_Lt**2 > X_Lt else X_Lt + 1e-5
            p_nb = max(min(X_Lt / var_u, 1 - 1e-12), 1e-12)
            r_nb = X_Lt**2 / (var_u - X_Lt) if var_u > X_Lt else 1e6
            ROP_u = float(np.percentile(nbinom.rvs(r_nb, p_nb, size=nb_sim, random_state=rng),
                                        100 * service_level))

            # ROP fournisseur
            totalL = lead_time + lead_time_supplier
            X_Lt_Lw = totalL * f
            sigma_Lt_Lw = sigma_period * np.sqrt(max(totalL, 1e-9))
            var_f = sigma_Lt_Lw**2 if sigma_Lt_Lw**2 > X_Lt_Lw else X_Lt_Lw + 1e-5
            p_nb_f = max(min(X_Lt_Lw / var_f, 1 - 1e-12), 1e-12)
            r_nb_f = X_Lt_Lw**2 / (var_f - X_Lt_Lw) if var_f > X_Lt_Lw else 1e6
            ROP_f = float(np.percentile(nbinom.rvs(r_nb_f, p_nb_f, size=nb_sim, random_state=rng),
                                        100 * service_level))

            out_rows.append({
                "date": test_date.date(),
                "real_demand": real_demand,
                "forecast_per_period": f,
                "forecast_error": real_demand - f,
                "ROP_usine": ROP_u,
                "ROP_fournisseur": ROP_f,
                "z_t": fc["z_t"],
                "p_t": fc["p_t"]
            })
    return pd.DataFrame(out_rows)

# ============================================================
# GRID SEARCH
# ============================================================
def grid_search(method, excel_file, codes):
    best_rows = []
    for code in codes:
        metrics_rows = []
        for a in ALPHAS:
            for w in WINDOW_RATIOS:
                for itv in RECALC_INTERVALS:
                    df_run = rolling_run(method, excel_file, code, SHEET_NAME,
                                         a, w, itv, LEAD_TIME, LEAD_TIME_SUPPLIER,
                                         SERVICE_LEVEL, NB_SIM, RNG_SEED)
                    ME, absME, MSE, RMSE = compute_metrics(df_run)
                    row = {"code": code, "alpha": a, "window_ratio": w,
                           "recalc_interval": itv, "ME": ME, "absME": absME,
                           "MSE": MSE, "RMSE": RMSE, "n_points": len(df_run)}
                    metrics_rows.append(row)
        df_metrics = pd.DataFrame(metrics_rows)
        if df_metrics.empty:
            continue
        best = df_metrics.loc[df_metrics["RMSE"].idxmin()]
        best_rows.append({
            "code": code,
            "best_alpha": best["alpha"],
            "best_window": best["window_ratio"],
            "best_interval": best["recalc_interval"],
            "best_RMSE": best["RMSE"]
        })
    return pd.DataFrame(best_rows)

# ============================================================
# MAIN STREAMLIT APP
# ============================================================
if uploaded_file is not None:
    df, prod_col = load_matrix_timeseries(uploaded_file, SHEET_NAME)
    CODES_PRODUITS = df[prod_col].dropna().unique().tolist()

    st.success("✅ File loaded successfully!")

    st.write("### Running Grid Search...")

    best_ses = grid_search("ses", uploaded_file, CODES_PRODUITS)
    best_cro = grid_search("croston", uploaded_file, CODES_PRODUITS)
    best_sba = grid_search("sba", uploaded_file, CODES_PRODUITS)

    st.write("#### Best Parameters (SES)")
    st.dataframe(best_ses)
    st.write("#### Best Parameters (Croston)")
    st.dataframe(best_cro)
    st.write("#### Best Parameters (SBA)")
    st.dataframe(best_sba)

    best_all = pd.concat([
        best_ses.assign(method="SES"),
        best_cro.assign(method="CROSTON"),
        best_sba.assign(method="SBA")
    ])
    best_final = best_all.loc[best_all.groupby("code")["best_RMSE"].idxmin()]
    _disp(best_final, title="Meilleure méthode par article")

    # Recalcul final multi-SL
    final_results = []
    for sl in SERVICE_LEVELS:
        st.write(f"### Recalcul final avec SL={sl}")
        for _, row in best_final.iterrows():
            code = row["code"]
            method = row["method"].lower()
            alpha = float(row["best_alpha"])
            window = float(row["best_window"])
            itv = int(row["best_interval"])
            df_run = rolling_run(method, uploaded_file, code, SHEET_NAME,
                                 alpha, window, itv, LEAD_TIME, LEAD_TIME_SUPPLIER,
                                 sl, NB_SIM, RNG_SEED)
            df_run["code"] = code
            df_run["method"] = method
            df_run["service_level"] = sl
            final_results.append(df_run)

    df_final = pd.concat(final_results, ignore_index=True)
    _disp(df_final.head(50), title="Aperçu résultats finaux")

    # Download button
    csv = df_final.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Final Results", csv, "final_results.csv", "text/csv")
else:
    st.info("👆 Please upload an Excel file to start.")
