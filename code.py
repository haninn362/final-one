import streamlit as st
import numpy as np
import pandas as pd
import re
from scipy.stats import nbinom

# ==================================================
# SCRIPT FINAL UNIFIÉ - PFE HANIN (Streamlit version)
# ==================================================

# ---------- PARAMÈTRES ----------
EXCEL_PATH = "PFE  HANIN (1).xlsx"
PRODUCT_CODES = ["EM0400","EM1499","EM1091","EM1523","EM0392","EM1526"]

LEAD_TIME = 10
LEAD_TIME_SUPPLIER = 3
SERVICE_LEVEL = 0.95
NB_SIM = 1000
RNG_SEED = 42

ALPHAS = [0.1, 0.2, 0.3, 0.4]
WINDOW_RATIOS = [0.6, 0.7, 0.8]
RECALC_INTERVALS = [5, 10, 20]

SERVICE_LEVELS = [0.90, 0.92, 0.95, 0.98]   # Sensibilité

# ==================================================
# PARTIE 1 : Qr* et Qw* (Base Stock)
# ==================================================
def _find_product_sheet(excel_path: str, code: str) -> str:
    xls = pd.ExcelFile(excel_path)
    sheets = [s.strip().lower() for s in xls.sheet_names]
    targets = [
        f"time serie {code}".lower(),
        f"time series {code}".lower(),
        code.lower()
    ]
    for t in targets:
        if t in sheets:
            return xls.sheet_names[sheets.index(t)]
    for s in sheets:
        if code.lower() in s:
            return xls.sheet_names[sheets.index(s)]
    raise ValueError(f"[Sheet] Onglet pour '{code}' introuvable.")

def compute_qstars(file_path, product_codes):
    df_conso = pd.read_excel(file_path, sheet_name="consommation depots externe")
    df_conso = df_conso.groupby('Code Produit')['Quantite STIAL'].sum()
    qr_map, qw_map = {}, {}
    for code in product_codes:
        sheet = _find_product_sheet(file_path, code)
        df = pd.read_excel(file_path, sheet_name=sheet)
        C_r = df['Cr : cout stockage/article '].iloc[0]
        C_w = df['Cw : cout stockage\nchez F'].iloc[0]
        A_w = df['Aw : cout de\nlancement chez U'].iloc[0]
        A_r = df['Ar : cout de \nlancement chez F'].iloc[0]
        n = (A_w * C_r) / (A_r * C_w)
        n = 1 if n < 1 else round(n)
        n1, n2 = int(n), int(n) + 1
        F_n1 = (A_r + A_w / n1) * (n1 * C_w + C_r)
        F_n2 = (A_r + A_w / n2) * (n2 * C_w + C_r)
        n_star = n1 if F_n1 <= F_n2 else n2
        D = df_conso.get(code, 0)
        tau = 1
        Qr_star = ((2 * (A_r + A_w / n_star) * D) / (n_star * C_w + C_r * tau)) ** 0.5
        Qw_star = n_star * Qr_star
        qr_map[code] = round(Qr_star, 2)
        qw_map[code] = round(Qw_star, 2)
    return qr_map, qw_map

# ==================================================
# PARTIE 2 : Prévisions (SES / Croston / SBA)
# ==================================================
def croston_or_sba_forecast(x, alpha=0.2, variant="croston"):
    x = pd.Series(x).fillna(0.0).astype(float).values
    if (x == 0).all():
        return 0.0
    nz_idx = [i for i, v in enumerate(x) if v > 0]
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

def ses_forecast(x, alpha=0.2):
    x = pd.Series(x).fillna(0.0).astype(float).values
    if len(x) == 0:
        return 0.0
    l = x[0]
    for t in range(1, len(x)):
        l = alpha * x[t] + (1 - alpha) * l
    return l

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

def rolling_forecast_with_metrics(excel_path, product_code, sheet_name,
                                  alpha, window_ratio, interval, method):
    df, prod_col = load_matrix_timeseries(excel_path, sheet_name)
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

def grid_search_all_methods():
    candidates = []
    for code in PRODUCT_CODES:
        for method in ["ses","croston","sba"]:
            metrics_rows = []
            for a in ALPHAS:
                for w in WINDOW_RATIOS:
                    for itv in RECALC_INTERVALS:
                        df_run = rolling_forecast_with_metrics(
                            EXCEL_PATH, code, "classification",
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
# PARTIE 3 : Simulation finale avec ROP
# ==================================================
def _interval_sum_next_days(daily: pd.Series, start_idx: int, interval: int) -> float:
    s, e = start_idx + 1, start_idx + 1 + interval
    return float(pd.Series(daily).iloc[s:e].sum())

def simulate_orders(best_per_code, qr_map, service_level=SERVICE_LEVEL):
    results = []
    rng = np.random.default_rng(RNG_SEED)
    for _, row in best_per_code.iterrows():
        code = row["code"]
        method = row["method"]
        alpha = row["alpha"]
        window_ratio = row["window_ratio"]
        interval = int(row["recalc_interval"])
        sheet = _find_product_sheet(EXCEL_PATH, code)
        df = pd.read_excel(EXCEL_PATH, sheet_name=sheet)
        dates = pd.to_datetime(df.iloc[:,0], errors="coerce")
        stock_col = pd.to_numeric(df.iloc[:,1], errors="coerce").astype(float)
        cons_col = pd.to_numeric(df.iloc[:,2], errors="coerce").fillna(0.0).astype(float)
        ts_cons = pd.DataFrame({"d":dates,"q":cons_col}).dropna().sort_values("d").set_index("d")["q"]
        ts_stock = pd.DataFrame({"d":dates,"s":stock_col}).dropna().sort_values("d").set_index("d")["s"]
        full_idx = pd.date_range(ts_cons.index.min(), ts_cons.index.max(), freq="D")
        cons_daily = ts_cons.reindex(full_idx, fill_value=0.0)
        stock_daily = ts_stock.reindex(full_idx).ffill().fillna(0.0)
        vals = cons_daily.values
        split_index = int(len(vals) * window_ratio)
        if split_index < 2:
            continue
        stock_after_interval = 0.0
        for i in range(split_index, len(vals)):
            if (i - split_index) % interval == 0:
                train = vals[:i]
                if method == "ses":
                    f = ses_forecast(train, alpha)
                elif method == "croston":
                    f = croston_or_sba_forecast(train, alpha, "croston")
                else:
                    f = croston_or_sba_forecast(train, alpha, "sba")
                sigma_period = float(pd.Series(train).std(ddof=1)) if i > 1 else 0.0
                sigma_period = sigma_period if np.isfinite(sigma_period) else 0.0
                # ROP usine
                X_Lt = LEAD_TIME * f
                sigma_Lt = sigma_period * np.sqrt(max(LEAD_TIME, 1e-9))
                var_u = sigma_Lt**2 if sigma_Lt**2 > X_Lt else X_Lt+1e-5
                p_nb = min(max(X_Lt/var_u, 1e-12),1-1e-12)
                r_nb = X_Lt**2/(var_u - X_Lt) if var_u > X_Lt else 1e6
                ROP_u = float(np.percentile(nbinom.rvs(r_nb, p_nb, size=NB_SIM, random_state=rng), 100*service_level))
                # ROP fournisseur
                totalL = LEAD_TIME + LEAD_TIME_SUPPLIER
                X_Lt_Lw = totalL * f
                sigma_Lt_Lw = sigma_period * np.sqrt(max(totalL, 1e-9))
                var_f = sigma_Lt_Lw**2 if sigma_Lt_Lw**2 > X_Lt_Lw else X_Lt_Lw+1e-5
                p_nb_f = min(max(X_Lt_Lw/var_f, 1e-12),1-1e-12)
                r_nb_f = X_Lt_Lw**2/(var_f - X_Lt_Lw) if var_f > X_Lt_Lw else 1e6
                ROP_f = float(np.percentile(nbinom.rvs(r_nb_f, p_nb_f, size=NB_SIM, random_state=rng), 100*service_level))
                # Demande réelle et stocks
                real_demand = _interval_sum_next_days(cons_daily, i, interval)
                stock_on_hand_running = _interval_sum_next_days(stock_daily, i, interval)
                stock_after_interval = stock_after_interval + stock_on_hand_running - real_demand
                # Politique de commande
                if stock_after_interval >= real_demand * LEAD_TIME:
                    order_policy = "no_order"
                else:
                    order_policy = f"order_Qr*_{qr_map[code]}"
                    stock_after_interval += qr_map[code]
                stock_status = "rupture" if real_demand > ROP_u else "holding"
                results.append({
                    "date": cons_daily.index[i].date(),
                    "code": code,
                    "interval": interval,
                    "real_demand": real_demand,
                    "stock_on_hand_running": stock_on_hand_running,
                    "stock_after_interval": stock_after_interval,
                    "order_policy": order_policy,
                    "Qr_star": qr_map[code],
                    "reorder_point_usine": ROP_u,
                    "reorder_point_fournisseur": ROP_f,
                    "stock_status": stock_status,
                    "service_level": service_level
                })
        # affichage par produit
        st.write(f"=== Final Results for {code} ({method.upper()}) SL={service_level} ===")
        df_display = pd.DataFrame(results)[pd.DataFrame(results)["code"] == code]
        st.dataframe(df_display.head(20))
    return pd.DataFrame(results)

# ==================================================
# PARTIE 4 : Analyse de sensibilité
# ==================================================
def run_sensitivity(best_per_code, qr_map):
    all_results = []
    for sl in SERVICE_LEVELS:
        st.write(f"🔎 Sensibilité pour Service Level = {sl*100:.0f}%")
        df_run = simulate_orders(best_per_code, qr_map, service_level=sl)
        if not df_run.empty:
            summary = df_run.groupby("code").agg(
                ROP_u_moy=("reorder_point_usine","mean"),
                ROP_f_moy=("reorder_point_fournisseur","mean"),
                holding_pct=("stock_status", lambda s: (s=="holding").mean()*100),
                rupture_pct=("stock_status", lambda s: (s=="rupture").mean()*100),
                Qr_star=("Qr_star","first")
            ).reset_index()
            st.write(f"=== Résumé SL={sl:.2f} ===")
            st.dataframe(summary)
            all_results.append(df_run)
    return pd.concat(all_results, ignore_index=True)

# ==================================================
# MAIN (Streamlit)
# ==================================================
def main():
    st.title("📊 PFE HANIN – Stock & Prévisions")

    if st.button("▶ Run Full Simulation"):
        # Qr* et Qw*
        qr_map, qw_map = compute_qstars(EXCEL_PATH, PRODUCT_CODES)
        st.subheader("Qr* et Qw*")
        st.write("Qr* :", qr_map)
        st.write("Qw* :", qw_map)

        # Grid search & sélection meilleure méthode
        all_candidates = grid_search_all_methods()
        idx = all_candidates.groupby("code")["RMSE"].idxmin()
        best_per_code = all_candidates.loc[idx].reset_index(drop=True)

        st.subheader("✅ Meilleure méthode par article (critère: RMSE)")
        st.dataframe(best_per_code)

        # Résumé
        st.subheader("Résumé")
        for _, r in best_per_code.iterrows():
            st.write(f"• {r['code']}: {r['method'].upper()} | RMSE={r['RMSE']:.4g} | "
                     f"α={r['alpha']}, win={r['window_ratio']}, itv={int(r['recalc_interval'])}")

        # Simulation finale SL=0.95
        st.subheader("Simulation finale (SL=0.95)")
        final_results = simulate_orders(best_per_code, qr_map, service_level=SERVICE_LEVEL)
        st.dataframe(final_results.head(50))

        # 🔥 Analyse de sensibilité multi-SL
        st.subheader("Analyse de sensibilité")
        sensitivity_results = run_sensitivity(best_per_code, qr_map)
        st.dataframe(sensitivity_results.head(50))

if __name__ == "__main__":
    main()
