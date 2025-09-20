import streamlit as st
import numpy as np
import pandas as pd
import re
from scipy.stats import nbinom

# ==============================================================
# STREAMLIT APP
# ==============================================================

st.set_page_config(page_title="Forecasting & ROP App", layout="wide")
st.title("📊 Forecasting, ROP & Sensitivity Analysis")

# ==============================================================
# FILE UPLOAD
# ==============================================================

uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])
if uploaded_file is None:
    st.warning("Please upload an Excel file to continue.")
    st.stop()

EXCEL_PATH = uploaded_file

# ==============================================================
# GLOBAL PARAMETERS
# ==============================================================

PRODUCT_CODES = ["EM0400", "EM1499", "EM1091", "EM1523", "EM0392", "EM1526"]

# Grid search params
ALPHAS = [0.1, 0.2, 0.3, 0.4]
WINDOW_RATIOS = [0.6, 0.7, 0.8]
RECALC_INTERVALS = [5, 10, 20]

# Supply / ROP
LEAD_TIME = 10
LEAD_TIME_SUPPLIER = 3
NB_SIM = 1000
RNG_SEED = 42
SERVICE_LEVELS = [0.90, 0.92, 0.95, 0.98]

rng = np.random.default_rng(RNG_SEED)

# ==============================================================
# PART 0 : Qr* and Qw*
# ==============================================================

def compute_qstars(excel_path: str, product_codes: list):
    df_conso = pd.read_excel(excel_path, sheet_name="consommation depots externe")
    df_conso = df_conso.groupby('Code Produit')['Quantite STIAL'].sum()

    qr_map, qw_map = {}, {}
    for code in product_codes:
        sheet = _find_product_sheet(excel_path, code)
        df = pd.read_excel(excel_path, sheet_name=sheet)

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
        Q_r_star = ((2 * (A_r + A_w / n_star) * D) / (n_star * C_w + C_r * tau)) ** 0.5
        Q_w_star = n_star * Q_r_star

        qr_map[code] = round(Q_r_star, 2)
        qw_map[code] = round(Q_w_star, 2)

    return qr_map, qw_map

# ==============================================================
# PART 1 : Helpers
# ==============================================================

def _find_product_sheet(excel_path: str, code: str) -> str:
    xls = pd.ExcelFile(excel_path)
    sheets = xls.sheet_names
    target = f"time serie {code}"
    if target in sheets:
        return target
    patt = re.compile(r"time\s*ser(i|ie)s?\s*", re.IGNORECASE)
    cand = [s for s in sheets if patt.search(s) and code.lower() in s.lower()]
    if cand:
        return sorted(cand, key=len, reverse=True)[0]
    raise ValueError(f"[Sheet] Onglet pour '{code}' introuvable.")

def _daily_consumption_and_stock(excel_path: str, sheet_name: str):
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    date_col, stock_col, cons_col = df.columns[0], df.columns[1], df.columns[2]

    dates = pd.to_datetime(df[date_col], errors="coerce")
    cons = pd.to_numeric(df[cons_col], errors="coerce").fillna(0.0).astype(float)
    stock = pd.to_numeric(df[stock_col], errors="coerce").astype(float)

    ts_cons = pd.DataFrame({"d": dates, "q": cons}).dropna().sort_values("d").set_index("d")["q"]
    ts_stock = pd.DataFrame({"d": dates, "s": stock}).dropna().sort_values("d").set_index("d")["s"]

    min_date, max_date = ts_cons.index.min(), ts_cons.index.max()
    full_idx = pd.date_range(min_date, max_date, freq="D")
    cons_daily = ts_cons.reindex(full_idx, fill_value=0.0)
    stock_daily = ts_stock.reindex(full_idx).ffill().fillna(0.0)

    return cons_daily, stock_daily

def _interval_sum_next_days(daily: pd.Series, start_idx: int, interval: int) -> float:
    s, e = start_idx + 1, start_idx + 1 + interval
    return float(pd.Series(daily).iloc[s:e].sum())

# ==============================================================
# PART 2 : Forecast methods
# ==============================================================

def _croston_or_sba(x, alpha: float, variant: str = "sba"):
    x = pd.Series(x).fillna(0.0).astype(float).values
    x = np.where(x < 0, 0.0, x)
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

def _ses(x, alpha: float):
    x = pd.Series(x).fillna(0.0).astype(float).values
    if len(x) == 0:
        return 0.0
    l = x[0]
    for t in range(1, len(x)):
        l = alpha * x[t] + (1 - alpha) * l
    return l

# ==============================================================
# PART 3 : Rolling
# ==============================================================

def rolling_with_new_logic(
    excel_path, product_code, alpha, window_ratio, interval,
    lead_time, lead_time_supplier, service_level, nb_sim, rng_seed,
    variant, qr_map
):
    sheet = _find_product_sheet(excel_path, product_code)
    cons_daily, stock_daily = _daily_consumption_and_stock(excel_path, sheet)
    vals = cons_daily.values
    split_index = int(len(vals) * window_ratio)
    if split_index < 2:
        return pd.DataFrame()

    rng = np.random.default_rng(rng_seed)
    rows, stock_after_interval = [], 0.0

    for i in range(split_index, len(vals)):
        if (i - split_index) % interval == 0:
            train = vals[:i]
            test_date = cons_daily.index[i]

            if variant == "sba":
                f = _croston_or_sba(train, alpha, "sba")
            elif variant == "croston":
                f = _croston_or_sba(train, alpha, "croston")
            else:
                f = _ses(train, alpha)

            sigma_period = float(pd.Series(train).std(ddof=1)) if i > 1 else 0.0
            sigma_period = sigma_period if np.isfinite(sigma_period) else 0.0

            real_demand = _interval_sum_next_days(cons_daily, i, interval)
            stock_on_hand_running = _interval_sum_next_days(stock_daily, i, interval)
            stock_after_interval = stock_after_interval + stock_on_hand_running - real_demand

            # ROP usine
            X_Lt = lead_time * f
            sigma_Lt = sigma_period * np.sqrt(max(lead_time, 1e-9))
            var_u = sigma_Lt**2 if sigma_Lt**2 > X_Lt else X_Lt + 1e-5
            p_nb = min(max(X_Lt / var_u, 1e-12), 1-1e-12)
            r_nb = X_Lt**2 / (var_u - X_Lt) if var_u > X_Lt else 1e6
            ROP_u = float(np.percentile(nbinom.rvs(r_nb, p_nb, size=nb_sim, random_state=rng), 100*service_level))

            # ROP fournisseur
            totalL = lead_time + lead_time_supplier
            X_Lt_Lw = totalL * f
            sigma_Lt_Lw = sigma_period * np.sqrt(max(totalL, 1e-9))
            var_f = sigma_Lt_Lw**2 if sigma_Lt_Lw**2 > X_Lt_Lw else X_Lt_Lw + 1e-5
            p_nb_f = min(max(X_Lt_Lw / var_f, 1e-12), 1-1e-12)
            r_nb_f = X_Lt_Lw**2 / (var_f - X_Lt_Lw) if var_f > X_Lt_Lw else 1e6
            ROP_f = float(np.percentile(nbinom.rvs(r_nb_f, p_nb_f, size=nb_sim, random_state=rng), 100*service_level))

            if stock_after_interval >= real_demand * lead_time:
                order_policy = "no_order"
            else:
                order_policy = f"order_Qr*_{qr_map[product_code]}"

            stock_status = "rupture" if real_demand > ROP_u else "holding"

            rows.append({
                "date": test_date.date(),
                "code": product_code,
                "interval": interval,
                "real_demand": real_demand,
                "stock_on_hand_running": stock_on_hand_running,
                "stock_after_interval": stock_after_interval,
                "order_policy": order_policy,
                "Qr_star": qr_map[product_code],
                "reorder_point_usine": ROP_u,
                "reorder_point_fournisseur": ROP_f,
                "stock_status": stock_status,
                "service_level": service_level
            })

    return pd.DataFrame(rows)

# ==============================================================
# MAIN PIPELINE
# ==============================================================

qr_map, qw_map = compute_qstars(EXCEL_PATH, PRODUCT_CODES)
st.subheader("Qr* and Qw* values")
st.write(pd.DataFrame({"Qr*": qr_map, "Qw*": qw_map}))

best_per_code = pd.read_excel("best_per_code.xlsx")  # you must prepare this like in your scripts
st.subheader("Best methods & parameters per code")
st.write(best_per_code)

st.subheader("Sensitivity Analysis")
final_results_sensitivity = []
for sl in SERVICE_LEVELS:
    st.markdown(f"### Service Level = {sl*100:.0f}%")
    results = []
    for _, row in best_per_code.iterrows():
        code = row["code"]
        method = row["method"].lower()
        alpha = row["alpha"]
        window_ratio = row["window_ratio"]
        interval = int(row["recalc_interval"])

        df_run = rolling_with_new_logic(
            excel_path=EXCEL_PATH,
            product_code=code,
            alpha=alpha, window_ratio=window_ratio, interval=interval,
            lead_time=LEAD_TIME, lead_time_supplier=LEAD_TIME_SUPPLIER,
            service_level=sl, nb_sim=NB_SIM, rng_seed=RNG_SEED,
            variant=method, qr_map=qr_map
        )
        results.append(df_run)
    df_concat = pd.concat(results, ignore_index=True)
    final_results_sensitivity.append(df_concat)
    st.write(df_concat.groupby("code")[["reorder_point_usine","reorder_point_fournisseur"]].mean())

summary = pd.concat(final_results_sensitivity, ignore_index=True)
st.subheader("📊 Global Summary")
st.write(summary.groupby(["code","service_level"])[["reorder_point_usine","reorder_point_fournisseur"]].mean().reset_index())
