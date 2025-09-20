# ============================================
# Streamlit App – Unifié (Final + Sensibilité)
# Minimal changes: original logic preserved, wrapped for Streamlit UI
# + Uploader for articles.xlsx (classification sheet) with code extraction
# ============================================

import numpy as np
import pandas as pd
import re
from scipy.stats import nbinom
import streamlit as st

st.set_page_config(page_title="Inventaire – Final & Sensibilité", layout="wide")

# ============================================
# --------- CHEMINS / LISTES (defaults) ---------
# (Can be overridden via the Streamlit sidebar uploaders.)
# ============================================
EXCEL_PATH_DATA_DEFAULT = "PFE  HANIN (1).xlsx"           # données par produit (onglets "time serie XXX")
PATH_SES_DEFAULT = "best_params_SES.xlsx"
PATH_CROSTON_DEFAULT = "best_params_CROSTON.xlsx"
PATH_SBA_DEFAULT = "best_params_SBA.xlsx"

CODES_PRODUITS_DEFAULT = ["EM0400", "EM1499", "EM1091", "EM1523", "EM0392", "EM1526"]

# --------- PARAMÈTRES SUPPLY / ROP ---------
DELAI_USINE_DEFAULT = 10            # jours
DELAI_FOURNISSEUR_DEFAULT = 3       # jours
NIVEAU_SERVICE_DEF_DEFAULT = 0.95
NB_SIM_DEFAULT = 1000
GRAINE_ALEA_DEFAULT = 42

# --------- COLONNES D'AFFICHAGE ---------
COLONNES_AFFICHAGE = [
    "date", "code", "methode", "intervalle",
    "demande_reelle", "stock_disponible_intervalle", "stock_apres_intervalle",
    "politique_commande", "Qr_etoile", "Qw_etoile", "n_etoile",
    "ROP_usine", "SS_usine",
    "ROP_fournisseur", "SS_fournisseur",
    "statut_stock", "service_level"
]

# --------- OUTILS AFFICHAGE ---------
def _disp(obj, n=None, title=None):
    """Display helper that works in Streamlit and falls back to print"""
    try:
        if title:
            st.subheader(title)
        if isinstance(obj, pd.DataFrame):
            st.dataframe(obj.head(n) if n else obj)
        else:
            st.write(obj)
    except Exception:
        if title:
            print(title)
        if isinstance(obj, pd.DataFrame):
            print(obj.head(n).to_string(index=False) if n else obj.to_string(index=False))
        else:
            print(obj)

# ======================================================
# PARTIE A : Q* (Qr*, Qw*, n*) depuis PFE HANIN
# ======================================================

def _trouver_feuille_produit(chemin_excel, code: str) -> str:
    xls = pd.ExcelFile(chemin_excel)
    feuilles = xls.sheet_names
    cible = f"time serie {code}"
    if cible in feuilles:
        return cible
    patt = re.compile(r"time\s*ser(i|ie)s?\s*", re.IGNORECASE)
    cand = [s for s in feuilles if patt.search(s) and code.lower() in s.lower()]
    if cand:
        return sorted(cand, key=len, reverse=True)[0]
    # parfois "time series"
    alt = f"time series {code}"
    if alt in feuilles:
        return alt
    raise ValueError(f"[Feuille] Onglet pour '{code}' introuvable dans le classeur fourni.")

def compute_qstars(chemin_excel, codes: list):
    """
    Calcule Qr*, Qw*, n* pour chaque article.
    Retourne trois dicts : qr_map, qw_map, n_map
    """
    df_conso = pd.read_excel(chemin_excel, sheet_name="consommation depots externe")
    df_conso = df_conso.groupby('Code Produit')['Quantite STIAL'].sum()

    qr_map, qw_map, n_map = {}, {}, {}
    for code in codes:
        feuille = _trouver_feuille_produit(chemin_excel, code)
        df = pd.read_excel(chemin_excel, sheet_name=feuille)

        C_r = df['Cr : cout stockage/article '].iloc[0]
        C_w = df['Cw : cout stockage\nchez F'].iloc[0]
        A_w = df['Aw : cout de\nlancement chez U'].iloc[0]
        A_r = df['Ar : cout de \nlancement chez F'].iloc[0]

        # n* (arrondi >=1)
        n_val = (A_w * C_r) / (A_r * C_w)
        n_val = 1 if n_val < 1 else round(n_val)
        n1, n2 = int(n_val), int(n_val) + 1
        F_n1 = (A_r + A_w / n1) * (n1 * C_w + C_r)
        F_n2 = (A_r + A_w / n2) * (n2 * C_w + C_r)
        n_star = n1 if F_n1 <= F_n2 else n2

        # Demande totale D (déjà agrégée)
        D = df_conso.get(code, 0)
        tau = 1  # période (jour)

        Qr_star = ((2 * (A_r + A_w / n_star) * D) / (n_star * C_w + C_r * tau)) ** 0.5
        Qw_star = n_star * Qr_star

        qr_map[code] = round(float(Qr_star), 2)
        qw_map[code] = round(float(Qw_star), 2)
        n_map[code]  = int(n_star)
    return qr_map, qw_map, n_map

# ======================================================
# PARTIE B : Séries conso/stock journalières
# ======================================================

def _series_conso_stock_jour(chemin_excel, feuille: str):
    df = pd.read_excel(chemin_excel, sheet_name=feuille)
    col_date, col_stock, col_conso = df.columns[0], df.columns[1], df.columns[2]

    dates = pd.to_datetime(df[col_date], errors="coerce")
    conso = pd.to_numeric(df[col_conso], errors="coerce").fillna(0.0).astype(float)
    stock = pd.to_numeric(df[col_stock], errors="coerce").fillna(0.0).astype(float)

    ts_conso = pd.DataFrame({"d": dates, "q": conso}).dropna().sort_values("d").set_index("d")["q"]
    ts_stock = pd.DataFrame({"d": dates, "s": stock}).dropna().sort_values("d").set_index("d")["s"]

    min_date = min(ts_conso.index.min(), ts_stock.index.min())
    max_date = max(ts_conso.index.max(), ts_stock.index.max())
    idx_complet = pd.date_range(min_date, max_date, freq="D")

    conso_jour = ts_conso.reindex(idx_complet, fill_value=0.0)
    stock_jour = ts_stock.reindex(idx_complet).ffill().fillna(0.0)
    return conso_jour, stock_jour

def _somme_intervalle(serie: pd.Series, start_idx: int, intervalle: int) -> float:
    s, e = start_idx + 1, start_idx + 1 + int(intervalle)
    return float(pd.Series(serie).iloc[s:e].sum())

# ======================================================
# PARTIE C : Méthodes de prévision
# ======================================================

def _croston_or_sba(x, alpha: float, variant: str = "sba"):
    x = pd.Series(x).fillna(0.0).astype(float).values
    x = np.where(x < 0, 0.0, x)
    if (x == 0).all():
        return 0.0
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
    if variant.lower() == "sba":
        f *= (1 - alpha / 2.0)
    return float(f)

def _ses(x, alpha: float):
    x = pd.Series(x).fillna(0.0).astype(float).values
    if len(x) == 0:
        return 0.0
    l = x[0]
    for t in range(1, len(x)):
        l = alpha * x[t] + (1 - alpha) * l
    return float(l)

# ======================================================
# PARTIE D : Rolling final (avec Qr*/Qw*/n* + ROP/SS + statut)
# ======================================================

def rolling_with_new_logic(
    excel_path, product_code, alpha, window_ratio, intervalle,
    delai_usine, delai_fournisseur, service_level, nb_sim, rng_seed,
    variant, qr_map, qw_map, n_map
):
    feuille = _trouver_feuille_produit(excel_path, product_code)
    conso_jour, stock_jour = _series_conso_stock_jour(excel_path, feuille)
    vals = conso_jour.values
    split_index = int(len(vals) * float(window_ratio))
    if split_index < 2:
        return pd.DataFrame()

    rng = np.random.default_rng(rng_seed)
    lignes = []
    stock_apres_intervalle = 0.0

    for i in range(split_index, len(vals)):
        if (i - split_index) % int(intervalle) == 0:
            train = vals[:i]
            date_test = conso_jour.index[i]

            # Prévision par méthode
            if variant == "sba":
                f = _croston_or_sba(train, alpha, "sba")
            elif variant == "croston":
                f = _croston_or_sba(train, alpha, "croston")
            else:
                f = _ses(train, alpha)

            sigma = float(pd.Series(train).std(ddof=1)) if i > 1 else 0.0
            sigma = sigma if np.isfinite(sigma) else 0.0

            demande_reelle = _somme_intervalle(conso_jour, i, intervalle)
            stock_dispo = _somme_intervalle(stock_jour, i, intervalle)
            stock_apres_intervalle = stock_apres_intervalle + stock_dispo - demande_reelle

            # ROP usine (demande sur delai_usine ~ NB)
            X_Lt = delai_usine * f
            sigma_Lt = sigma * np.sqrt(max(delai_usine, 1e-9))
            var_u = sigma_Lt**2 if sigma_Lt**2 > X_Lt else X_Lt + 1e-5
            p_nb = min(max(X_Lt / var_u, 1e-12), 1 - 1e-12)
            r_nb = X_Lt**2 / (var_u - X_Lt) if var_u > X_Lt else 1e6
            ROP_u = float(np.percentile(nbinom.rvs(r_nb, p_nb, size=nb_sim, random_state=rng), 100 * service_level))
            SS_u = max(ROP_u - X_Lt, 0.0)

            # ROP fournisseur (demande sur delai total)
            totalL = delai_usine + delai_fournisseur
            X_Lt_Lw = totalL * f
            sigma_Lt_Lw = sigma * np.sqrt(max(totalL, 1e-9))
            var_f = sigma_Lt_Lw**2 if sigma_Lt_Lw**2 > X_Lt_Lw else X_Lt_Lw + 1e-5
            p_nb_f = min(max(X_Lt_Lw / var_f, 1e-12), 1 - 1e-12)
            r_nb_f = X_Lt_Lw**2 / (var_f - X_Lt_Lw) if var_f > X_Lt_Lw else 1e6
            ROP_f = float(np.percentile(nbinom.rvs(r_nb_f, p_nb_f, size=nb_sim, random_state=rng), 100 * service_level))
            SS_f = max(ROP_f - X_Lt_Lw, 0.0)

            # Comparaison à l’intervalle : ROP usine à l’échelle de l’intervalle
            ROP_u_interval = ROP_u * (intervalle / max(delai_usine, 1e-9))

            # Politique de commande : Qr* (pas Qw*)
            if stock_apres_intervalle >= demande_reelle * delai_usine:
                politique = "pas_de_commande"
            else:
                politique = f"commander_Qr*_{qr_map[product_code]}"

            statut = "rupture" if demande_reelle > ROP_u_interval else "holding"

            lignes.append({
                "date": date_test.date(),
                "code": product_code,
                "methode": variant,
                "intervalle": int(intervalle),
                "demande_reelle": float(demande_reelle),
                "stock_disponible_intervalle": float(stock_dispo),
                "stock_apres_intervalle": float(stock_apres_intervalle),
                "politique_commande": politique,
                "Qr_etoile": float(qr_map[product_code]),
                "Qw_etoile": float(qw_map[product_code]),
                "n_etoile": int(n_map[product_code]),
                "ROP_usine": float(ROP_u),
                "SS_usine": float(SS_u),
                "ROP_fournisseur": float(ROP_f),
                "SS_fournisseur": float(SS_f),
                "statut_stock": statut,
                "service_level": float(service_level),
            })

    return pd.DataFrame(lignes)

# ======================================================
# PARTIE E : Charger les meilleurs paramètres + méthode
# ======================================================

def _normalize_df_best(df_best: pd.DataFrame, method_name: str, pick_metric: str = "RMSE") -> pd.DataFrame:
    metric_key = pick_metric.upper()
    if metric_key == "ABSME":
        a, w, itv, s = "best_ME_alpha", "best_ME_window", "best_ME_interval", "best_absME"
    else:
        a, w, itv, s = f"best_{metric_key}_alpha", f"best_{metric_key}_window", f"best_{metric_key}_interval", f"best_{metric_key}"

    # fallback au cas où
    for cand in [
        (a, w, itv, s),
        ("best_RMSE_alpha", "best_RMSE_window", "best_RMSE_interval", "best_RMSE"),
        ("best_MSE_alpha",  "best_MSE_window",  "best_MSE_interval",  "best_MSE"),
        ("best_ME_alpha",   "best_ME_window",   "best_ME_interval",   "best_ME"),
    ]:
        if all(c in df_best.columns for c in cand):
            a, w, itv, s = cand
            break

    out = df_best.rename(columns={
        a: "alpha", w: "window_ratio", itv: "recalc_interval", s: "score"
    })
    keep = ["code", "alpha", "window_ratio", "recalc_interval", "score"]
    if "n_points_used" in df_best.columns:
        out["n_points_used"] = df_best["n_points_used"]
        keep.append("n_points_used")
    out = out[keep].copy()
    for c in ["alpha", "window_ratio", "recalc_interval", "score"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["method"] = method_name
    return out

def select_best_method_from_files(
    path_ses, path_cro, path_sba,
    product_filter=None, pick_metric="RMSE"
):
    df_best_SES = pd.read_excel(path_ses)
    df_best_CRO = pd.read_excel(path_cro)
    df_best_SBA = pd.read_excel(path_sba)

    cand_ses = _normalize_df_best(df_best_SES, "ses", pick_metric)
    cand_cro = _normalize_df_best(df_best_CRO, "croston", pick_metric)
    cand_sba = _normalize_df_best(df_best_SBA, "sba", pick_metric)

    candidates = pd.concat([cand_ses, cand_cro, cand_sba], ignore_index=True)
    if product_filter:
        candidates = candidates[candidates["code"].isin(product_filter)].copy()

    # meilleure ligne par article (score min)
    idx = candidates.groupby("code")["score"].idxmin()
    best_per_code = candidates.loc[idx].sort_values(["code"]).reset_index(drop=True)
    return best_per_code

# ======================================================
# PARTIE F : Final (SL unique) + Sensibilité (plusieurs SL)
# ======================================================

def run_final_once(best_per_code: pd.DataFrame, service_level=0.95, excel_path_data=None):
    qr_map, qw_map, n_map = compute_qstars(excel_path_data, best_per_code["code"].tolist())
    results = []
    for _, row in best_per_code.iterrows():
        code = row["code"]
        method = str(row["method"]).lower()
        alpha = float(row["alpha"])
        window_ratio = float(row["window_ratio"])
        intervalle = int(row["recalc_interval"])
        df_run = rolling_with_new_logic(
            excel_path=excel_path_data,
            product_code=code,
            alpha=alpha, window_ratio=window_ratio, intervalle=intervalle,
            delai_usine=st.session_state.get('DELAI_USINE', DELAI_USINE_DEFAULT),
            delai_fournisseur=st.session_state.get('DELAI_FOURNISSEUR', DELAI_FOURNISSEUR_DEFAULT),
            service_level=service_level, nb_sim=st.session_state.get('NB_SIM', NB_SIM_DEFAULT), rng_seed=st.session_state.get('GRAINE_ALEA', GRAINE_ALEA_DEFAULT),
            variant=method, qr_map=qr_map, qw_map=qw_map, n_map=n_map
        )
        results.append(df_run)
        if not df_run.empty:
            _disp(df_run[COLONNES_AFFICHAGE], n=10, title=f"\n=== {code} — {method.upper()} (SL={service_level:.2f}) ===")
    if not results:
        return pd.DataFrame()
    return pd.concat(results, ignore_index=True)

def run_sensitivity(best_per_code: pd.DataFrame, service_levels=[0.90, 0.92, 0.95, 0.98], excel_path_data=None):
    qr_map, qw_map, n_map = compute_qstars(excel_path_data, best_per_code["code"].tolist())
    all_results = []
    for sl in service_levels:
        st.markdown(f"## 🔎 Simulation avec Service Level = {sl*100:.0f}%")
        runs = []
        for _, row in best_per_code.iterrows():
            code = row["code"]
            method = str(row["method"]).lower()
            alpha = float(row["alpha"])
            window_ratio = float(row["window_ratio"])
            intervalle = int(row["recalc_interval"])

            df_run = rolling_with_new_logic(
                excel_path=excel_path_data,
                product_code=code,
                alpha=alpha, window_ratio=window_ratio, intervalle=intervalle,
                delai_usine=st.session_state.get('DELAI_USINE', DELAI_USINE_DEFAULT),
                delai_fournisseur=st.session_state.get('DELAI_FOURNISSEUR', DELAI_FOURNISSEUR_DEFAULT),
                service_level=sl, nb_sim=st.session_state.get('NB_SIM', NB_SIM_DEFAULT), rng_seed=st.session_state.get('GRAINE_ALEA', GRAINE_ALEA_DEFAULT),
                variant=method, qr_map=qr_map, qw_map=qw_map, n_map=n_map
            )
            df_run["service_level"] = sl
            runs.append(df_run)

        df_concat = pd.concat(runs, ignore_index=True) if runs else pd.DataFrame()
        all_results.append(df_concat)

        # Aperçu par article : moyennes ROP & SS + parts holding/rupture + rappel n*
        if not df_concat.empty:
            grp = df_concat.groupby("code").agg(
                ROP_usine_moy=("ROP_usine", "mean"),
                SS_usine_moy=("SS_usine", "mean"),
                ROP_fournisseur_moy=("ROP_fournisseur", "mean"),
                SS_fournisseur_moy=("SS_fournisseur", "mean"),
                holding_pct=("statut_stock", lambda s: (s == "holding").mean()*100),
                rupture_pct=("statut_stock", lambda s: (s == "rupture").mean()*100),
                Qr_star=("Qr_etoile", "first"),
                Qw_star=("Qw_etoile", "first"),
                n_star=("n_etoile", "first"),
            ).reset_index()
            _disp(grp, title=f"=== Résultats pour SL {sl*100:.0f}% (moyennes par article) ===")

    return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()

# ============================================
# STREAMLIT APP
# ============================================

def main():
    st.title("📦 Simulation d'inventaire – Final & Sensibilité")
    st.caption("Version Streamlit dérivée du script d'origine (logique inchangée, UI ajoutée)")

    # Sidebar – fichiers
    st.sidebar.header("Fichiers d'entrée")
    excel_data = st.sidebar.file_uploader("Classeur principal (PFE HANIN)", type=["xlsx", "xls"], key="data")
    best_ses = st.sidebar.file_uploader("Paramètres SES", type=["xlsx", "xls"], key="ses")
    best_cro = st.sidebar.file_uploader("Paramètres CROSTON", type=["xlsx", "xls"], key="cro")
    best_sba = st.sidebar.file_uploader("Paramètres SBA", type=["xlsx", "xls"], key="sba")

    # --- NEW: articles.xlsx uploader + sheet picker + auto codes ---
    articles_file = st.sidebar.file_uploader(
        "Matrice d'articles (articles.xlsx)", type=["xlsx", "xls"], key="articles"
    )

    available_codes = None
    if articles_file is not None:
        xls = pd.ExcelFile(articles_file)
        sheet_names = xls.sheet_names
        # default to 'classification' if present (case-insensitive)
        default_idx = 0
        for i, s in enumerate(sheet_names):
            if str(s).strip().lower() == "classification":
                default_idx = i
                break
        sheet_choice = st.sidebar.selectbox("Feuille matrice", options=sheet_names, index=default_idx)
        df_matrix = pd.read_excel(articles_file, sheet_name=sheet_choice)
        prod_col = df_matrix.columns[0]
        available_codes = df_matrix[prod_col].astype(str).dropna().unique().tolist()

    # Sidebar – paramètres
    st.sidebar.header("Paramètres")

    # If we have the matrix, pick codes from it; otherwise fallback to manual input
    if available_codes:
        codes_products = st.sidebar.multiselect(
            "Codes produits (depuis articles.xlsx)",
            options=available_codes,
            default=available_codes
        )
    else:
        codes_text = st.sidebar.text_input("Codes produits (séparés par des virgules)", ", ".join(CODES_PRODUITS_DEFAULT))
        codes_products = [c.strip() for c in codes_text.split(",") if c.strip()]

    pick_metric = st.sidebar.selectbox("Métrique d'optimisation", ["RMSE", "MSE", "ME", "ABSME"], index=0)

    st.session_state['DELAI_USINE'] = st.sidebar.number_input("Délai usine (jours)", min_value=1, value=DELAI_USINE_DEFAULT)
    st.session_state['DELAI_FOURNISSEUR'] = st.sidebar.number_input("Délai fournisseur (jours)", min_value=0, value=DELAI_FOURNISSEUR_DEFAULT)
    st.session_state['NIVEAU_SERVICE_DEF'] = st.sidebar.slider("Niveau de service par défaut", 0.80, 0.999, value=float(NIVEAU_SERVICE_DEF_DEFAULT))
    st.session_state['NB_SIM'] = st.sidebar.number_input("Nombre de simulations (NB)", min_value=100, value=NB_SIM_DEFAULT, step=100)
    st.session_state['GRAINE_ALEA'] = st.sidebar.number_input("Graine aléatoire", value=GRAINE_ALEA_DEFAULT)

    sensi_levels_str = st.sidebar.text_input("Niveaux de service (sensibilité)", "0.90, 0.92, 0.95, 0.98")
    try:
        sensi_levels = [float(x.strip()) for x in sensi_levels_str.split(',') if x.strip()]
    except Exception:
        sensi_levels = [0.90, 0.92, 0.95, 0.98]

    # Tabs
    tab1, tab2, tab3 = st.tabs(["▶️ Run 95%", "📈 Sensibilité", "📊 Résumé global"])

    # Pre-flight checks
    if not (excel_data and best_ses and best_cro and best_sba):
        st.info("Chargez les 4 fichiers Excel (PFE HANIN + best_params) dans la barre latérale pour lancer les calculs.")
        return

    if not codes_products:
        st.warning("Aucun code produit sélectionné.")
        return

    # Best per code
    with st.spinner("Chargement des meilleurs paramètres par article..."):
        best_per_code = select_best_method_from_files(
            path_ses=best_ses, path_cro=best_cro, path_sba=best_sba,
            product_filter=codes_products, pick_metric=pick_metric
        )
    _disp(best_per_code, title="✅ Meilleure méthode et meilleurs paramètres par article")

    with tab1:
        st.markdown("### Recalcul final au niveau de service par défaut")
        with st.spinner("Exécution du calcul final..."):
            final_df = run_final_once(best_per_code, service_level=st.session_state['NIVEAU_SERVICE_DEF'], excel_path_data=excel_data)
        if final_df is not None and not final_df.empty:
            st.dataframe(final_df)
            csv = final_df.to_csv(index=False).encode('utf-8')
            st.download_button("Télécharger CSV (final)", data=csv, file_name="final_95.csv", mime="text/csv")
        else:
            st.warning("Aucun résultat (vérifiez les données ou les paramètres).")

    with tab2:
        st.markdown("### Analyse de sensibilité")
        with st.spinner("Exécution de la sensibilité..."):
            sensi_df = run_sensitivity(best_per_code, service_levels=sensi_levels, excel_path_data=excel_data)
        if sensi_df is not None and not sensi_df.empty:
            st.dataframe(sensi_df)
            csv2 = sensi_df.to_csv(index=False).encode('utf-8')
            st.download_button("Télécharger CSV (sensibilité)", data=csv2, file_name="sensibilite.csv", mime="text/csv")
        else:
            st.warning("Aucune sortie de sensibilité.")

    with tab3:
        st.markdown("### Résumé global (par code & SL)")
        # Use sensi_df if available from tab2
        if 'sensi_df' in locals() and sensi_df is not None and not sensi_df.empty:
            summary = sensi_df.groupby(["code", "service_level"]).agg(
                ROP_u_moy=("ROP_usine", "mean"),
                SS_u_moy=("SS_usine", "mean"),
                ROP_f_moy=("ROP_fournisseur", "mean"),
                SS_f_moy=("SS_fournisseur", "mean"),
                holding_pct=("statut_stock", lambda s: (s == "holding").mean()*100),
                rupture_pct=("statut_stock", lambda s: (s == "rupture").mean()*100),
                Qr_star=("Qr_etoile", "first"),
                Qw_star=("Qw_etoile", "first"),
                n_star=("n_etoile", "first"),
            ).reset_index()
            st.dataframe(summary)
            csv3 = summary.to_csv(index=False).encode('utf-8')
            st.download_button("Télécharger CSV (résumé)", data=csv3, file_name="resume_global.csv", mime="text/csv")
        else:
            st.info("Exécutez d'abord l'analyse de sensibilité pour voir le résumé global.")

if __name__ == "__main__":
    main()
