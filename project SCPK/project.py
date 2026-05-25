import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64

with open("logo.jpg", "rb") as img:
    LOGO_B64 = base64.b64encode(img.read()).decode("utf-8")

st.set_page_config(
    page_title="SPK Restoran Yogyakarta",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(f"""
<style>
  :root {{
    --primary: #FF3B30; --brown-dark: #6B3C2E; --brown-mid: #D68A6B;
    --caramel: #C98D6B; --cream: #FFF1E1; --skin: #FFD9B3;
    --bg: #FFF8F3; --surface: #FFFFFF; --border: #F0D9C8;
    --text: #3A1A10; --text-muted: #8C6052;
  }}
  html, body, [class*="css"] {{ font-family: Times New Roman, serif; color: var(--text); }}
  .stApp {{ background: var(--bg); }}
  h1,h2,h3,h4 {{ font-family: Georgia, serif; color: var(--brown-dark) !important; }}
  hr {{ border-color: var(--border) !important; }}

  [data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #3A1A10, #6B3C2E) !important;
  }}
  .sidebar-header {{ text-align:center; padding:20px 12px 14px; }}
  .sidebar-logo-img {{ width:80px; height:80px; border-radius:50%; object-fit:cover; border:2px solid rgba(255,255,255,0.3); margin-bottom:8px; }}
  .sidebar-title {{ font-family:Georgia,serif; font-size:20px; font-weight:bold; color:#FFFFFF; margin-bottom:3px; }}
  .sidebar-subtitle {{ font-size:12px; color:#FFD9B3; }}
  .sidebar-divider {{ border:none; height:1px; background:rgba(255,255,255,0.1); margin:14px 0; }}
  .sidebar-label {{ font-size:11px; letter-spacing:1.5px; color:#E7A57C; margin-bottom:8px; padding-left:6px; text-transform:uppercase; }}
  [data-testid="stSidebar"] .stRadio label {{ border-radius:8px !important; padding:8px 12px !important; }}
  [data-testid="stSidebar"] .stRadio label[data-selected="true"] {{ background:rgba(255,59,48,0.25) !important; border-left:3px solid #FF3B30 !important; }}
  [data-testid="stSidebar"] .stRadio label p {{ font-size:14px !important; color:#FFF1E1 !important; }}
  [data-testid="stSidebar"] input[type="radio"] {{ display:none; }}

  .metric-card {{ background:var(--surface); border-radius:10px; padding:16px 18px; border:1px solid var(--border); }}
  .metric-label {{ font-size:11px; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.6px; }}
  .metric-value {{ font-size:26px; font-weight:bold; margin:4px 0 2px; font-family:Georgia,serif; }}
  .metric-sub {{ font-size:12px; color:var(--text-muted); }}
  .metric-red .metric-value {{ color:var(--primary); }}
  .metric-brown .metric-value {{ color:var(--brown-dark); }}
  .metric-caramel .metric-value {{ color:var(--caramel); }}

  .hero-box {{ background:linear-gradient(135deg,#6B3C2E,#3A1A10); border-radius:12px; padding:24px 28px; margin-bottom:20px; border-left:5px solid #FF3B30; }}
  .hero-title {{ font-family:Georgia,serif; font-size:26px; font-weight:bold; color:#FFFFFF; margin-bottom:6px; }}
  .hero-sub {{ font-size:14px; color:#FFD9B3; line-height:1.6; max-width:520px; }}

  .info-box {{ background:var(--cream); border-left:4px solid var(--brown-mid); border-radius:0 8px 8px 0; padding:12px 16px; font-size:14px; margin:10px 0; }}
  .warn-box {{ background:#FFF0ED; border-left:4px solid var(--primary); border-radius:0 8px 8px 0; padding:12px 16px; font-size:14px; color:#7A1A10; margin:10px 0; }}

  .stButton > button {{ background:#FF3B30 !important; color:white !important; border:none !important; border-radius:8px !important; font-weight:bold !important; font-family:Georgia,serif !important; }}

  .profile-card {{ background:var(--surface); border-radius:12px; padding:24px; border:1px solid var(--border); text-align:center; }}
  .avatar-ring {{ width:72px; height:72px; border-radius:50%; background:linear-gradient(135deg,#FF3B30,#D68A6B); display:flex; align-items:center; justify-content:center; font-size:28px; margin:0 auto 12px; }}
</style>
""", unsafe_allow_html=True)

# ─── SESSION STATE ────────────────────────────────────────────────────────────
if "current_page" not in st.session_state:
    st.session_state.current_page = "Beranda"
if "ahp_result" not in st.session_state:
    st.session_state.ahp_result = None
if "comparisons" not in st.session_state:
    st.session_state.comparisons = {}
if "ranking_list" not in st.session_state:
    st.session_state.ranking_list = None

# ─── AHP FUNCTIONS ────────────────────────────────────────────────────────────
def normalize_comparation(M):
    M = np.array(M, dtype=float)
    return M / M.sum(axis=0)

def weight(M_norm):
    return np.mean(M_norm, axis=1)

RI_TABLE = {1:0.00,2:0.00,3:0.58,4:0.90,5:1.12,6:1.24,
            7:1.32,8:1.41,9:1.45,10:1.51,11:1.53,12:1.54,13:1.56,14:1.57}

def validity_check(M, W):
    M = np.array(M, dtype=float)
    W = np.array(W, dtype=float)
    n = len(M)
    CV = (M @ W) / W
    eigen = np.mean(CV)
    ri = RI_TABLE.get(n, 1.57)
    CI = (eigen - n) / (n - 1) if n > 1 else 0
    CR = CI / ri if ri > 0 else 0
    return {"CV": CV, "eigen": eigen, "CI": CI, "RI": ri, "CR": CR, "valid": CR <= 0.1}

def build_reciprocal_matrix(values):
    n = len(values)
    values = np.array(values, dtype=float)
    M = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            M[i][j] = values[i] / values[j] if values[j] != 0 else 1.0
    return M

def build_reciprocal_matrix_lower(values):
    n = len(values)
    values = np.array(values, dtype=float)
    M = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            M[i][j] = values[j] / values[i] if values[i] != 0 else 1.0
    return M

def final_weight(W_alt, W_crit):
    return np.array(W_alt, dtype=float).T @ np.array(W_crit, dtype=float)

def saaty_to_matrix(n_criteria, comparisons):
    M = np.ones((n_criteria, n_criteria))
    for (i, j), v in comparisons.items():
        M[i][j] = v
        M[j][i] = 1.0 / v
    return M

# ─── LOAD DATA ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("places_to_eat_in_the_jogja_region.csv")
    df["Jarak (km)"] = df["Lokasi Restoran"].str.extract(r"([\d.]+)").astype(float)
    df["Diskon"] = (df["Toko Sering Diskon (Ya/Tidak)"] == "Ya").astype(int)
    df["Pasangan"] = (df["Pelayanan Khusus Pasangan (Ya/Tidak)"] == "Ya").astype(int)
    df["AYCE"] = (df["All You Can Eat atau Ala Carte"] == "All You Can Eat").astype(int)
    df["Nama Singkat"] = df["Nama Restoran"].str.split(",").str[0].str.strip()
    return df

df = load_data()

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
PAGES = ["Beranda", "Input Bobot AHP", "Proses AHP", "Hasil Ranking", "Profil Kelompok"]
PAGES_LABEL = ["Beranda", "Input Bobot AHP", "Proses AHP", "Hasil Ranking", "Profil Kelompok"]

with st.sidebar:
    st.markdown(f"""
    <div class="sidebar-header">
        <img src="data:image/jpeg;base64,{LOGO_B64}" class="sidebar-logo-img" alt="Logo SPK Restoran"/>
        <div class="sidebar-title">SPK Restoran</div>
        <div class="sidebar-subtitle">Yogyakarta &bull; Metode AHP</div>
    </div>
    <hr class="sidebar-divider">
    <div class="sidebar-label">Navigasi</div>
    """, unsafe_allow_html=True)

    cur = st.session_state.current_page
    cur_idx = PAGES.index(cur) if cur in PAGES else 0
    page = st.radio("Pilih halaman:", PAGES, index=cur_idx, label_visibility="collapsed")
    st.session_state.current_page = page

    st.markdown("""
    <hr style='border-color:rgba(231,165,124,0.2);margin:12px 0;'/>
    <div style='text-align:center;font-size:11px;color:#E7A57C;opacity:0.8;padding-bottom:16px;'>
      Sistem Cerdas Pendukung Keputusan<br/>© 2025 &middot; UPN Veteran Yogyakarta
    </div>
    """, unsafe_allow_html=True)

# ─── KRITERIA SETUP ───────────────────────────────────────────────────────────
CRITERIA = {
    "Rating":        {"col": "Rating Toko",                          "benefit": True,  "label": "Rating"},
    "Harga":         {"col": "Harga Rata-Rata Makanan di Toko (Rp)", "benefit": False, "label": "Harga"},
    "Jarak":         {"col": "Jarak (km)",                           "benefit": False, "label": "Jarak"},
    "Entertainment": {"col": "Entertainment",                        "benefit": True,  "label": "Entertainment"},
    "Keramaian":     {"col": "Keramaian Restoran",                   "benefit": True,  "label": "Keramaian"},
}
CRIT_NAMES = list(CRITERIA.keys())
N_CRIT = len(CRIT_NAMES)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: BERANDA
# ═════════════════════════════════════════════════════════════════════════════
if page == "Beranda":
    st.markdown("""
    <div class='hero-box'>
        <div class='hero-title'>Sistem Pendukung Keputusan Restoran</div>
        <div class='hero-sub'>
            Pemilihan restoran terbaik di Yogyakarta menggunakan metode
            <strong>Analytical Hierarchy Process (AHP)</strong>.
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class='metric-card metric-red'>
            <div class='metric-label'>Total Restoran</div>
            <div class='metric-value'>{len(df)}</div>
            <div class='metric-sub'>Data restoran tersedia</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class='metric-card metric-brown'>
            <div class='metric-label'>Jumlah Kriteria</div>
            <div class='metric-value'>5</div>
            <div class='metric-sub'>Kriteria keputusan AHP</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class='metric-card metric-caramel'>
            <div class='metric-label'>Metode SPK</div>
            <div class='metric-value'>AHP</div>
            <div class='metric-sub'>Analytical Hierarchy Process</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Tentang Metode AHP")
    st.markdown("""
    <div class='info-box'>
    <b>Analytical Hierarchy Process (AHP)</b> adalah metode pengambilan keputusan multikriteria
    yang dikembangkan oleh <b>Dr. Thomas L. Saaty</b> pada awal tahun 1970-an.
    Metode ini bekerja dengan membandingkan setiap kriteria secara berpasangan
    (<i>pairwise comparison</i>) menggunakan <b>Skala Saaty (1-9)</b>,
    menghitung bobot prioritas setiap kriteria dan alternatif, lalu memvalidasi
    konsistensi penilaian melalui <b>Consistency Ratio (CR &le; 0.1)</b>.
    Hasil akhirnya berupa ranking restoran terbaik berdasarkan preferensi pengguna.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Dataset Restoran Yogyakarta")

    col1, col2 = st.columns([2, 1])
    with col1:
        search = st.text_input("Cari nama restoran", placeholder="Contoh: Gudeg, Steak, Sushi...")
    with col2:
        rating_range = st.slider(
            "Range Rating",
            float(df["Rating Toko"].min()), float(df["Rating Toko"].max()),
            (float(df["Rating Toko"].min()), float(df["Rating Toko"].max()))
        )
    with st.expander("Filter Suasana", expanded=False):
        suasana_filter = st.multiselect(
            "Jenis Suasana",
            df["Jenis Suasana"].unique().tolist(),
            default=df["Jenis Suasana"].unique().tolist()
        )

    filtered = df.copy()
    if search:
        filtered = filtered[filtered["Nama Restoran"].str.contains(search, case=False, na=False)]
    filtered = filtered[filtered["Jenis Suasana"].isin(suasana_filter)]
    filtered = filtered[(filtered["Rating Toko"] >= rating_range[0]) & (filtered["Rating Toko"] <= rating_range[1])]

    st.markdown(f"<p style='color:var(--text-muted);font-size:13px;'>Menampilkan <strong>{len(filtered)}</strong> dari {len(df)} restoran</p>", unsafe_allow_html=True)

    show_cols = ["Nama Singkat", "Preferensi Makanan", "Lokasi Restoran", "Rating Toko",
                 "Harga Rata-Rata Makanan di Toko (Rp)", "Jenis Suasana",
                 "Entertainment", "Keramaian Restoran", "Toko Sering Diskon (Ya/Tidak)"]
    st.dataframe(filtered[show_cols].reset_index(drop=True), use_container_width=True, height=420)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Mulai Input Bobot AHP"):
        st.session_state.current_page = "Input Bobot AHP"
        st.rerun()

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: INPUT BOBOT AHP
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Input Bobot AHP":
    st.markdown("""
    <div class='hero-box'>
        <div class='hero-title'>Input Perbandingan Kriteria AHP</div>
        <div class='hero-sub'>Tentukan tingkat kepentingan antar kriteria menggunakan skala Saaty.</div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Panduan Skala Saaty", expanded=False):
        saaty_df = pd.DataFrame({
            "Nilai": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            "Keterangan": [
                "Sama penting", "Antara 1 dan 3", "Sedikit lebih penting",
                "Antara 3 dan 5", "Lebih penting", "Antara 5 dan 7",
                "Sangat lebih penting", "Antara 7 dan 9", "Mutlak lebih penting"
            ]
        })
        st.dataframe(saaty_df, use_container_width=True, hide_index=True)

    st.markdown("### Perbandingan Berpasangan Kriteria")
    st.caption("Pilih tingkat kepentingan antar kriteria untuk membentuk matriks pairwise comparison.")

    comparisons = {}
    pairs = [(i, j) for i in range(N_CRIT) for j in range(i + 1, N_CRIT)]
    for idx in range(0, len(pairs), 2):
        row_pairs = pairs[idx:idx + 2]
        cols = st.columns(2)
        for col_idx, (i, j) in enumerate(row_pairs):
            with cols[col_idx]:
                ci_name = CRIT_NAMES[i]
                cj_name = CRIT_NAMES[j]
                label_i = CRITERIA[ci_name]["label"]
                label_j = CRITERIA[cj_name]["label"]
                key = f"comp_{i}_{j}"
                default_val = st.session_state.comparisons.get(key, 0)
                st.markdown(f"""
                <div style='font-size:14px;font-weight:700;margin-bottom:8px;margin-top:8px;color:var(--text);'>
                    {label_i} vs {label_j}
                </div>""", unsafe_allow_html=True)
                c1, c2 = st.columns([2, 1])
                with c1:
                    direction = st.selectbox(
                        "Arah",
                        [f"{ci_name} lebih penting", "Sama penting", f"{cj_name} lebih penting"],
                        index=1 if default_val == 0 else (0 if default_val > 0 else 2),
                        key=f"dir_{i}_{j}", label_visibility="collapsed"
                    )
                with c2:
                    magnitude = st.select_slider(
                        "Nilai", options=[1,2,3,4,5,6,7,8,9],
                        value=max(1, abs(int(default_val))) if default_val != 0 else 1,
                        key=f"mag_{i}_{j}", label_visibility="collapsed"
                    )
                st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
                if "Sama" in direction:
                    val = 1.0
                elif ci_name in direction:
                    val = float(magnitude)
                else:
                    val = 1.0 / float(magnitude)
                comparisons[(i, j)] = val
                st.session_state.comparisons[key] = val

    st.session_state.comparisons.update({f"comp_{i}_{j}": v for (i, j), v in comparisons.items()})

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Lanjut Hitung AHP"):
        st.session_state.current_page = "Proses AHP"
        st.rerun()

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: PROSES AHP
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Proses AHP":
    st.markdown("""
    <div class='hero-box'>
        <div class='hero-title'>Proses Perhitungan AHP</div>
        <div class='hero-sub'>
            Menampilkan proses AHP: matriks perbandingan, normalisasi,
            bobot prioritas, dan consistency ratio (CR).
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class='info-box'>
        Klik tombol di bawah untuk menjalankan proses AHP berdasarkan bobot pairwise comparison yang telah diinput.
    </div>
    """, unsafe_allow_html=True)

    if st.button("Hitung AHP"):
        with st.spinner("Menghitung proses AHP..."):
            comp_dict = {
                (i, j): st.session_state.comparisons.get(f"comp_{i}_{j}", 1.0)
                for i in range(N_CRIT) for j in range(i + 1, N_CRIT)
            }
            M_crit = saaty_to_matrix(N_CRIT, comp_dict)
            M_crit_norm = normalize_comparation(M_crit)
            W_crit = weight(M_crit_norm)

            crit_validity = validity_check(M_crit, W_crit)
            if not crit_validity["valid"]:
                st.error(
                    f"Consistency Ratio (CR) = {crit_validity['CR']:.4f}\n\n"
                    "Matriks perbandingan kriteria tidak konsisten.\n"
                    "Silakan perbaiki input bobot pairwise comparison hingga CR <= 0.1!")
                st.stop()

            alt_weights = {}
            alt_details = {}
            for cname, cinfo in CRITERIA.items():
                vals = df[cinfo["col"]].values.astype(float)
                M_alt = build_reciprocal_matrix(vals) if cinfo["benefit"] else build_reciprocal_matrix_lower(vals)
                M_alt_norm = normalize_comparation(M_alt)
                W_alt = weight(M_alt_norm)
                alt_validity = validity_check(M_alt, W_alt)
                alt_weights[cname] = W_alt
                alt_details[cname] = {"matrix": M_alt, "weights": W_alt, "validity": alt_validity}

            W_total = np.array([alt_weights[c] for c in CRIT_NAMES])
            W_final = final_weight(W_total, W_crit)

            st.session_state.ahp_result = {
                "M_crit": M_crit, "M_crit_norm": M_crit_norm,
                "W_crit": W_crit, "crit_validity": crit_validity,
                "alt_details": alt_details, "W_final": W_final,
                "W_crit_named": dict(zip(CRIT_NAMES, W_crit)),
            }
            df_init = pd.DataFrame({
                "Nama Restoran": df["Nama Singkat"].values,
                "Skor AHP": W_final,
            }).sort_values("Skor AHP", ascending=False).reset_index(drop=True)
            st.session_state.ranking_list = df_init["Nama Restoran"].tolist()

            st.success("Proses perhitungan AHP selesai!")

    res = st.session_state.ahp_result
    if res is None:
        st.markdown("""
        <div class='warn-box'>
            Proses AHP belum dijalankan. Klik tombol <b>Hitung AHP</b> untuk memulai.
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("### Matriks Kriteria")
        tab1, tab2, tab3 = st.tabs(["Matriks Asli", "Normalisasi", "Bobot & CR"])
        with tab1:
            st.dataframe(pd.DataFrame(res["M_crit"], index=CRIT_NAMES, columns=CRIT_NAMES).round(4), use_container_width=True)
        with tab2:
            st.dataframe(pd.DataFrame(res["M_crit_norm"], index=CRIT_NAMES, columns=CRIT_NAMES).round(4), use_container_width=True)
        with tab3:
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(pd.DataFrame({"Kriteria": CRIT_NAMES, "Bobot": res["W_crit"].round(4)}),
                             use_container_width=True, hide_index=True)
            with col2:
                v = res["crit_validity"]
                st.dataframe(pd.DataFrame({
                    "Parameter": ["Eigen Value (lambda_max)", "CI", "RI", "CR", "Status"],
                    "Nilai": [round(v["eigen"],4), round(v["CI"],4), round(v["RI"],4), round(v["CR"],4),
                              "Konsisten" if v["valid"] else "Tidak Konsisten"]
                }), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("### Matriks Alternatif per Kriteria")
        st.caption("Ditampilkan sebagai sampel. Setiap kriteria menyertakan tabel CV dan status konsistensi.")

        sample_size = 6
        for cname, detail in res["alt_details"].items():
            with st.expander(f"{CRITERIA[cname]['label']}", expanded=False):
                sample_idx = list(range(min(sample_size, len(df))))
                sample_names = df["Nama Singkat"].iloc[sample_idx].tolist()

                st.markdown("**Sampel Matriks Pairwise**")
                M_sample = detail["matrix"][np.ix_(sample_idx, sample_idx)]
                st.dataframe(pd.DataFrame(M_sample, index=sample_names, columns=sample_names).round(3),
                             use_container_width=True)

                st.markdown("**Bobot Alternatif Teratas**")
                top_alt = pd.DataFrame({
                    "Restoran": df["Nama Singkat"], "Bobot": detail["weights"]
                }).sort_values("Bobot", ascending=False).head(5)
                st.dataframe(top_alt.reset_index(drop=True), use_container_width=True, hide_index=True)

                st.markdown("**Tabel CV & Konsistensi**")
                v = detail["validity"]
                cv_df = pd.DataFrame({
                    "Parameter": ["Eigen Value (lambda_max)", "CI", "RI", "CR", "Status"],
                    "Nilai": [round(v["eigen"],4), round(v["CI"],4), round(v["RI"],4), round(v["CR"],4),
                              "Konsisten" if v["valid"] else "Tidak Konsisten"]
                })
                st.dataframe(cv_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("### Bobot Akhir Alternatif (Top 5)")
        df_final5 = pd.DataFrame({
            "Nama Restoran": df["Nama Singkat"].values,
            "Skor AHP": res["W_final"]
        }).sort_values("Skor AHP", ascending=False).head(5)
        st.dataframe(df_final5.reset_index(drop=True), use_container_width=True, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Lihat Hasil Ranking"):
            st.session_state.current_page = "Hasil Ranking"
            st.rerun()

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: HASIL RANKING
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Hasil Ranking":
    st.markdown("""
    <div class='hero-box'>
        <div class='hero-title'>Hasil Ranking Restoran</div>
        <div class='hero-sub'>Menampilkan hasil ranking restoran terbaik berdasarkan metode AHP.</div>
    </div>
    """, unsafe_allow_html=True)

    res = st.session_state.ahp_result
    if res is None:
        st.markdown("""
        <div class='warn-box'>
            Perhitungan AHP belum dijalankan.
            Silakan buka halaman <b>Proses AHP</b> dan tekan <b>Hitung AHP</b>.
        </div>""", unsafe_allow_html=True)
    else:
        df_full = pd.DataFrame({
            "Nama Restoran": df["Nama Singkat"].values,
            "Skor AHP":      res["W_final"],
            "Rating":        df["Rating Toko"].values,
            "Harga":         df["Harga Rata-Rata Makanan di Toko (Rp)"].values,
            "Lokasi":        df["Lokasi Restoran"].values,
            "Entertainment": df["Entertainment"].values,
            "Jenis Suasana": df["Jenis Suasana"].values,
            "Keramaian":     df["Keramaian Restoran"].values,
        }).sort_values("Skor AHP", ascending=False).reset_index(drop=True)

        best = df_full.iloc[0]

        st.markdown("### Kesimpulan Alternatif Terbaik")

        card_style = """
        background: var(--surface);
        border-radius: 14px;
        padding: 20px;
        border: 1px solid var(--border);
        box-shadow: 0 2px 10px rgba(107,60,46,0.07);
        height: 150px;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        """

        b1, b2, b3, b4 = st.columns(4)
        with b1:
            st.markdown(f"""
            <div style="{card_style}">
                <div class='metric-label'>Peringkat Terbaik</div>
                <div class='metric-value' style='font-size:18px;color:#6B3C2E;'>
                    #1 - {best['Nama Restoran']}
                </div>
            </div>
            """, unsafe_allow_html=True)
        with b2:
            st.markdown(f"""
            <div style="{card_style}">
                <div class='metric-label'>Rating</div>
                <div class='metric-value' style='color:#FF3B30;'>{best['Rating']:.1f}</div>
            </div>
            """, unsafe_allow_html=True)
        with b3:
            st.markdown(f"""
            <div style="{card_style}">
                <div class='metric-label'>Harga</div>
                <div class='metric-value' style='font-size:22px;color:#C98D6B;'>
                    Rp {int(best['Harga']):,}
                </div>
            </div>
            """, unsafe_allow_html=True)
        with b4:
            st.markdown(f"""
            <div style="{card_style}">
                <div class='metric-label'>Skor AHP</div>
                <div class='metric-value' style='font-size:22px;color:#6B3C2E;'>
                    {best['Skor AHP']:.6f}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(
            "<div style='margin-top:40px;'></div>",
            unsafe_allow_html=True
        )

        st.markdown("### Visualisasi Skor AHP (Top 10)")
        top_chart = df_full.head(10)
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor("#FFF8F3")
        ax.set_facecolor("#FFF8F3")
        bar_colors = ["#FF3B30" if i < 3 else "#C98D6B" for i in range(len(top_chart))]
        ax.bar(top_chart["Nama Restoran"], top_chart["Skor AHP"], color=bar_colors)
        ax.set_ylabel("Skor AHP", fontsize=10, color="#3A1A10")
        ax.set_xlabel("Restoran", fontsize=10, color="#3A1A10")
        ax.set_title("Top 10 Restoran Terbaik", fontsize=13, fontweight="bold", pad=14, color="#3A1A10")
        ax.tick_params(axis="x", rotation=20, labelsize=9, colors="#3A1A10")
        ax.tick_params(axis="y", colors="#3A1A10")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#D68A6B")
        ax.spines["bottom"].set_color("#D68A6B")
        st.pyplot(fig)
        plt.close()

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("### Tabel Ranking Restoran")

        jumlah_data = st.number_input(
            "Masukkan jumlah data yang ingin ditampilkan:",
            min_value=1,
            value=10,
            step=1
        )

        df_show = df_full.head(int(jumlah_data)).copy()
        df_show.insert(0, "Peringkat", range(1, len(df_show) + 1))

        display_cols = [
            "Peringkat", "Nama Restoran", "Rating", "Harga",
            "Skor AHP", "Lokasi", "Entertainment", "Jenis Suasana", "Keramaian"
        ]
        df_display = df_show[display_cols]

        def highlight_rank(row):
            if row["Peringkat"] == 1:
                return ["background-color: rgba(255,59,48,0.12); font-weight:600"] * len(row)
            elif row["Peringkat"] == 2:
                return ["background-color: rgba(214,138,107,0.10)"] * len(row)
            elif row["Peringkat"] == 3:
                return ["background-color: rgba(201,141,107,0.10)"] * len(row)
            return [""] * len(row)

        styled = df_display.style.apply(highlight_rank, axis=1).format({
            "Skor AHP": "{:.6f}",
            "Harga": "Rp {:,.0f}",
            "Rating": "{:.1f}"
        })

        st.dataframe(styled, use_container_width=True, hide_index=True)

        csv_data = df_display.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Ranking CSV",
            data=csv_data,
            file_name="ranking_restoran_ahp.csv",
            mime="text/csv"
        )

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: PROFIL KELOMPOK
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Profil Kelompok":
    st.markdown("""
    <div class='hero-box'>
        <div class='hero-title'>Profil Kelompok</div>
        <div class='hero-sub'>
            Proyek Akhir Praktikum Sistem Cerdas Pendukung Keputusan (SCPK)
            menggunakan metode Analytical Hierarchy Process (AHP).
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Informasi Proyek")
    ic1, ic2 = st.columns(2)
    with ic1:
        st.markdown("""
        <div class='info-box'>
        <b>Nama Proyek</b><br>
        SPK Rekomendasi Restoran Terbaik di Yogyakarta<br><br>
        <b>Metode</b><br>
        Analytical Hierarchy Process (AHP)
        </div>""", unsafe_allow_html=True)
    with ic2:
        st.markdown("""
        <div class='info-box'>
        <b>Dataset</b><br>
        Places to Eat in the Jogja Region (Kaggle)<br><br>
        <b>Framework</b><br>
        Streamlit &middot; Python &middot; Pandas &middot; Matplotlib
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Anggota Kelompok")

    members = [
        {"emoji": "👩", "name": "Vania Christabella",   "nim": "123240235"},
        {"emoji": "👩", "name": "Jelita Syahwa Nabila", "nim": "123240243"},
        {"emoji": "👩", "name": "Aliffa Izza Ferryna",  "nim": "123240258"},
    ]
    mcols = st.columns(3)
    for col, m in zip(mcols, members):
        with col:
            st.markdown(f"""
            <div class='profile-card'>
                <div class='avatar-ring'>{m['emoji']}</div>
                <div style='font-size:17px;font-weight:700;color:var(--text);margin-top:8px;'>{m['name']}</div>
                <div style='font-size:13px;color:var(--text-muted);margin-top:4px;'>NIM: {m['nim']}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Referensi")
    st.markdown("""
    <div class='info-box'>
    Sasongko et al. (2017).<br>
    <b>Modul Praktikum Kecerdasan Buatan - Analytical Hierarchy Process.</b><br>
    Program Studi Informatika UPN "Veteran" Yogyakarta.<br><br>
    Dataset: <b>Kaggle - Places to Eat in the Jogja Region</b>
    </div>""", unsafe_allow_html=True)