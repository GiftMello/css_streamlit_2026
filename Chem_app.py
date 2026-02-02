# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 09:35:59 2026

@author: MelloG
"""

import streamlit as st
import pandas as pd
import numpy as np

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="ATR-FTIR Water Quality Research App",
    layout="wide"
)

# =========================================================
# DATA
# =========================================================
heavy_metal_data = pd.DataFrame({
    "Sample ID": ["IRR-01", "IRR-02", "MIN-01", "RIV-01", "DAM-01"],
    "Pb (ppm)": [0.05, 0.12, 0.30, 0.08, 0.15],
    "Cr (ppm)": [0.02, 0.06, 0.18, 0.05, 0.09],
    "Zn (ppm)": [0.50, 0.75, 1.20, 0.65, 0.90],
    "Fe (ppm)": [1.8, 2.4, 5.6, 2.1, 3.2],
    "As (ppm)": [0.01, 0.03, 0.07, 0.02, 0.04],
    "Source Type": ["Irrigation", "Irrigation", "Mining", "River", "Dam"]
})

ftir_spectra = pd.DataFrame({
    "3400 cm⁻¹": [0.82, 0.76, 0.91, 0.80, 0.85],
    "2920 cm⁻¹": [0.40, 0.35, 0.50, 0.38, 0.42],
    "1630 cm⁻¹": [0.65, 0.60, 0.78, 0.63, 0.70],
    "1030 cm⁻¹": [0.70, 0.68, 0.88, 0.72, 0.75],
    "520 cm⁻¹":  [0.30, 0.28, 0.45, 0.32, 0.34]
}, index=heavy_metal_data["Sample ID"])

# =========================================================
# PAGE FUNCTIONS
# =========================================================
def researcher_profile():
    st.title("👩🏽‍🔬 Researcher Profile")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.write("**Name:** Gift Mello")
        st.write("**Field:** Environmental Chemistry & Spectroscopy")
        st.write("**Institution:** University of Limpopo")

        st.markdown("""
        **Research Focus**
        - ATR-FTIR spectroscopy  
        - Heavy metal contamination in water  
        - Chemometric data analysis  
        - Irrigation and environmental water sources  
        """)


def research_data():
    st.title("💧 Research Data")

    

    st.subheader("Heavy Metal Concentrations (ppm)")
    st.dataframe(heavy_metal_data, use_container_width=True)

    metal = st.selectbox(
        "Select metal",
        ["Pb (ppm)", "Cr (ppm)", "Zn (ppm)", "Fe (ppm)", "As (ppm)"]
    )

    st.bar_chart(
        heavy_metal_data.set_index("Sample ID")[metal]
    )

    st.subheader("ATR-FTIR Spectral Features")
    st.dataframe(ftir_spectra, use_container_width=True)


def chemometric_analysis():
    st.title("📈 Chemometric Analysis")


    analysis = st.radio(
        "Select analysis",
        ["PCA (Exploratory)", "Calibration Demo"]
    )

    if analysis == "PCA (Exploratory)":
        X = ftir_spectra.values
        Xc = X - np.mean(X, axis=0)
        cov = np.cov(Xc, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)

        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        scores = Xc @ eigvecs[:, :2]
        var = eigvals / np.sum(eigvals)

        df_scores = pd.DataFrame(
            scores, columns=["PC1", "PC2"], index=ftir_spectra.index
        )

        st.dataframe(df_scores)
        st.bar_chart(df_scores)
        st.info(
            f"Explained variance → PC1: {var[0]*100:.2f}% | "
            f"PC2: {var[1]*100:.2f}%"
        )

    else:
        target = st.selectbox(
            "Target metal",
            ["Pb (ppm)", "Cr (ppm)", "Zn (ppm)", "Fe (ppm)", "As (ppm)"]
        )

        X = ftir_spectra.mean(axis=1).values
        y = heavy_metal_data[target].values

        coeff = np.polyfit(X, y, 1)
        y_pred = np.polyval(coeff, X)

        results = pd.DataFrame(
            {"Measured": y, "Predicted": y_pred},
            index=ftir_spectra.index
        )

        st.dataframe(results)
        st.line_chart(results)

        rmse = np.sqrt(np.mean((y - y_pred)**2))
        st.success(f"RMSE = {rmse:.3f} ppm")


def publications():
    st.title("📚 Publications")

    uploaded = st.file_uploader("Upload CSV", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df)


def contact():
    st.title("📬 Contact")


    st.write("📧 201912820@myturf.ul.ac.za")
    st.write("Research Area: ATR-FTIR • Water Quality • Chemometrics")

# =========================================================
# SIDEBAR NAVIGATION CONTROLLER
# =========================================================
st.sidebar.title("🔍 Navigation")

page = st.sidebar.radio(
    "Go to",
    (
        "Researcher Profile",
        "Research Data",
        "Chemometric Analysis",
        "Publications",
        "Contact"
    )
)

# =========================================================
# PAGE ROUTING
# =========================================================
if page == "Researcher Profile":
    researcher_profile()
elif page == "Research Data":
    research_data()
elif page == "Chemometric Analysis":
    chemometric_analysis()
elif page == "Publications":
    publications()
elif page == "Contact":
    contact()


