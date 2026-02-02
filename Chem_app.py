# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 09:35:59 2026

@author: MelloG
"""

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, mean_squared_error

# =========================================================
# APP CONFIG
# =========================================================
st.set_page_config(
    page_title="ATR-FTIR Water Quality Dashboard",
    layout="wide"
)

# =========================================================
# TITLE & CONTEXT
# =========================================================
st.title("ATR-FTIR Spectroscopy for Water Quality Assessment")

st.caption(
    "Investigating the potential of spectroscopic techniques for testing irrigation water "
    "from selected water sources in South Africa"
)

# =========================================================
# RESEARCHER PROFILE
# =========================================================
st.header("Researcher Overview")

name = "Gift Mello"
field = "Environmental Chemistry & Spectroscopy"
institution = "University of Limpopo"

st.write(f"**Name:** {name}")
st.write(f"**Field of Research:** {field}")
st.write(f"**Institution:** {institution}")

st.image(
    "https://cdn.pixabay.com/photo/2017/08/01/00/18/water-2563199_1280.jpg",
    caption="Water quality monitoring (Pixabay)",
    use_column_width=True
)

# =========================================================
# RESEARCH DESCRIPTION
# =========================================================
st.header("Research Focus")

st.markdown("""
This research evaluates the potential of **ATR-FTIR spectroscopy combined with chemometric techniques**
for **rapid assessment of irrigation and environmental water quality**.

**Key elements include:**
- Detection of heavy metals using spectral fingerprints
- Exploratory data analysis using PCA
- Calibration modelling using PLS regression
- Application to South African water sources
""")

# =========================================================
# HEAVY METAL DATA (ppm)
# =========================================================
st.header("Water Quality Dataset")

heavy_metal_data = pd.DataFrame({
    "Sample ID": ["IRR-01", "IRR-02", "MIN-01", "RIV-01", "DAM-01"],
    "Pb (ppm)": [0.05, 0.12, 0.30, 0.08, 0.15],
    "Cr (ppm)": [0.02, 0.06, 0.18, 0.05, 0.09],
    "Zn (ppm)": [0.50, 0.75, 1.20, 0.65, 0.90],
    "Fe (ppm)": [1.8, 2.4, 5.6, 2.1, 3.2],
    "As (ppm)": [0.01, 0.03, 0.07, 0.02, 0.04],
    "Source Type": ["Irrigation", "Irrigation", "Mining", "River", "Dam"]
})

st.dataframe(heavy_metal_data, use_container_width=True)

# =========================================================
# WHO / SA GUIDELINE VALUES
# =========================================================
st.subheader("Reference Guideline Limits (ppm)")

guidelines = pd.DataFrame({
    "Metal": ["Pb", "Cr", "Zn", "Fe", "As"],
    "Guideline Limit (ppm)": [0.01, 0.05, 3.0, 2.0, 0.01]
})

st.dataframe(guidelines)

# =========================================================
# ATR-FTIR SPECTRAL DATA
# =========================================================
st.header("ATR-FTIR Spectral Features")

ftir_spectra = pd.DataFrame({
    "3400 cm⁻¹": [0.82, 0.76, 0.91, 0.80, 0.85],
    "2920 cm⁻¹": [0.40, 0.35, 0.50, 0.38, 0.42],
    "1630 cm⁻¹": [0.65, 0.60, 0.78, 0.63, 0.70],
    "1030 cm⁻¹": [0.70, 0.68, 0.88, 0.72, 0.75],
    "520 cm⁻¹":  [0.30, 0.28, 0.45, 0.32, 0.34]
}, index=heavy_metal_data["Sample ID"])

st.dataframe(ftir_spectra, use_container_width=True)

# =========================================================
# DATA VISUALIZATION
# =========================================================
st.header("Heavy Metal Concentration Visualization")

selected_metal = st.selectbox(
    "Select metal",
    ["Pb (ppm)", "Cr (ppm)", "Zn (ppm)", "Fe (ppm)", "As (ppm)"]
)

st.bar_chart(
    heavy_metal_data.set_index("Sample ID")[selected_metal]
)

# =========================================================
# CHEMOMETRICS SECTION
# =========================================================
st.header("Chemometric Analysis")

analysis_type = st.radio(
    "Choose analysis method",
    ["Principal Component Analysis (PCA)", "PLS Regression Calibration"]
)

# ---------------- PCA ----------------
if analysis_type == "Principal Component Analysis (PCA)":

    st.subheader("PCA of ATR-FTIR Spectra")

    st.markdown("""
    **Objective:**  
    - Reduce spectral dimensionality  
    - Identify clustering and variance patterns  
    - Explore contamination-related spectral differences
    """)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(ftir_spectra)

    pca = PCA(n_components=2)
    scores = pca.fit_transform(X_scaled)

    pca_scores = pd.DataFrame(
        scores,
        columns=["PC1", "PC2"],
        index=ftir_spectra.index
    )

    st.dataframe(pca_scores)

    st.bar_chart(pca_scores)

    st.write(
        f"**Explained Variance:** "
        f"PC1 = {pca.explained_variance_ratio_[0]*100:.2f}% | "
        f"PC2 = {pca.explained_variance_ratio_[1]*100:.2f}%"
    )

# ---------------- PLS ----------------
else:
    st.subheader("PLS Regression: FTIR vs Heavy Metals")

    st.markdown("""
    **Objective:**  
    - Develop calibration models  
    - Predict heavy metal concentrations from FTIR spectra  
    - Demonstrate rapid screening potential
    """)

    target = st.selectbox(
        "Select target metal",
        ["Pb (ppm)", "Cr (ppm)", "Zn (ppm)", "Fe (ppm)", "As (ppm)"]
    )

    X = ftir_spectra.values
    y = heavy_metal_data[target].values

    pls = PLSRegression(n_components=2)
    pls.fit(X, y)

    y_pred = pls.predict(X).ravel()

    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    results = pd.DataFrame({
        "Measured (ppm)": y,
        "Predicted (ppm)": y_pred
    }, index=ftir_spectra.index)

    st.dataframe(results)

    st.line_chart(results)

    st.success(f"Model Performance → R² = {r2:.3f} | RMSE = {rmse:.3f}")

    st.caption(
        "This calibration is demonstrative. "
        "Robust models require independent validation and larger datasets."
    )

# =========================================================
# CONCLUSION
# =========================================================
st.header("Conclusion")

st.markdown("""
This application demonstrates the feasibility of integrating **ATR-FTIR spectroscopy**
with **chemometric modelling** for the assessment of water quality.
The approach supports **rapid, non-destructive screening** of irrigation water,
with potential application in environmental monitoring and decision support.
""")

# =========================================================
# CONTACT
# =========================================================
st.header("Contact Information")
st.write("📧 gift.mello@university.ac.za")
