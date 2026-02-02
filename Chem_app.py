# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 09:35:59 2026

@author: MelloG
"""

import streamlit as st
import pandas as pd
import numpy as np

# =========================================================
# APP CONFIG
# =========================================================
st.set_page_config(
    page_title="ATR-FTIR Water Quality Research App",
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
def numpy_pca(X, n_components=2):
    # Mean center
    X_centered = X - np.mean(X, axis=0)

    # Covariance matrix
    cov_matrix = np.cov(X_centered, rowvar=False)

    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort descending
    sorted_idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_idx]
    eigenvectors = eigenvectors[:, sorted_idx]

    # Select components
    components = eigenvectors[:, :n_components]
    scores = X_centered @ components

    explained_variance = eigenvalues / np.sum(eigenvalues)

    return scores, explained_variance[:n_components]

if analysis_type == "Principal Component Analysis (PCA)":

    st.subheader("PCA of ATR-FTIR Spectra (NumPy Implementation)")

    X = ftir_spectra.values
    scores, var = numpy_pca(X, n_components=2)

    pca_scores = pd.DataFrame(
        scores,
        columns=["PC1", "PC2"],
        index=ftir_spectra.index
    )

    st.dataframe(pca_scores)
    st.bar_chart(pca_scores)

    st.write(
        f"**Explained Variance:** "
        f"PC1 = {var[0]*100:.2f}% | PC2 = {var[1]*100:.2f}%"
    )


# ---------------- PLS ----------------
else:
    st.subheader("FTIR-Based Calibration (Linear Demonstration)")

    target = st.selectbox(
        "Select target metal",
        ["Pb (ppm)", "Cr (ppm)", "Zn (ppm)", "Fe (ppm)", "As (ppm)"]
    )

    X = ftir_spectra.values
    y = heavy_metal_data[target].values

    # Simple linear regression using numpy
    X_aug = np.c_[np.ones(X.shape[0]), X.mean(axis=1)]
    coeffs = np.linalg.lstsq(X_aug, y, rcond=None)[0]

    y_pred = X_aug @ coeffs

    results = pd.DataFrame({
        "Measured (ppm)": y,
        "Predicted (ppm)": y_pred
    }, index=ftir_spectra.index)

    st.dataframe(results)
    st.line_chart(results)

    rmse = np.sqrt(np.mean((y - y_pred)**2))
    st.success(f"Calibration RMSE = {rmse:.3f} ppm")

    st.caption(
        "Demonstration calibration using FTIR spectral intensity averages. "
        "Advanced PLS models require external libraries and validation data."
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
st.write("📧 201912820@myturf.ul.ac.za")


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="ATR-FTIR Water Quality Research App",
    layout="wide"
)

# =========================================================
# SIDEBAR NAVIGATION
# =========================================================
st.sidebar.title("🔍 Navigation")
menu = st.sidebar.radio(
    "Select Section",
    (
        "Researcher Profile",
        "Research Data",
        "Chemometric Analysis",
        "Publications",
        "Contact"
    )
)

st.sidebar.markdown("---")
st.sidebar.caption("ATR-FTIR | Water Quality | South Africa")

# =========================================================
# PRELIMINARY DATA
# =========================================================

# Heavy metal concentrations (ppm)
heavy_metal_data = pd.DataFrame({
    "Sample ID": ["IRR-01", "IRR-02", "MIN-01", "RIV-01", "DAM-01"],
    "Pb (ppm)": [0.05, 0.12, 0.30, 0.08, 0.15],
    "Cr (ppm)": [0.02, 0.06, 0.18, 0.05, 0.09],
    "Zn (ppm)": [0.50, 0.75, 1.20, 0.65, 0.90],
    "Fe (ppm)": [1.8, 2.4, 5.6, 2.1, 3.2],
    "As (ppm)": [0.01, 0.03, 0.07, 0.02, 0.04],
    "Source Type": ["Irrigation", "Irrigation", "Mining", "River", "Dam"]
})

# ATR-FTIR absorbance features
ftir_spectra = pd.DataFrame({
    "3400 cm⁻¹ (O–H)": [0.82, 0.76, 0.91, 0.80, 0.85],
    "2920 cm⁻¹ (C–H)": [0.40, 0.35, 0.50, 0.38, 0.42],
    "1630 cm⁻¹ (H–O–H)": [0.65, 0.60, 0.78, 0.63, 0.70],
    "1030 cm⁻¹ (C–O/Si–O)": [0.70, 0.68, 0.88, 0.72, 0.75],
    "520 cm⁻¹ (Metal–O)": [0.30, 0.28, 0.45, 0.32, 0.34]
}, index=heavy_metal_data["Sample ID"])

# =========================================================
# RESEARCHER PROFILE
# =========================================================
if menu == "Researcher Profile":

    st.title("👩🏽‍🔬 Researcher Profile")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.write("**Name:** Gift Mello")
        st.write("**Field:** Environmental Chemistry & Spectroscopy")
        st.write("**Institution:** University of [Your Institution]")

        st.markdown("""
        **Research Focus**
        - ATR-FTIR spectroscopy  
        - Heavy metal contamination in water  
        - Chemometric data analysis  
        - Irrigation and environmental water sources  
        """)

    with col2:
        st.image(
            "https://images.unsplash.com/photo-1501004318641-b39e6451bec6",
            caption="Water quality research",
            use_column_width=True
        )

# =========================================================
# RESEARCH DATA
# =========================================================
elif menu == "Research Data":

    st.title("💧 Water Quality Data")

    st.image(
        "https://images.unsplash.com/photo-1509395176047-4a66953fd231",
        caption="Sampling of environmental water sources",
        use_column_width=True
    )

    st.subheader("Heavy Metal Concentrations (ppm)")
    st.dataframe(heavy_metal_data, use_container_width=True)

    metal = st.selectbox(
        "Select metal to visualize",
        ["Pb (ppm)", "Cr (ppm)", "Zn (ppm)", "Fe (ppm)", "As (ppm)"]
    )

    st.bar_chart(
        heavy_metal_data.set_index("Sample ID")[metal]
    )

    st.subheader("ATR-FTIR Spectral Features")
    st.dataframe(ftir_spectra, use_container_width=True)

# =========================================================
# CHEMOMETRIC ANALYSIS 
# =========================================================
elif menu == "Chemometric Analysis":

    st.title("📈 Chemometric Analysis")

    st.image(
        "https://images.unsplash.com/photo-1581092160562-40aa08e78837",
        caption="Spectroscopic and data analysis workflow",
        use_column_width=True
    )

    analysis = st.radio(
        "Choose analysis type",
        ["PCA (Exploratory)", "Calibration Demo"]
    )

    # ---------- PCA ----------
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

        pca_scores = pd.DataFrame(
            scores, columns=["PC1", "PC2"], index=ftir_spectra.index
        )

        st.subheader("PCA Scores")
        st.dataframe(pca_scores)
        st.bar_chart(pca_scores)

        st.info(
            f"Explained Variance → PC1: {var[0]*100:.2f}% | PC2: {var[1]*100:.2f}%"
        )

    # ---------- CALIBRATION ----------
    else:
        target = st.selectbox(
            "Target metal",
            ["Pb (ppm)", "Cr (ppm)", "Zn (ppm)", "Fe (ppm)", "As (ppm)"]
        )

        X = ftir_spectra.mean(axis=1).values
        y = heavy_metal_data[target].values

        coeff = np.polyfit(X, y, 1)
        y_pred = np.polyval(coeff, X)

        results = pd.DataFrame({
            "Measured (ppm)": y,
            "Predicted (ppm)": y_pred
        }, index=ftir_spectra.index)

        st.subheader("Calibration Results")
        st.dataframe(results)
        st.line_chart(results)

        rmse = np.sqrt(np.mean((y - y_pred)**2))
        st.success(f"RMSE = {rmse:.3f} ppm")

# =========================================================
# PUBLICATIONS
# =========================================================
elif menu == "Publications":

    st.title("📚 Publications & Outputs")

    st.image(
        "https://images.unsplash.com/photo-1517976487492-5750f3195933",
        caption="Scientific research and publication",
        use_column_width=True
    )

    uploaded = st.file_uploader("Upload CSV", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df)

# =========================================================
# CONTACT
# =========================================================
elif menu == "Contact":

    st.title("📬 Contact")

    st.image(
        "https://images.unsplash.com/photo-1521791136064-7986c2920216",
        caption="Academic collaboration",
        use_column_width=True
    )

    st.write("**Email:** gift.mello@university.ac.za")
    st.write("**Research Area:** ATR-FTIR • Water Quality • Chemometrics")

