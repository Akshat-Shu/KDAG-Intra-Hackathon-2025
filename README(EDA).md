# Skin Lesion Analysis — EDA, Feature Engineering & Clustering

## Notebook: stuffintras.ipynb
This repository has a ipynb file for skin lesion analysis. The work covers exploratory analysis, dimensionality reduction, clustering, engineered feature design, patient‑level signal extraction.

### 1) Data
Primary table: subject_data.csv
Contains clinical features (age, sex, site), and the target label (0 = benign, 1 = malignant).

Key libraries:
- Data handling: pandas, numpy
- Visualization: matplotlib, seaborn
- ML/Clustering: scikit‑learn, umap-learn


### 2) Robust CSV Loading

- Handles problematic rows (e.g., EOF inside string) by using engine='python', low_memory=False.
- Ensures large CSVs (>250 MB) are parsed without truncation.



### 3) Exploratory Data Analysis (EDA)

Univariate Distributions-
- Histograms & boxplots for numerical TBP features.
- Count plots for categorical fields: sex, anatom_site_general.

Bivariate (Benign vs Malignant)-
- Violin/box/KDE plots for:
- Symmetry (tbp_lv_symm_2axis)
- Border irregularity (tbp_lv_norm_border)
- Color variation (tbp_lv_norm_color)
- Lesion size (clin_size_long_diam_mm)
- Malignancy rate by anatomical site.
- Malignancy rate by age group (0–20, 21–40, 41–60, 61–80, 81+).

Correlations-
- Heatmap of TBP numeric features.
Highlights strong relationships between area/perimeter ratios, eccentricity, and color metrics.
- Hard vs Easy Malignant Cases

Defined using tbp_lv_nevi_confidence:
- Hard Malignant: looks like a nevus but is malignant.
- Easy Malignant: clear malignant patterns.
- Typical Benign: benign control group.

KDE overlays reveal that hard cases often overlap with benigns in symmetry/size, while easy cases are more distinct.



### 3) Dimensionality Reduction & Visualization

Standardizes numeric features.

Produces 2D embeddings with:
- PCA
- t‑SNE
- UMAP

Plots embeddings twice:
- Colored by KMeans clusters
- Colored by ground truth (benign vs malignant)

Also creates malignant‑only embeddings to explore subtypes within cancers.




### 4) Feature Engineering & Ranking

Selected Features-

- log_lesion_area = log(tbp_lv_areaMM2 + 1)
- normalized_lesion_size = clin_size_long_diam_mm / (age_approx + 1e-3)
- perimeter_to_area_ratio = tbp_lv_perimeterMM / tbp_lv_areaMM2
- area_to_perimeter_ratio = tbp_lv_areaMM2 / tbp_lv_perimeterMM
- lesion_severity_index = mean(border, color, eccentricity)
- size_age_interaction = size * age
- shape_color_consistency = eccentricity * color_std
- index_age_size_symmetry = age * area * symmetry
- lesion_visibility_score = deltaLBnorm + norm_color
- color_asymmetry_index = symmetry * radial_color_std
- color_variance_ratio = color_std / stdLExt
- hue_color_std_interaction = hue * color_std
- hue_contrast = |H − Hext|
- color_contrast_index = deltaA + deltaB + deltaL + deltaLBnorm
- lesion_shape_index = area / perimeter²

Ranking Approach-
- Mutual Information
- Single‑feature ROC‑AUC
- Pearson correlation

Plots:
- Violin plots (feature vs target)
- Correlation heatmap of engineered features



### 5) Patient‑Level “Ugly Duckling” Signals

Aggregates features per patient_id.
Measures cross‑mole variability in:
- Color
- Border
- Diameter

Flags has_malignant per patient.

Finding: Patients with at least one malignant lesion show higher variability across moles — a potential patient‑level risk indicator.



### 6) How to Reproduce

Install requirements-
`pip install pandas numpy matplotlib seaborn scikit-learn umap-learn`

Can run the notebook on colab/jupyter enviornment.

### 7) Notes
t‑SNE, UMAP, Spectral, and DEC are compute‑intensive. They run on CPU but may be slow.

For large CSVs with parse errors, use:
pd.read_csv('subject_data.csv', engine='python', low_memory=False)

PCA preprocessing is used to keep clustering memory‑efficient.