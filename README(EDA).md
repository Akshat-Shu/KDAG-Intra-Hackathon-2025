Skin Lesion Analysis — EDA, Feature Engineering & Clustering

Notebook: Exploratory_Data_Analysis.ipynb

This notebook walks through a full tabular‑and‑image analysis pipeline for skin lesion data. It covers robust data loading/cleaning, exploratory analysis, dimensionality reduction, multiple clustering strategies, engineered features, patient‑level "ugly duckling" signals, and optional image‑feature fusion from both classical CV and a pretrained CNN.


1) Data & Setup

Primary table: subject_data.csv (TBP metrics + clinical fields + target where 1 = malignant, 0 = benign).

Optional image data: downloaded via gdown from Google Drive (folder id used in the notebook). Images are matched to the table by isic_id.

Key libraries

pandas, numpy, matplotlib, seaborn

scikit‑learn (StandardScaler, PCA, t‑SNE, SpectralClustering, KMeans, GaussianMixture, metrics)

umap‑learn, hdbscan

TensorFlow / Keras (for DEC autoencoder & ResNet50 features)

OpenCV (cv2), scikit‑image (Gabor + GLCM), PIL

gdown, tqdm, h5py


Robust CSV loading

Handles problematic rows (e.g., EOF inside string) by falling back to engine='python', low_memory=False.


Light cleaning before analysis

Drops leakage/meta columns when present (e.g., IDs, filenames, prediction outputs from other systems).

Imputes demographics (sex mode, age_approx median).

Consolidates very rare anatom_site_general values into other.

Removes rows with missing target and casts to int.


2) Exploratory Data Analysis (EDA)

Univariate

Distributions for numerical TBP features (histograms + boxplots).

Count plots for key categoricals: sex, anatom_site_general.


Bivariate (label‑aware)

Violin/box/KDE plots comparing benign vs malignant for:

tbp_lv_symm_2axis (asymmetry proxy), tbp_lv_norm_border (border irregularity),

tbp_lv_norm_color (color variation), clin_size_long_diam_mm (diameter),

plus other TBP color deltas/shape metrics.


Malignancy rate by anatomical site (bar chart of % malignant per site).

Malignancy rate by age group (0–20, 21–40, 41–60, 61–80, 81+) — monotonically increasing trend.


Correlations

Heatmap of core TBP numerics (area/perimeter ratio, eccentricity, color std, deltaA/B/L, symmetry, etc.).


“Hard vs Easy” malignant cases (nevi‑confidence slicing)

Defines Hard Malignant: target=1 & tbp_lv_nevi_confidence > 90 (malignant that looks like a nevus).

Easy Malignant: target=1 & tbp_lv_nevi_confidence < 10.

Typical Benign: random benign sample.

KDE overlays show Hard Malignant tends to have higher color variability, less symmetry, and larger size than Typical Benign, but distributions partially overlap with Easy Malignant — useful for error analysis and thresholding.


3) Dimensionality Reduction & Visualization

Standardizes numeric features (StandardScaler).

2‑D embeddings with PCA, t‑SNE, and UMAP.

Two colorings per view:

1. by KMeans labels (quick visual clusters),


2. by ground‑truth target (benign vs malignant).



Also plots malignant‑only embeddings to inspect substructure within cancers.


4) Advanced Clustering (memory‑aware)

Preprocess: scale → PCA to ≤30 components for stability.

Methods run side‑by‑side:

Gaussian Mixture (GMM) — components tuned via BIC over n_components = 2..8.

HDBSCAN — density clusters with min_cluster_size=50, min_samples=10 (labels -1 = noise).

Spectral Clustering — n_clusters=5, affinity='nearest_neighbors', n_neighbors=15.

Deep Embedding Clustering (DEC) — shallow autoencoder (32→8→32), trained on PCA space; apply KMeans (k=5) on encoded features.


Evaluation

Prints silhouette scores per method (when valid).

Writes full assignments to clustering_results_full.csv.


5) Feature Engineering & Ranking

Engineered signals (highlights)

log_lesion_area = log(tbp_lv_areaMM2 + 1)

normalized_lesion_size = clin_size_long_diam_mm / (age_approx + 1e-3)

perimeter_to_area_ratio = tbp_lv_perimeterMM / (tbp_lv_areaMM2 + 1e-3)

area_to_perimeter_ratio = tbp_lv_areaMM2 / (tbp_lv_perimeterMM + 1e-3)

lesion_severity_index = (tbp_lv_norm_border + tbp_lv_norm_color + tbp_lv_eccentricity) / 3

size_age_interaction = clin_size_long_diam_mm * age_approx

shape_color_consistency = tbp_lv_eccentricity * tbp_lv_color_std_mean

index_age_size_symmetry = age_approx * tbp_lv_areaMM2 * tbp_lv_symm_2axis

lesion_visibility_score = tbp_lv_deltaLBnorm + tbp_lv_norm_color

color_asymmetry_index = tbp_lv_symm_2axis * tbp_lv_radial_color_std_max

color_variance_ratio = tbp_lv_color_std_mean / (tbp_lv_stdLExt + 1e-3)

hue_color_std_interaction = tbp_lv_H * tbp_lv_color_std_mean

hue_contrast = |tbp_lv_H - tbp_lv_Hext|

color_contrast_index = tbp_lv_deltaA + tbp_lv_deltaB + tbp_lv_deltaL + tbp_lv_deltaLBnorm

lesion_shape_index = tbp_lv_areaMM2 / (tbp_lv_perimeterMM^2 + 1e-3)


Ranking

Computes mutual information, single‑feature ROC‑AUC, and Pearson r vs target to score features.

Plots: violins (per feature vs target), and a correlation heatmap of the engineered set.


6) Patient‑Level “Ugly Duckling” Signals

For each patient_id, aggregates dispersion of moles: std of color, border, and diameter proxies; and variance of clin_size_long_diam_mm.

Flags has_malignant per patient.

Compares distributions (strip + box plots; diameter variance on log‑scale).

Finding: Patients who have any malignant lesion tend to show higher cross‑mole variability in color/border/diameter — a patient‑level risk cue.


7) How to Reproduce (quick start)

# recommended environment
python -m pip install \
  pandas numpy matplotlib seaborn scikit-learn umap-learn hdbscan \
  tensorflow opencv-python scikit-image pillow gdown tqdm h5py

# run the notebook
jupyter lab  # or jupyter notebook


8) Notes
Some steps are compute‑intensive (t‑SNE/UMAP, Spectral, DEC, ResNet50). They will run on CPU but may take longer.

If CSV parsing errors occur, use: pd.read_csv('subject_data.csv', engine='python', low_memory=False).

For clustering on large data, the notebook reduces dimensionality first (PCA) to keep memory in check.
