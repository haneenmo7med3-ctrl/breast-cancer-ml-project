# 🧬 Breast Cancer Diagnosis — Biotech ML Final Project

A complete end-to-end Machine Learning pipeline to classify breast tumors as **Malignant** or **Benign** using clinical cell nucleus measurements, deployed as a Flask web application.

---

## 📋 Project Overview

| Item | Detail |
|------|--------|
| **Domain** | Biomedical / Oncology |
| **Task** | Binary Classification (Malignant vs Benign) |
| **Dataset** | Breast Cancer Clinical Features (762 clean rows × 20 columns) |
| **Best Model** | Random Forest (Tuned via GridSearchCV) |
| **Precision** | 0.776 |
| **Recall** | 0.846 |
| **AUC-ROC** | 0.850 |

---

## 🔬 Research Questions

1. Can we predict whether a breast tumor is malignant or benign from clinical measurements?
2. Which morphological features are most predictive of malignancy?
3. How does patient age correlate with tumor diagnosis?
4. What is the best-performing ML algorithm for this task?

---

## 📁 Project Structure

```
breast_cancer_project/
│
├── breast_cancer_raw.csv          # Raw (messy) dataset
├── breast_cancer_analysis.ipynb   # Main Jupyter notebook (all steps)
├── app.py                         # Flask web app (deployment)
├── requirements.txt               # Python dependencies
├── README.md                      # This file
│
├── models/
│   ├── best_model.pkl             # Tuned Random Forest
│   ├── scaler.pkl                 # StandardScaler transformer
│   ├── selected_features.pkl      # List of top-12 features
│   └── le_hospital.pkl            # LabelEncoder for hospital
│
└── plots/
    ├── 01_class_dist.png
    ├── 02_histograms.png
    ├── 03_boxplots.png
    ├── 04_heatmap.png
    ├── 05_scatter.png
    ├── 06_violin.png
    ├── 07_hospital_bar.png
    ├── 08_feature_selection.png
    ├── 09_model_comparison.png
    ├── 10_roc.png
    ├── 11_confusion_matrix.png
    └── 12_importances.png
```

---

##  How to Run

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/breast-cancer-biotech-ml.git
cd breast-cancer-biotech-ml
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Jupyter notebook
```bash
jupyter notebook breast_cancer_analysis.ipynb
```

### 4. Launch the web app
```bash
python app.py
```
Then open **http://localhost:5000** in your browser.

---

## 🔄 ML Pipeline Steps

### Step 1 — Project Understanding
- Defined 4 research questions
- Identified dataset domain and target variable

### Step 2 — Data Mining & Processing (Cleaning)
- Fixed mixed-type `radius_mean` column (strings with trailing spaces)
- Standardized 8 inconsistent diagnosis labels → binary 0/1
- Removed 20 duplicate rows
- Imputed 6 columns with missing values using median
- Removed extreme outliers using IQR × 3 method
- Normalized hospital name casing

### Step 3 — Data Exploration (EDA)
Investigated **6+ variables** using both univariate and bivariate analysis:
-  Class distribution (bar + pie)
-  Histograms by diagnosis (6 features)
-  Boxplots by diagnosis (6 features)
-  Correlation heatmap
-  Scatter plot (radius vs area)
-  Violin plot (age by diagnosis)
-  Hospital malignancy stacked bar

### Step 4 — Feature Engineering & Selection
**New features created:**
- `compactness_ratio` = compactness / fractal_dimension
- `area_radius_ratio` = area / radius
- `age_group_encoded` = bucketed age groups

**Feature Selection:** SelectKBest with ANOVA F-scores → top 12 features selected

### Step 4b — Model Training (3 Algorithms)
| Model | Precision | Recall | F1 | AUC-ROC |
|-------|-----------|--------|----|---------|
| Logistic Regression | 0.775 | 0.885 | 0.826 | ~0.84 |
| Random Forest | 0.776 | 0.846 | 0.810 | 0.850 |
| Gradient Boosting | 0.779 | 0.859 | 0.817 | ~0.85 |

### Step 4c — Parameter Tuning
- Used **GridSearchCV** on Random Forest
- Tuned: `n_estimators`, `max_depth`, `min_samples_split`, `max_features`
- 5-fold cross-validation scoring on F1

### Step 5 — Validate & Evaluate
**Evaluation metrics used:**
- ✅ Precision: 0.776 (> 0.3 threshold)
- ✅ Recall: 0.846 (> 0.3 threshold)
- ✅ F1 Score: 0.810
- ✅ AUC-ROC: 0.850
- ✅ Confusion Matrix

### Step 6 — Deployment
Flask web application with:
- Interactive input form for all clinical features
- Real-time prediction with probability bar
- Color-coded malignant/benign result

---

## 📊 Key Findings

1. **Yes** — we can predict tumor malignancy with >85% recall from clinical measurements
2. **Top predictors:** `concavity_mean`, `area_mean`, `radius_mean`, `compactness_ratio`
3. **Age:** Malignant diagnoses skew slightly older, but morphological features dominate
4. **Best model:** Tuned Random Forest (AUC-ROC = 0.850)

---

## Deployment link
(https://breast-cancer-ml-project-ilpzuaukikss8getgys5me.streamlit.app/)

---

*Biotech Machine Learning Course — Final Project*

---

**Student:** Haneen Mohamed Ismail  
**Student ID:** 221000582
