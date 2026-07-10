# Integrating the Multidimensional Error Model with GNN and RNN for Continuous Data Prediction

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)
![PyG](https://img.shields.io/badge/PyTorch%20Geometric-GNN-3C2179)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-Academic-lightgrey)

> **Final-year thesis · BSc (Hons) Statistical Computing and Operations Research · Universiti Tunku Abdul Rahman (UTAR)**

**In one line:** A deep learning framework that fuses a **Graph Neural Network** (to capture relational/spatial structure), a **Recurrent Neural Network** (to capture temporal dynamics), and a **Multidimensional Error Model** (to model and correct structured prediction error) — improving accuracy on continuous-value prediction tasks.

---

## Motivation

Real-world continuous data — such as sensor networks, financial series, or spatio-temporal signals — is rarely well described by a single modelling assumption. It carries **relational structure** (variables influence one another), **temporal dependence** (the present depends on the past), and **complex, correlated error** that standard models treat as simple noise.

This thesis asks: *can prediction accuracy be improved by explicitly modelling all three at once?* The proposed architecture integrates three complementary components rather than relying on any single one.

## Approach

| Component | Role | Captures |
|-----------|------|----------|
| **Graph Neural Network (GNN)** | Encodes dependencies between entities as a graph | Relational / spatial structure |
| **Recurrent Neural Network (RNN)** | Models how signals evolve over time | Temporal / sequential dependence |
| **Multidimensional Error Model (MEM)** | Models the structure of residual error across dimensions | Correlated, structured prediction error |

The GNN and RNN are combined into a hybrid spatio-temporal predictor, and the MEM is integrated to refine predictions by accounting for error structure that the neural components alone do not capture. The work includes **feature engineering** and a formal **model validation** protocol to test performance against baselines.

### Headline results

> *Fill in with your actual thesis figures — replace the placeholders below.*

- Improved predictive performance over baseline models — **[e.g. ↓ X% RMSE / ↓ Y% MAE vs. standalone RNN]**.
- Ablation showing the contribution of each component (GNN, RNN, MEM) to the final result.
- Validation on **[your dataset / domain]** across **[metric(s): RMSE, MAE, R², …]**.

---

## Notebooks

The full pipeline is organized as a sequence of notebooks. Each links to its Google Colab version.

> ⚠️ **Action for you:** I couldn't open your Colab files (they require sign-in), so the labels below are a *suggested* pipeline order. Match each of your 10 links to the correct stage and rename accordingly.

| # | Notebook | Purpose | Colab |
|---|----------|---------|-------|
| 01 | `01_data_preprocessing.ipynb` | Load, clean, and structure the raw dataset | [Open](https://colab.research.google.com/drive/1u03zpU9A0sZxCp1IYnxcfYJfpKMBhbB5?usp=sharing) |
| 02 | `02_exploratory_analysis.ipynb` | EDA: distributions, correlations, temporal patterns | [Open](https://colab.research.google.com/drive/1prpOBR6mvu154OIfCY9FM2pCoBpz85B3?usp=sharing) |
| 03 | `03_feature_engineering.ipynb` | Construct features and graph inputs | [Open](https://colab.research.google.com/drive/1hY9bsV3CTVE2f1bs_Ea2RaFQINv0kcTn?usp=sharing) |
| 04 | `04_baseline_models.ipynb` | Baseline predictors for comparison | [Open](https://colab.research.google.com/drive/1ZNTqPjiqDS9hPWM15xlMphQc6sGlRYxt?usp=sharing) |
| 05 | `05_gnn_component.ipynb` | Graph Neural Network module | [Open](https://colab.research.google.com/drive/1yVWE_FuSnor-Fl-qNac56_UW0dtq9iXF?usp=sharing) |
| 06 | `06_rnn_component.ipynb` | Recurrent Neural Network module | [Open](https://colab.research.google.com/drive/12Y3i7lUHquN3ysRgkfff697nMOuU4028?usp=sharing) |
| 07 | `07_mem_integration.ipynb` | Multidimensional Error Model integration | [Open](https://colab.research.google.com/drive/1FjIDdv01kGLbcCBm75ElsRUK9tz2oM8N?usp=sharing) |
| 08 | `08_combined_model.ipynb` | Full GNN + RNN + MEM architecture | [Open](https://colab.research.google.com/drive/14aTuLUi5M0e2BBg-NQDDl6q4lnD1oP4q?usp=sharing) |
| 09 | `09_model_validation.ipynb` | Validation protocol and metric evaluation | [Open](https://colab.research.google.com/drive/1zr6xmQoVQDcWUKIZjU0MpEm26s2RaM-w?usp=sharing) |
| 10 | `10_results_and_ablation.ipynb` | Results, ablation study, and visualizations | [Open](https://colab.research.google.com/drive/1KqaxXjT942W9Q4P4przbkXgSw5293BMM?usp=sharing) |

---

## Repository structure

```
mem-gnn-rnn-continuous-prediction/
├── README.md
├── notebooks/              # The 10 notebooks above, exported from Colab
├── src/                    # (optional) refactored model / util code
│   ├── models/
│   └── utils/
├── data/                   # sample or link to dataset (do not commit large/private data)
├── figures/                # key result plots for the README
└── requirements.txt
```

---

## Tech stack

**Languages & core:** Python, NumPy, pandas
**Deep learning:** PyTorch, PyTorch Geometric (GNN), sequential/recurrent architectures
**Analysis & viz:** scikit-learn, Matplotlib / Seaborn
**Environment:** Google Colab / Jupyter

## How to run

```bash
git clone https://github.com/your-username/mem-gnn-rnn-continuous-prediction.git
cd mem-gnn-rnn-continuous-prediction
pip install -r requirements.txt
# open notebooks/ in Jupyter or Colab and run in order 01 → 10
```

---

## Author

**Tan Xin Ru** — BSc (Hons) Statistical Computing and Operations Research, UTAR

📫 [LinkedIn](https://www.linkedin.com/in/your-linkedin) · ✉️ tanxinru05@gmail.com

> Academic thesis project. Please cite or credit if you build on this work.
