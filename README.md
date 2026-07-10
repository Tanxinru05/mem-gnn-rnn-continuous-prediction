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

Concretely, the graph component is instantiated as a **Temporal Graph Attention Network (TGAT)** and the sequential component as a **CNN–BiLSTM–Attention** stack; robustness to structured error is probed through **noise-injection experiments**, and model behaviour is examined with **SHAP interpretability**. The GNN and RNN are combined into a hybrid spatio-temporal predictor, and the MEM is integrated to refine predictions by accounting for error structure that the neural components alone do not capture. The work includes **feature engineering** and a formal **model validation** protocol to test performance against baselines.

### Key outcomes

- The integrated GNN + RNN + MEM architecture **improved predictive performance over baseline models**, validated through a formal model-validation protocol.
- Systematic **feature engineering** and graph construction to capture relational structure in the data.
- Component-wise evaluation demonstrating the contribution of each element (GNN, RNN, MEM) to the final result.
- Full experimental detail, metrics, and figures are in the notebooks below.

---

## Notebooks

The experiments are organized by dataset and purpose. Each notebook renders directly on GitHub, or can be run in Colab.

| Notebook | What it covers | Colab |
|----------|----------------|-------|
| [`Power_Plant__noise_injection_all_models.ipynb`](./Power_Plant__noise_injection_all_models.ipynb) | **Full model suite** on the UCI Combined Cycle Power Plant dataset — CNN–BiLSTM–Attention, TGAT, and baselines — with **noise-injection experiments** testing robustness under structured error | [Open](https://colab.research.google.com/github/Tanxinru05/mem-gnn-rnn-continuous-prediction/blob/main/Power_Plant__noise_injection_all_models.ipynb) |
| [`power_plant_tgat_only.ipynb`](./power_plant_tgat_only.ipynb) | **TGAT in isolation** — the Temporal Graph Attention component evaluated on its own (Power Plant) | [Open](https://colab.research.google.com/github/Tanxinru05/mem-gnn-rnn-continuous-prediction/blob/main/power_plant_tgat_only.ipynb) |
| [`Interpretability_for_Power_Plant_DataSet.ipynb`](./Interpretability_for_Power_Plant_DataSet.ipynb) | **Interpretability** — SHAP-based analysis of what drives the trained models' predictions (Power Plant) | [Open](https://colab.research.google.com/github/Tanxinru05/mem-gnn-rnn-continuous-prediction/blob/main/Interpretability_for_Power_Plant_DataSet.ipynb) |
| [`CNN+BILSTM+Attn__Beijing_PM2_5.ipynb`](./CNN%2BBILSTM%2BAttn__Beijing_PM2_5.ipynb) | **CNN–BiLSTM–Attention** applied to the Beijing PM2.5 air-pollution dataset | [Open](https://colab.research.google.com/github/Tanxinru05/mem-gnn-rnn-continuous-prediction/blob/main/CNN%2BBILSTM%2BAttn__Beijing_PM2_5.ipynb) |
| [`Pollution_baselines_2.ipynb`](./Pollution_baselines_2.ipynb) | **Baseline models** on the pollution dataset, for comparison against the deep architectures | [Open](https://colab.research.google.com/github/Tanxinru05/mem-gnn-rnn-continuous-prediction/blob/main/Pollution_baselines_2.ipynb) |
| [`Bike Sharing Dataset_all_models.ipynb`](./Bike%20Sharing%20Dataset_all_models.ipynb) | **Full model suite** on the Bike Sharing dataset — testing generalization to a third domain | [Open](https://colab.research.google.com/github/Tanxinru05/mem-gnn-rnn-continuous-prediction/blob/main/Bike%20Sharing%20Dataset_all_models.ipynb) |

**Datasets:** UCI Combined Cycle Power Plant · Beijing PM2.5 · Bike Sharing — three continuous-prediction benchmarks with different relational and temporal characteristics.

## Repository structure

```
mem-gnn-rnn-continuous-prediction/
├── README.md
├── *.ipynb                 # experiment notebooks (see table above)
├── data/                   # dataset notes / samples
├── figures/                # key result plots
├── src/                    # (optional) refactored model & utility code
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
# open any notebook in Jupyter, or click its Colab badge to run in the cloud
```

---

## Author

**Tan Xin Ru** — BSc (Hons) Statistical Computing and Operations Research, UTAR

📫 [LinkedIn](https://www.linkedin.com/in/your-linkedin) · ✉️ tanxinru05@gmail.com

> Academic thesis project. Please cite or credit if you build on this work.
