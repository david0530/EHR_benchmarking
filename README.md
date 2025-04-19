# Clinical Graph Learning for Predictive Modeling

This repository provides the implementation for predictive modeling experiments on two clinical datasets — **CRADLE** and **MIMIC-III** — using various machine learning and graph-based techniques.

> **Publication Pending**  
> 📄 If you use this code, please consider citing our work (link to paper coming soon).

---

## 🚑 Datasets

We perform experiments on two benchmark Electronic Health Record (EHR) datasets:

- **MIMIC-III**:  
  - 36,875 total visits  
  - 7,423 unique medical codes  
  - 12,353 labeled visits

- **CRADLE**:  
  - 36,611 total visits  
  - 12,725 unique medical codes  

---

## 🧠 Objective

The goal is to perform clinical prediction tasks using structured EHR data. We evaluate models such as Multi-Layer Perceptron (MLP) and Support Vector Machine (SVM) on graph-structured inputs, comparing performance across different random seeds.

---

## 🛠️ Installation

```bash
git clone https://github.com/your-username/clinical-graph-learning.git
cd clinical-graph-learning
conda create -n clinicalgraph python=3.8
conda activate clinicalgraph
pip install -r requirements.txt
```

---

## 📂 Project Structure

```bash
.
├── convert_datasets_to_pygDataset.py  # Converts raw data into PyG-compatible format
├── dataset.py                         # Defines dataset handling and loading
├── preprocessing.py                   # Preprocessing and feature extraction
├── train.py                           # Training pipeline for MLP and SVM models
├── sklearn/                           # (Optional) Extra scripts for sklearn-based baselines
├── try.sh                             # SLURM script for batch job submission
```

---

## 🚀 Usage

### Run training with MLP:
```bash
python -u train.py \
  --mlp=True \
  --dname='mimic3' \
  --num_labels=25 \
  --num_nodes=7423 \
  --num_labeled_data=12353 \
  --rand_seed=0
```

### Run training with SVM:
```bash
python -u train.py \
  --svm=True \
  --dname='cradle' \
  --num_labels=25 \
  --num_nodes=12725 \
  --num_labeled_data=?? \
  --rand_seed=1
```

*(Replace `??` with the correct number of labeled CRADLE samples if known.)*

---

## 🧪 SLURM Execution

You can submit training jobs via SLURM using the provided shell script:

```bash
bash try.sh
```

Make sure to adjust SLURM directives and parameters inside `try.sh` to match your compute environment.

---

## 📊 Results

### CRADLE Dataset (David's Models)

| Model               | ACC   | AUC   | AUPR  | F1    |
|--------------------|-------|-------|-------|-------|
| Logistic Regression| 0.789 | 0.823 | 0.690 | 0.362 |
| Naive Bayes        | 0.507 | 0.730 | 0.592 | 0.474 |
| Random Forest      | 0.782 | 0.814 | 0.692 | 0.348 |
| XGBoost            | 0.785 | 0.820 | 0.699 | 0.414 |
| SVM                | 0.794 | 0.824 | 0.703 | 0.395 |
| MLP                | 0.753 | 0.783 | 0.609 | 0.475 |
| hyEHR              | 0.804 | 0.839 | 0.733 | 0.444 |

### MIMIC-III Dataset (David's Models)

| Model               | ACC   | AUC   | AUPR  | F1    |
|--------------------|-------|-------|-------|-------|
| Logistic Regression| 0.784 | 0.823 | 0.690 | 0.362 |
| Naive Bayes        | 0.507 | 0.730 | 0.592 | 0.474 |
| Random Forest      | 0.782 | 0.814 | 0.692 | 0.348 |
| XGBoost            | 0.785 | 0.820 | 0.699 | 0.414 |
| SVM                | 0.794 | 0.824 | 0.703 | 0.395 |
| MLP                | 0.753 | 0.783 | 0.609 | 0.475 |
| hyEHR              | 0.804 | 0.839 | 0.733 | 0.444 |

---

## 📎 License

This project is licensed under the MIT License. See `LICENSE` for more details.

---

## 📬 Contact

For questions, please reach out to **[Your Name]** at **[your.email@example.com]**
