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

The goal is to perform clinical prediction tasks using structured EHR data. We evaluate models such as Multi-Layer Perceptron (MLP), Support Vector Machine (SVM), and our proposed graph-based model (hyEHR) on graph-structured inputs, comparing performance across different random seeds.

---

## 🛠️ Installation

```bash
git clone https://github.com/your-username/clinical-graph-learning.git
cd clinical-graph-learning
conda env create -f hypehr/environment.yml
conda activate hypehr
```

---

## 📂 Project Structure

```bash
.
├── hypehr/                         # Proposed model (hyEHR) implementation
│   ├── convert_datasets_to_pygDataset.py
│   ├── environment.yml            # Conda environment configuration
│   ├── layers.py                  # Custom GNN layers
│   ├── models.py                  # Model architectures
│   ├── preprocessing.py           # Data preprocessing logic
│   └── train.py                   # Training script for hyEHR
│
├── sklearn/                       # Baseline models (Logistic Regression, SVM, etc.)
│   ├── convert_datasets_to_pygDataset.py
│   ├── dataset.py
│   ├── preprocessing.py
│   ├── sklearn                    # Baseline configuration/instructions
│   ├── train.py                   # Training script for baseline models
│   └── try.sh                     # SLURM job submission script
```

---

## 🚀 Usage

### Train hyEHR (graph-based model)

```bash
cd hypehr
python train.py
```

### Train baseline models (MLP, SVM, etc.)

```bash
cd sklearn
bash try.sh
```

You can also run directly with Python:

```bash
python train.py --mlp=True --dname='mimic3' --num_labels=25 --num_nodes=7423 --num_labeled_data=12353 --rand_seed=0
```

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
