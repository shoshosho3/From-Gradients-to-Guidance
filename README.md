# MINDS: An Active Learning Strategy via Minimum-Impact Dissimilarity Sampling

**Team Members:**
*   Ben Hellmann (ben.hellmann@campus.technion.ac.il)
*   Tomer Katz (katztomer@campus.technion.ac.il)
*   Gilad Karpel (gilad.karpel@campus.technion.ac.il)
*   Anis Barhoum (anis.barhoum@campus.technion.ac.il)

**Course:** Data Analysis and Presentation – Final Project

---

## 1. Project Overview

This project introduces MINDS (Minimum-Impact Dissimilarity Sampling), a novel active learning strategy. The core idea is to identify the "least informative" sample in a newly labeled batch (the one with the minimum impact on model parameters) and then select the next batch by choosing unlabeled points that are most *dissimilar* to it. This promotes exploration of new, informative regions of the feature space.

We evaluate MINDS against standard active learning baselines (Random, Uncertainty, and Diversity sampling) on image, text, and tabular datasets.

## 2. Repository Structure

```
minds-active-learning/
├── .gitignore
├── README.md
├── requirements.txt
├── configs/              # Experiment configuration files
├── data/                 # Local data storage (not tracked by Git)
├── notebooks/            # Notebooks for exploration and analysis
├── report/               # Final report and presentation slides
├── results/              # Experiment outputs (logs, plots, models)
└── src/                  # All project source code
```

## 3. Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/<your-username>/minds-active-learning.git
    cd minds-active-learning
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download datasets:**
    *The datasets (e.g., CIFAR-10, SVHN) will be downloaded automatically by the PyTorch scripts on the first run. For other datasets like AG News, you can run the download script:*
    ```bash
    # (Optional) Add a script for custom datasets if needed
    # sh data/get_data.sh
    ```


## 4. How to Run Experiments

The main training and evaluation loop is handled by a central script. You can run an experiment by specifying a configuration file.

**Example: Running MINDS on CIFAR-10**

```bash
python -m src.main --config configs/experiment_cifar10_minds.yaml
```

This will:
1.  Load the CIFAR-10 dataset.
2.  Initialize a lightweight CNN model.
3.  Run the active learning loop using the MINDS strategy.
4.  Save logs, model checkpoints, and evaluation plots to the `results/` directory.

To run a different experiment, simply point to another config file.

---