# An Empirical Study of EGL-based Active Learning Strategies

**Course:** Data Analysis and Presentation – Final Project

**Team Members:**
*   Ben Hellmann (ben.hellmann@campus.technion.ac.il)
*   Tomer Katz (katztomer@campus.technion.ac.il)
*   Gilad Karpel (gilad.karpel@campus.technion.ac.il)
*   Anis Barhoum (anis.barhoum@campus.technion.ac.il)

## Project Overview

This project conducts a comprehensive empirical study of **Expected Gradient Length (EGL)**, a modern family of active learning strategies. Built as an extension to the excellent [DeepAL+ toolkit](https://github.com/ant-lab/DeepAL_plus), our work introduces a modular, robust, and reproducible framework for evaluating gradient-based methods against traditional baselines.

Our core contributions include the implementation of several novel strategies. We introduce **AdvancedEGL**, which modulates gradient scores with predictive uncertainty, and **DiversityEGL**, which combines informativeness with sample diversity in a two-stage selection process. Furthermore, we propose a novel and efficient approach called **Learned EGL (LEGL)**, which trains a secondary model head to predict gradient norms directly, accelerating the querying process by avoiding expensive gradient computations. Our findings demonstrate that EGL-based methods consistently and significantly outperform traditional uncertainty-based strategies, establishing gradient information as a powerful heuristic for deep active learning.

## Core Contributions

The heart of this project is a set of new, modular Python files that implement the EGL strategies and the necessary backend infrastructure. These files were created for this project and are not part of the original DeepAL+ repository.

```
.
├── analyze_results.py          # Script to analyze and plot results
├── updated_environment.yml             # Conda environment file
├── run_experiments.sh          # Master script to run all experiments
├── deepalplus/                 # Original DeepAL+ codebase with our modifications
│   ├── query_strategies/       # Directory containing our new strategy files
│   │   ├── vanilla_egl.py
│   │   ├── advanced_egl.py
│   │   ├── advanced_legl.py
│   │   ├── base_handler.py
│   │   ├── base_egl_handler.py
│   │   ├── common_nets.py
│   │   ├── common_strategies.py
│   │   └── egl_entropy_utils.py
│   └── new_results/            # (Generated) Directory for experiment log files
└── ...                         # Other DeepAL+ files
```

*   **Novel Strategy Implementations**:
    *   `vanilla_egl.py`: The foundational Expected Gradient Length strategy.
    *   `advanced_egl.py`: Implements `AdvancedEGL` (EGL + uncertainty) and `DiversityEGL` (EGL + diversity).
    *   `advanced_legl.py`: Implements `AdvancedLEGL` and `DiversityLEGL` that learn to *predict* gradient norms for faster querying.
*   **Modular Backend for Extensibility**:
    *   `base_handler.py` & `base_egl_handler.py`: Abstract network handlers that create a clean structure for training and querying.
    *   `common_nets.py`: A refactored network module providing a standardized ResNet18 architecture.
    *   `common_strategies.py`: A base class for creating two-stage diversity-aware strategies.
*   **Reproducibility Framework**:
    *   `run_experiments.sh`: A shell script that automates the entire experimental suite across all strategies, datasets, and random seeds.
    *   `analyze_results.py`: A Python script to parse raw log files, calculate AUBC and final accuracy, generate a summary table, and plot learning curves.

## Results

### Summary Table

Performance averaged over 3 random seeds. **AUBC** (Area Under the Budget-Accuracy Curve) measures learning efficiency, and **F-acc** is the final accuracy.

```
--- Active Learning Results Summary ---
dataset         CIFAR10        FashionMNIST         MNIST          SVHN
                   AUBC  F-acc         AUBC  F-acc   AUBC  F-acc   AUBC  F-acc
method
AdvancedEGL      0.6438 0.7519       0.8417 0.8757 0.9804 0.9917 0.7783 0.8666
AdvancedLEGL     0.6190 0.7242       0.8318 0.8649 0.9754 0.9922 0.7537 0.8480
DiversityEGL     0.6389 0.7656       0.8432 0.8780 0.9823 0.9932 0.7738 0.8658
DiversityLEGL    0.6454 0.7710       0.8375 0.8669 0.9787 0.9925 0.7713 0.8601
EntropySampling  0.5695 0.6772       0.8047 0.8455 0.9532 0.9871 0.7283 0.8349
LeastConfidence  0.5715 0.6499       0.8039 0.8553 0.9474 0.9841 0.7357 0.8447
RandomSampling   0.5803 0.6417       0.8094 0.8486 0.9386 0.9775 0.7320 0.8128
VanillaEGL       0.6242 0.7487       0.8341 0.8711 0.9794 0.9939 0.7617 0.8612
```

### Learning Curves






## How to Reproduce

### 1. Prerequisites
This project requires Python and several scientific computing libraries. Key dependencies include:
- `pytorch` & `torchvision`
- `scikit-learn`
- `numpy`
- `pandas` & `matplotlib`
- `faiss-gpu` (optional, for certain strategies)

### 2. Installation
We recommend using Conda to manage the environment.

```bash
# Create and activate the conda environment from the provided file
conda env create -f updated_environment.yml
conda activate deepAL
```

### 3. Running Experiments
The `run_experiments.sh` script automates the entire process. It will iterate through all strategies, datasets, and seeds (42-44).

**Warning**: This is computationally intensive and will take a significant amount of time and GPU resources to complete.

```bash
# Make the script executable and run it
chmod +x run_experiments.sh
./run_experiments.sh
```
This will create a `deepalplus/new_results/` directory and populate it with log files (`*_res.txt`).

### 4. Analyzing Results
After the experiments finish, run the analysis script from the **root directory** to generate the summary table and plots.

```bash
python analyze_results.py deepalplus/new_results/
```
The script will print the summary table to your console and save four learning curve plots (`.png` files) in the project's root directory.

## Acknowledgements
This project is an extension of the **DeepAL+** toolkit. We are grateful to the original authors for providing a comprehensive and well-structured codebase, which served as the foundation for our work. For more information on the base toolkit and its implemented strategies, please refer to the original repository and paper.

-   **DeepAL+ Paper**: [*A comparative survey of deep active learning*](https://arxiv.org/pdf/2203.13450.pdf)
-   **DeepAL+ GitHub**: [https://github.com/ant-lab/DeepAL_plus](https://github.com/ant-lab/DeepAL_plus)