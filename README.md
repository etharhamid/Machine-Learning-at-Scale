# Machine Learning at Scale: Recommender Systems

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![Dataset](https://img.shields.io/badge/dataset-MovieLens%2032M-orange)
![Status](https://img.shields.io/badge/status-active-brightgreen)

## 📖 Overview

This repository demonstrates the implementation and optimization of Recommender Systems designed to handle large-scale datasets. Specifically, it tackles the challenge of processing the **MovieLens 32M** dataset (32 million ratings, 87,000+ movies).

The project progresses from standard exploratory data analysis and dimensionality reduction to building a highly optimized, vectorized recommender engine capable of efficient similarity calculations using **Numba** and **JIT compilation**.

## 🚀 Key Features

- **Scalable Data Processing**: Efficient handling of 32 million interaction records using optimized data structures.
- **Dimensionality Reduction**: Implementation of **PCA (Principal Component Analysis)** to reduce feature space and visualize movie genre clusters.
- **High-Performance Computing**: utilization of **Numba** to JIT-compile critical similarity calculation paths, achieving performance significantly faster than standard Python/Pandas loops.
- **Item-Based Collaborative Filtering**: A custom implementation of item-item similarity for generating personalized movie recommendations.

## 📂 Repository Structure

* `Recommender_System_32M.ipynb`: The baseline (naïve) Python implementation. Useful for understanding the core algorithm without optimization complexity.
* `numba_Recommender_System_32M.ipynb`: The **production** code. Contains the Numba-optimized ALS kernels, custom Sparse Matrix classes (CSR/CSC), and the full training pipeline.
* `random_search_results.csv`: The complete log of the 50-trial hyperparameter search.

## 📊 Dataset

This project utilizes the **MovieLens 32M** dataset, distinct for its size and complexity:
- **32,000,000+** Ratings
- **200,000+** Users
- **87,000+** Movies

*Note: The notebooks contain instructions/scripts to download and format the data directly.*

## 🛠️ Tech Stack

- **Language**: Python
- **Data Manipulation**: Pandas, NumPy
- **Machine Learning**: Scikit-Learn (PCA, Preprocessing)
- **Optimization**: Numba (Just-In-Time compilation)
- **Visualization**: Matplotlib, Seaborn

## 💻 Usage

1.  **Install Dependencies:**
    ```bash
    pip install numpy pandas matplotlib seaborn numba
    ```
2.  **Download Data:**
    Place the `ratings.csv` and `movies.csv` from the MovieLens 32M dataset in the root directory.
3.  **Run the Notebook:**
    Open `numba_Recommender_System_32M.ipynb` and execute the cells. The script handles data loading, sparse matrix construction, training, and visualization automatically.
