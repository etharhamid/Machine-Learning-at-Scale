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

| File | Description |
| :--- | :--- |
| `Full_Recommender_System.ipynb` | **Exploration & Analysis**: Covers EDA, data preprocessing, feature engineering, and dimensionality reduction (PCA) to understand the latent structure of the MovieLens data. |
| `OptimizedVec_32M_Recommender_System.ipynb` | **Scale Implementation**: The high-performance engine. This notebook uses Numba and vectorized operations to perform similarity calculations on the full 32M dataset efficiently. |

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

## ⚡ Getting Started

### Prerequisites

Ensure you have Python installed. You can install the required dependencies using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn numba
