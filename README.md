[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://machine-learning-at-scale-bvceyhyya6dhafws5dwwdx.streamlit.app/)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-181717?logo=github)](https://github.com/etharhamid/Machine-Learning-at-Scale.git)

# Machine Learning at Scale: MovieLens 32M Recommender System

This repository contains the implementation of a high-performance **Matrix Factorization** recommender system scaled to the **MovieLens 32M** dataset (32 million ratings, 200k users, 84k movies).

The project demonstrates the transition from a naive Python implementation to a highly optimized, JIT-compiled solution using **Numba**, achieving a **~100x speedup**. It also introduces a **Genre-Augmented ALS** model to solve cold-start problems by regularizing embeddings towards semantic priors.

---

## 🔗 Live Demo
**Interactive Web Application:** [👉 Click here to try the Streamlit App](https://machine-learning-at-scale-bvceyhyya6dhafws5dwwdx.streamlit.app/)  
*Experience real-time movie recommendations powered by the Numba-optimized backend.*

---

## 📂 Repository Structure

### 1. Core Notebooks
| File Name | Description |
| :--- | :--- |
| **`Full_Submission_Recommender_System.ipynb`** | **The Full Models.** . Implements all the Models and contains te full code | Including **The Baseline (Unoptimized).** Contains the original, pure Python implementation of  Bias only Model and Alternating Least Squares (ALS) Model. Useful for understanding the algorithmic logic before optimization. Includes baseline RMSE analysis. | **The Optimized Solution for ALS.** Contains the high-performance **Numba** implementation. Features JIT-compiled kernels, parallelized loops, and custom Sparse Matrix structures (CSR/CSC). This code runs ~100x faster than the baseline. |

### 2. Application & Deployment
| File Name | Description |
| :--- | :--- |
| `app.py` | The source code for the Streamlit web application. Contains the inference-only Numba kernels for real-time scoring. |
| `requirements.txt` | List of Python dependencies (Numba, NumPy, Pandas, Streamlit, SciPy). |

### 3. Model Artifacts (Saved Weights)
* `movie_embeddings.npy` / `movie_biases.npy`: Pre-trained latent factors ($k=20$) used by the web app.
* `movie_counts.npy`: Used for popularity filtering in the demo.
* `movie_map.pkl`: Dictionary mapping raw MovieLens IDs to internal matrix indices.

---
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
  
## 🚀 Key Features & Optimizations

### 1. Computational Scalability (Numba)
* **Challenge:** The naive Python implementation took **~4.5 hours** per 15 iterations due to the Global Interpreter Lock (GIL) and loop overhead.
* **Solution:** We implemented custom ALS kernels using `@njit(parallel=True, fastmath=True)`.
* **Result:** Training time dropped to **2-3 minutes**, enabling a 50-trial hyperparameter search that improved Test RMSE to **0.7704**.

### 2. Algorithmic Innovation (Genre Augmentation)
* **Challenge:** Standard MF maps unrated (cold-start) items to the zero vector ($\mathbf{0}$), causing unrelated obscure movies to cluster together.
* **Solution:** We modified the loss function to regularize item vectors towards a **Genre Prior** (centroid of their genre tags) instead of zero.
* **Result:** Unrated "Horror" movies now cluster with other Horror movies automatically, solving the semantic consistency problem.

### 3. Latent Space Analysis
* **Polarization:** identified "High-Norm" movies (e.g., *Star Wars*, *Pulp Fiction*) as polarizing content.
* **Semantic Clustering:** PCA visualizations confirm that the model implicitly learned to group franchises (*Harry Potter*, *Saw*) and genres without explicit supervision.

---

## 💻 Installation & Usage

### Prerequisites
* Python 3.8+
* The **MovieLens 32M Dataset** (`ratings.csv`, `movies.csv`) must be downloaded from [GroupLens](https://grouplens.org/datasets/movielens/32m/) and placed in the root directory.

### Setup
1.  Clone the repository:
    ```bash
    git clone [https://github.com/etharhamid/Machine-Learning-at-Scale.git](https://github.com/etharhamid/Machine-Learning-at-Scale.git)
    cd Machine-Learning-at-Scale
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    pip install numpy pandas matplotlib seaborn numba
    ```

### Running the Web App Locally
To launch the interactive dashboard on your local machine:
```bash
streamlit run app.py

