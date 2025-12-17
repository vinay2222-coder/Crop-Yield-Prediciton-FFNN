This is a professional, comprehensive **`README.md`** file tailored specifically for your GitHub repository. You can copy and paste this directly into your project.

---

# 🌾 Crop Yield Prediction using Deep Learning (FFNN)

## 📋 Project Overview

This project implements a **Feed-Forward Neural Network (FFNN)** to predict agricultural crop yields. By analyzing environmental factors, historical data, and farming inputs, the model provides an accurate estimation of yield (Production/Area). This tool is designed to assist farmers, researchers, and policymakers in making data-driven decisions to optimize agricultural productivity.

## 🚀 Key Features

* **Deep Learning Architecture:** A multi-layered neural network with 256, 128, and 64 neurons.
* **Robust Preprocessing:** Handles categorical variables via One-Hot Encoding and numerical features via Standard Scaling.
* **Overfitting Prevention:** Integrated **Dropout layers**, **Batch Normalization**, and **Early Stopping** to ensure high generalization.
* **Data Visualization:** Automatically generates training vs. validation loss plots to monitor model performance.

## 📊 Dataset Description

The model uses `crop_yield.csv`, which contains historical agricultural data.

* **Features:**
* `Area`: Total land used for cultivation.
* `Annual_Rainfall`: Total yearly precipitation.
* `Fertilizer`: Quantity of fertilizer used.
* `Pesticide`: Quantity of pesticide used.
* `Crop_Year`: Year of harvest.
* `Crop`, `Season`, `State`: Categorical environmental and regional context.


* **Target Variable:**
* `Yield`: The calculated output (Production \div Area).



## 🛠️ Tech Stack

* **Language:** Python
* **Libraries:** * `TensorFlow / Keras` (Model building)
* `Pandas / NumPy` (Data manipulation)
* `Scikit-Learn` (Preprocessing & Splitting)
* `Matplotlib` (Visualization)



## 🏗️ Model Architecture

The neural network is structured as follows:

1. **Input Layer:** Dynamic shape based on preprocessed features.
2. **Hidden Layer 1:** 256 Neurons (ReLU) + Batch Normalization + Dropout (0.3).
3. **Hidden Layer 2:** 256 Neurons (ReLU).
4. **Hidden Layer 3:** 128 Neurons (ReLU) + Batch Normalization + Dropout (0.2).
5. **Hidden Layer 4:** 64 Neurons (ReLU) + Batch Normalization + Dropout (0.2).
6. **Output Layer:** 1 Neuron (Linear) to predict the Yield value.

## ⚙️ How to Run

1. **Clone the Repository:**
```bash
git clone https://github.com/your-username/crop-yield-prediction.git
cd crop-yield-prediction

```


2. **Install Dependencies:**
```bash
pip install pandas numpy scikit-learn tensorflow matplotlib

```


3. **Run the Script:**
```bash
python crop_yield_ffnn.py

```



## 📈 Results

Upon completion, the script will:

1. Output the **Mean Absolute Error (MAE)** and **Mean Squared Error (MSE)** for the test set.
2. Save a file named `ffnn_loss_plot.png` showing the convergence of training and validation loss over epochs.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---
