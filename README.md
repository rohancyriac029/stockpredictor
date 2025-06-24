# 📈 Multivariate LSTM Stock Price Predictor

This project demonstrates how to build a multivariate Long Short-Term Memory (LSTM) neural network to predict the closing price of a stock. It uses historical price data along with several technical indicators as features to create a more robust and context-aware prediction model.

The example uses the **iShares MSCI All Country Asia ex Japan ETF (AAXJ)**, but the framework can be easily adapted for any stock or time-series dataset.

## 📊 Final Prediction Plot

The model's performance is evaluated by plotting the predicted prices against the actual prices on the test dataset.



*(Note: You would replace the link above with a screenshot of your generated `matplotlib` plot)*

---

## ✅ Key Features

-   **Multivariate Input:** Uses multiple features (`Open`, `High`, `Low`, `Close`, `Volume`, `SMA`, `RSI`) instead of just a single time series.
-   **Automated Feature Engineering:** Leverages the `pandas_ta` library to easily calculate and append technical indicators like Simple Moving Averages (SMA) and the Relative Strength Index (RSI).
-   **Stacked LSTM Architecture:** Implements a deep neural network with two stacked LSTM layers for learning complex temporal patterns.
-   **Data Normalization:** Employs `MinMaxScaler` to scale all features, which is crucial for the stable training of neural networks.
-   **Clear Visualization:** Plots the actual vs. predicted prices for a clear, visual evaluation of the model's performance.

---

## 🛠️ Technology Stack

-   **Python 3.x**
-   **TensorFlow / Keras:** For building and training the LSTM model.
-   **Pandas:** For data manipulation and analysis.
-   **Pandas TA:** For technical analysis feature engineering.
-   **Scikit-learn:** For data preprocessing (`MinMaxScaler`) and evaluation (`mean_squared_error`).
-   **NumPy:** For numerical operations.
-   **Matplotlib:** For plotting and visualizing the results.

---

## 🚀 Setup & Usage

Follow these steps to get the project up and running on your local machine.

### 1. Clone the Repository

```sh
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

---



