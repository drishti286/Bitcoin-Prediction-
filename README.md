# Bitcoin-Prediction-

This project is focused on forecasting Bitcoin prices using various deep learning techniques, including a custom implementation of the N-BEATS model. The main objective is to build an ensemble of forecasting models to predict future Bitcoin prices based on historical data, aiming to achieve higher accuracy than standard baseline models like naive forecasting.

## Tech Stack:
* Python: Core programming language used.
* TensorFlow & Keras: Used for building deep learning models.
* NumPy & Pandas: For data manipulation and handling time-series data.
* Matplotlib: Visualization of predictions and metrics.
* Scikit-learn: Preprocessing and scaling of data.
* Custom N-BEATS Model: For building the ensemble learning framework.

## Methodology
Data Collection: Bitcoin historical prices and block reward information are used as the primary dataset.

## Data Preprocessing:

Windowing the time series data into rolling windows to train models on historical price data.
Including features like block reward changes to enhance multivariate predictions.
Min-Max scaling to normalize the dataset for better model performance.
Models Implemented:

Naive Baseline: Predicts the next timestep's price as the current timestep's price.
N-Beats Model: A neural network-based forecasting model designed for time series data.
Dense Neural Network: Several variations with different window sizes and horizon setups.
1D Convolutional Neural Network: To capture patterns in the sequential Bitcoin prices.
LSTM Model: A recurrent neural network for capturing time-dependent relationships.
Multivariate Model: Incorporates additional features like block reward alongside price.

## Evaluation:
Models are evaluated based on common metrics like MAE (Mean Absolute Error), RMSE (Root Mean Squared Error), and MAPE (Mean Absolute Percentage Error).
Comparisons are made between naive forecasting, N-Beats, dense networks, and other implemented models.
Ensemble Learning: An ensemble of multiple model predictions is combined to enhance prediction stability and accuracy.

## Results
The ensemble models and advanced deep learning architectures (N-Beats, LSTM, and Conv1D) consistently outperformed the naive forecasting baseline, achieving better accuracy in predicting Bitcoin price trends. The project provides a robust framework for time series forecasting, useful for cryptocurrency market analysis

