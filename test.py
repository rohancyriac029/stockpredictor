import pandas as pd
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# --- For Reproducibility ---
np.random.seed(42)
tf.random.set_seed(42)

# --- Load and Process Data ---
try:
    df = pd.read_csv("aaxj_stock_data.txt")
except FileNotFoundError:
    print("Error: 'aaxj_stock_data.txt' not found. Please ensure the file is in the correct directory.")
    exit()

df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df.sort_index(inplace=True)
df = df.drop(columns=['OpenInt'])

# --- 1. FEATURE ENGINEERING ---
# Add technical indicators using pandas_ta
print("\n--- Engineering new features (SMA, RSI) ---")
df.ta.sma(length=10, append=True) # 10-period Simple Moving Average
df.ta.sma(length=50, append=True) # 50-period Simple Moving Average
df.ta.rsi(length=14, append=True) # 14-period Relative Strength Index

# The first few rows will have NaN values because the indicators need prior data to be calculated.
# We must drop these rows before training the model.
df.dropna(inplace=True)

print("\n--- Data Head with New Features ---")
print(df.head())

# --- 2. PREPARE DATA FOR MULTIVARIATE LSTM ---
# We will use all columns as features to predict the future 'Close' price.
feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_10', 'SMA_50', 'RSI_14']
target_column = 'Close'

# Get the index of our target column (we'll need this later)
target_col_index = df.columns.get_loc(target_column)

# Create the dataset with all features
features = df[feature_columns].values

# Scale all features to a range between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler.fit_transform(features)

# --- Create sequences for LSTM ---
sequence_length = 60
X, y = [], []

for i in range(sequence_length, len(scaled_features)):
    # X contains the last 'sequence_length' days of ALL features
    X.append(scaled_features[i-sequence_length:i, :])
    # y contains the 'Close' price for the next day
    y.append(scaled_features[i, target_col_index])

X, y = np.array(X), np.array(y)

# Get the number of features from the shape of X
n_features = X.shape[2]
print(f"\nNumber of features being used for training: {n_features}")

# --- Split Data into Training and Testing Sets ---
split_ratio = 0.8
split = int(len(X) * split_ratio)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Get the corresponding dates for our test set for plotting later
test_dates = df.index[sequence_length + split:]

print(f"Training set size: {len(X_train)} samples")
print(f"Test set size: {len(X_test)} samples")


# --- 3. BUILD THE MULTIVARIATE LSTM MODEL ---
model = Sequential([
    # The input shape now includes the number of features
    LSTM(50, return_sequences=True, input_shape=(sequence_length, n_features)),
    Dropout(0.2), # Dropout layer to prevent overfitting
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25, activation='relu'),
    Dense(1) # Output is a single value: the predicted 'Close' price
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# --- Train the Model ---
print("\n--- Training Model ---")
history = model.fit(
    X_train,
    y_train,
    epochs=25, # Increased epochs slightly for more complex data
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# --- 4. EVALUATE THE MODEL (WITH MODIFIED INVERSE SCALING) ---
print("\n--- Evaluating Model ---")
predictions_scaled = model.predict(X_test)

# The scaler expects a 2D array with the same number of columns it was fitted on (n_features).
# Our predictions are just a single column. So, we create a dummy array of the correct shape,
# fill it with zeros, place our predictions into the 'Close' column, and then inverse transform.

# Create a dummy array with the same shape as the original features
dummy_array_pred = np.zeros((len(predictions_scaled), n_features))
# Place our scaled predictions into the correct column index
dummy_array_pred[:, target_col_index] = predictions_scaled.flatten()
# Now, inverse transform the entire dummy array
predicted_prices = scaler.inverse_transform(dummy_array_pred)[:, target_col_index]

# Do the same for the actual prices (y_test)
dummy_array_actual = np.zeros((len(y_test), n_features))
dummy_array_actual[:, target_col_index] = y_test.flatten()
actual_prices = scaler.inverse_transform(dummy_array_actual)[:, target_col_index]

# Calculate RMSE on the un-scaled prices for a real-world metric
rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
print(f"\nRoot Mean Squared Error (RMSE) on actual prices: ${rmse:.2f}")


# --- 5. VISUALIZE THE RESULTS ---
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(15, 7))

plt.plot(test_dates, actual_prices, color='blue', label='Actual Price')
plt.plot(test_dates, predicted_prices, color='red', alpha=0.7, label='Predicted Price')

plt.title(f'Multivariate Stock Price Prediction (AAXJ) - {n_features} Features', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price (USD)', fontsize=12)
plt.legend()
plt.gcf().autofmt_xdate()
plt.show()