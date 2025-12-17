import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import matplotlib.pyplot as plt

# Load and preprocess data 
data = pd.read_csv('crop_yield.csv')

# Features and target 
features = ['Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide', 'Production', 'Crop_Year']
target = 'Yield'
categorical_cols = [col for col in ['Crop', 'Season', 'State'] if col in data.columns]
if categorical_cols:
    data = pd.get_dummies(data, columns=categorical_cols)

X = data.drop(columns=[target]).values
y = data[target].values

# Standardize feature values
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split into train/test sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# FFNN model 
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Early stopping callback to prevent overfitting
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=64,
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)

# Plot training and validation loss 
plt.figure() # Ensures a new figure is used
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()
plt.savefig('ffnn_loss_plot.png') # Saves the file
plt.close()

# Predict and plot results
y_pred = model.predict(X_test).flatten()
plt.figure() # Ensures a new figure is used
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('True Yield')
plt.ylabel('Predicted Yield')
plt.title('True vs Predicted Yield (Test Data)')
plt.savefig('ffnn_prediction_scatter.png') # Saves the file
plt.close()