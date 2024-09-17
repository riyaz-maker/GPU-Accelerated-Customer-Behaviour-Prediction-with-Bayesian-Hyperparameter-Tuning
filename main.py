import optuna
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn

def objective(trial):
    lstm_units = trial.suggest_int('lstm_units', 64, 256)
    lstm_layers = trial.suggest_int('lstm_layers', 1, 2)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    batch_size = trial.suggest_int('batch_size', 32, 128)
    epochs = trial.suggest_int('epochs', 10, 50)

    model = Sequential()
    for i in range(lstm_layers):
        if i == 0:
            model.add(LSTM(lstm_units, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
        else:
            model.add(LSTM(lstm_units // 2, return_sequences=False))
    model.add(Dense(1))

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')

    with tf.device('/GPU:0'):
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=0)

    # Evaluate the model on the test data
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse

for trial in study.trials:
    print(f"Trial {trial.number}: Value: {trial.value}, Params: {trial.params}")

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=7)

# Print the best parameters
print(f"Best Hyperparameters: {study.best_params}")

# Train the final model with the best hyperparameters
best_params = study.best_params

model.fit(X_train, y_train, epochs=best_params['epochs'], batch_size=best_params['batch_size'], validation_split=0.1, callbacks=[early_stopping])

# Build the final model using the best hyperparameters
final_model = Sequential()
for i in range(best_params['lstm_layers']):
    if i == 0:
        final_model.add(LSTM(best_params['lstm_units'], input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
    else:
        final_model.add(LSTM(best_params['lstm_units'] // 2, return_sequences=False))

final_model.add(Dense(1))
final_optimizer = tf.keras.optimizers.Adam(learning_rate=best_params['learning_rate'])
final_model.compile(optimizer=final_optimizer, loss='mse')

with tf.device('/GPU:0'):
    final_model.fit(X_train, y_train, epochs=best_params['epochs'], batch_size=best_params['batch_size'], validation_split=0.1)

# Evaluate the final model
y_pred_final = final_model.predict(X_test)
mse_final = mean_squared_error(y_test, y_pred_final)
print(f"Final Model MSE: {mse_final}")