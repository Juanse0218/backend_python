
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_FILE = ROOT_DIR / "data" / "ventas_diarias.csv"
MODELS_DIR = ROOT_DIR / "models"

WINDOW_SIZE = 30

# Entrenar con datos hasta junio de 2025.
TRAIN_END = pd.Timestamp("2025-06-30")

# Evaluar con datos de julio a diciembre de 2025.
TEST_START = pd.Timestamp("2025-07-01")
TEST_END = pd.Timestamp("2025-12-31")

EPOCHS = 50
BATCH_SIZE = 32


def create_sequences(values, window):

    X, y = [], []

    for i in range(window, len(values)):
        X.append(values[i - window:i])
        y.append(values[i])

    return np.array(X), np.array(y)


def build_model():

    model = Sequential([
        tf.keras.Input(shape=(WINDOW_SIZE, 1)),

        LSTM(64, return_sequences=True),
        Dropout(0.2),

        LSTM(32),
        Dropout(0.2),

        Dense(16, activation="relu"),
        Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mean_squared_error",
        metrics=["mae"]
    )

    return model


def train_lstm():

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_FILE, parse_dates=["fecha"])

    df["fecha"] = pd.to_datetime(df["fecha"])

    # Excluir 2026 y cualquier fecha posterior.
    df = df[
        (df["fecha"] >= "2021-01-01") &
        (df["fecha"] <= "2025-12-31")
    ].copy()

    df = df.sort_values(["sabor", "fecha"])

    flavors = sorted(df["sabor"].unique())

    metrics_by_flavor = []

    all_real = []
    all_predictions = []

    final_scalers = {}

    print("\n========== ENTRENAMIENTO LSTM ==========")
    print("Periodo disponible: 2021-2025")
    print(f"Entrenamiento: hasta {TRAIN_END.date()}")
    print(f"Evaluación: {TEST_START.date()} a {TEST_END.date()}")
    print(f"Sabores: {len(flavors)}")

    for flavor in flavors:

        print(f"\nEntrenando sabor: {flavor}")

        flavor_df = df[
            df["sabor"] == flavor
        ].copy()

        flavor_df = flavor_df.sort_values("fecha")

        train_df = flavor_df[
            flavor_df["fecha"] <= TRAIN_END
        ]

        test_df = flavor_df[
            (flavor_df["fecha"] >= TEST_START) &
            (flavor_df["fecha"] <= TEST_END)
        ]

        train_values = train_df[
            "cantidad_vendida"
        ].values.reshape(-1, 1)

        test_values = test_df[
            "cantidad_vendida"
        ].values.reshape(-1, 1)

        if len(train_values) <= WINDOW_SIZE:
            raise ValueError(
                f"Historial insuficiente para entrenar {flavor}"
            )

        if len(test_values) == 0:
            raise ValueError(
                f"No hay datos de evaluación para {flavor}"
            )

        # Escalador ajustado únicamente con entrenamiento.
        scaler = MinMaxScaler()

        train_scaled = scaler.fit_transform(train_values)

        test_scaled = scaler.transform(test_values)

        # Contexto histórico + periodo de evaluación.
        combined = np.concatenate([
            train_scaled,
            test_scaled
        ])

        X, y = create_sequences(
            combined,
            WINDOW_SIZE
        )

        # Primera secuencia cuyo objetivo pertenece a julio de 2025.
        test_start_index = len(train_scaled) - WINDOW_SIZE

        X_train = X[:test_start_index]
        y_train = y[:test_start_index]

        X_test = X[test_start_index:]
        y_test = y[test_start_index:]

        if len(X_train) == 0 or len(X_test) == 0:
            raise ValueError(
                f"No hay suficientes secuencias para {flavor}"
            )

        model = build_model()

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=7,
            restore_best_weights=True
        )

        history = model.fit(
            X_train,
            y_train,
            validation_split=0.15,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[early_stop],
            verbose=0,
            shuffle=False
        )

        best_epochs = len(history.history["loss"])

        predictions_scaled = model.predict(
            X_test,
            verbose=0
        )

        predictions = scaler.inverse_transform(
            predictions_scaled
        ).flatten()

        real = scaler.inverse_transform(
            y_test.reshape(-1, 1)
        ).flatten()

        predictions = np.maximum(predictions, 0)

        mae = mean_absolute_error(
            real,
            predictions
        )

        rmse = mean_squared_error(
            real,
            predictions
        ) ** 0.5

        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"Épocas utilizadas: {best_epochs}")

        metrics_by_flavor.append({
            "sabor": flavor,
            "MAE": float(mae),
            "RMSE": float(rmse),
            "epocas": int(best_epochs)
        })

        all_real.extend(real.tolist())
        all_predictions.extend(predictions.tolist())

        # Modelo final: entrenar con todo 2021-2025.
        # No se incorporan datos de 2026.
        full_values = flavor_df[
            "cantidad_vendida"
        ].values.reshape(-1, 1)

        full_scaler = MinMaxScaler()

        full_scaled = full_scaler.fit_transform(
            full_values
        )

        X_full, y_full = create_sequences(
            full_scaled,
            WINDOW_SIZE
        )

        final_model = build_model()

        final_model.fit(
            X_full,
            y_full,
            epochs=max(1, best_epochs),
            batch_size=BATCH_SIZE,
            verbose=0,
            shuffle=False
        )

        final_model.save(
            MODELS_DIR / f"lstm_{flavor}.keras"
        )

        final_scalers[flavor] = full_scaler

    # Guardar escaladores.
    joblib.dump(
        final_scalers,
        MODELS_DIR / "lstm_scalers.pkl"
    )

    global_mae = mean_absolute_error(
        all_real,
        all_predictions
    )

    global_rmse = mean_squared_error(
        all_real,
        all_predictions
    ) ** 0.5

    metadata = {
        "model": "LSTM",
        "window_size": WINDOW_SIZE,
        "forecast_horizons": [7, 30],
        "training_start": "2021-01-01",
        "training_end": str(TRAIN_END.date()),
        "evaluation_start": str(TEST_START.date()),
        "evaluation_end": str(TEST_END.date()),
        "final_model_training_end": "2025-12-31",
        "excluded_years": [2026],
        "flavors": flavors,
        "global_metrics": {
            "MAE": float(global_mae),
            "RMSE": float(global_rmse)
        },
        "metrics_by_flavor": metrics_by_flavor
    }

    with open(
        MODELS_DIR / "lstm_metadata.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=4
        )

    print("\n========== RESULTADOS GENERALES ==========")

    print(f"MAE global: {global_mae:.4f}")
    print(f"RMSE global: {global_rmse:.4f}")

    print("\nModelos LSTM guardados correctamente.")


if __name__ == "__main__":
    train_lstm()
