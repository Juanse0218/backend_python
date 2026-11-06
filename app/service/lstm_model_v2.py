
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT_DIR / "data" / "ventas_diarias.csv"
MODELS_DIR = ROOT_DIR / "models_v2"

WINDOW_SIZE = 30
TEST_START = pd.Timestamp("2024-07-01")

EPOCHS = 60
BATCH_SIZE = 32


def create_features(df):
    """Genera variables temporales y estadísticas."""

    df = df.copy()
    dates = pd.to_datetime(df["fecha"])

    df["dia_semana"] = dates.dt.day_of_week
    df["mes"] = dates.dt.month

    day_year = dates.dt.day_of_year

    df["sin_dia_anio"] = np.sin(
        2 * np.pi * day_year / 365.25
    )
    df["cos_dia_anio"] = np.cos(
        2 * np.pi * day_year / 365.25
    )

    df["promedio_7"] = (
        df["cantidad_vendida"]
        .shift(1)
        .rolling(7)
        .mean()
    )

    df["promedio_30"] = (
        df["cantidad_vendida"]
        .shift(1)
        .rolling(30)
        .mean()
    )

    return df


def build_model():
    """Construye la arquitectura LSTM."""

    model = Sequential([
        Input(shape=(WINDOW_SIZE, 7)),

        LSTM(64, return_sequences=True),
        Dropout(0.2),

        LSTM(32),
        Dropout(0.2),

        Dense(32, activation="relu"),
        Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mean_squared_error",
        metrics=["mae"]
    )

    return model


def create_sequences(features, targets, window):
    X, y = [], []

    for i in range(window, len(features)):
        X.append(features[i-window:i])
        y.append(targets[i])

    return (
        np.array(X, dtype=np.float32),
        np.array(y, dtype=np.float32)
    )


def train_lstm_v2():

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(
        DATA_FILE,
        parse_dates=["fecha"]
    )

    df = df.sort_values(["sabor", "fecha"])

    flavors = sorted(df["sabor"].unique())

    metrics = []
    scalers = {}
    epochs_by_flavor = {}

    for flavor in flavors:

        print(f"\nEntrenando LSTM V2: {flavor}")

        flavor_df = df[
            df["sabor"] == flavor
        ].copy()

        flavor_df = create_features(flavor_df)

        flavor_df = flavor_df.dropna().reset_index(drop=True)

        # Separación cronológica
        train_mask = flavor_df["fecha"] < TEST_START

        # Escalador de ventas, ajustado únicamente con entrenamiento
        scaler = MinMaxScaler()

        train_values = flavor_df.loc[
            train_mask, "cantidad_vendida"
        ].values.reshape(-1, 1)

        scaler.fit(train_values)

        # Transformar toda la serie con el escalador de entrenamiento
        scaled_sales = scaler.transform(
            flavor_df["cantidad_vendida"].values.reshape(-1, 1)
        ).flatten()

        # Variables de entrada
        temporal_features = flavor_df[
            [
                "dia_semana",
                "mes",
                "sin_dia_anio",
                "cos_dia_anio",
                "promedio_7",
                "promedio_30"
            ]
        ].values

        # Escalar promedios usando el mismo escalador de ventas
        scaled_avg_7 = scaler.transform(
            flavor_df["promedio_7"].values.reshape(-1, 1)
        ).flatten()

        scaled_avg_30 = scaler.transform(
            flavor_df["promedio_30"].values.reshape(-1, 1)
        ).flatten()

        # Normalizar día y mes
        temporal_features[:, 0] /= 6.0
        temporal_features[:, 1] = (
            temporal_features[:, 1] - 1
        ) / 11.0

        features = np.column_stack([
            scaled_sales,
            temporal_features[:, 0:4],
            scaled_avg_7,
            scaled_avg_30
        ])

        targets = scaled_sales

        X, y = create_sequences(
            features,
            targets,
            WINDOW_SIZE
        )

        sequence_dates = flavor_df["fecha"].iloc[
            WINDOW_SIZE:
        ].reset_index(drop=True)

        test_mask = (
            sequence_dates >= TEST_START
        ).values

        train_mask_sequences = ~test_mask

        X_train = X[train_mask_sequences]
        y_train = y[train_mask_sequences]

        X_test = X[test_mask]
        y_test = y[test_mask]

        model = build_model()

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=8,
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

        # Evaluación de un paso
        predictions_scaled = model.predict(
            X_test,
            verbose=0
        ).flatten()

        predictions = scaler.inverse_transform(
            predictions_scaled.reshape(-1, 1)
        ).flatten()

        real = scaler.inverse_transform(
            y_test.reshape(-1, 1)
        ).flatten()

        predictions = np.maximum(predictions, 0)

        mae = mean_absolute_error(real, predictions)

        rmse = np.sqrt(
            mean_squared_error(real, predictions)
        )

        metrics.append({
            "sabor": flavor,
            "MAE": float(mae),
            "RMSE": float(rmse),
            "epochs": len(history.history["loss"])
        })

        # Reentrenamiento final usando toda la historia disponible
        final_model = build_model()

        final_model.fit(
            X,
            y,
            epochs=len(history.history["loss"]),
            batch_size=BATCH_SIZE,
            verbose=0,
            shuffle=False
        )

        final_model.save(
            MODELS_DIR / f"lstm_v2_{flavor}.keras"
        )

        scalers[flavor] = scaler
        epochs_by_flavor[flavor] = len(history.history["loss"])

        print(
            f"MAE: {mae:.4f} | RMSE: {rmse:.4f}"
        )

    joblib.dump(
        scalers,
        MODELS_DIR / "lstm_v2_scalers.pkl"
    )

    global_mae = np.mean([
        item["MAE"] for item in metrics
    ])

    global_rmse = np.mean([
        item["RMSE"] for item in metrics
    ])

    metadata = {
        "model": "LSTM V2",
        "window_size": WINDOW_SIZE,
        "features": [
            "ventas",
            "dia_semana",
            "mes",
            "sin_dia_anio",
            "cos_dia_anio",
            "promedio_7",
            "promedio_30"
        ],
        "test_start": str(TEST_START.date()),
        "metrics_by_flavor": metrics,
        "global_mean_mae": float(global_mae),
        "global_mean_rmse": float(global_rmse)
    }

    with open(
        MODELS_DIR / "lstm_v2_metadata.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=4
        )

    print("\n========== RESULTADOS LSTM V2 ==========")

    print(f"MAE promedio: {global_mae:.4f}")
    print(f"RMSE promedio: {global_rmse:.4f}")

    print("\nModelos LSTM V2 guardados correctamente.")


if __name__ == "__main__":
    train_lstm_v2()
