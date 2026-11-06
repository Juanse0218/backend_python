
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path
from datetime import timedelta


ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_FILE = ROOT_DIR / "data" / "ventas_diarias.csv"
MODELS_DIR = ROOT_DIR / "models_v2"

WINDOW_SIZE = 30

FLAVORS = [
    "cafe",
    "caramelo",
    "chocolate",
    "coco",
    "durazno",
    "fresa",
    "limon",
    "mango",
    "maracuya",
    "menta",
    "mora",
    "nutella",
    "vainilla"
]


def load_resources():
    """Carga los modelos y escaladores V2."""

    scalers = joblib.load(
        MODELS_DIR / "lstm_v2_scalers.pkl"
    )

    models = {}

    for flavor in FLAVORS:

        model_path = (
            MODELS_DIR / f"lstm_v2_{flavor}.keras"
        )

        models[flavor] = tf.keras.models.load_model(
            model_path,
            compile=False
        )

    return models, scalers


def load_historical_data(flavor):
    """Carga el historial de ventas de un sabor."""

    df = pd.read_csv(
        DATA_FILE,
        parse_dates=["fecha"]
    )

    df = df[df["sabor"] == flavor].copy()

    df = df.sort_values("fecha")

    if len(df) < WINDOW_SIZE:
        raise ValueError(
            f"No hay suficientes datos para {flavor}"
        )

    return df.reset_index(drop=True)


def get_temporal_features(date, history):
    """
    Calcula las siete variables de entrada:
    ventas, día, mes, seno, coseno, promedio 7 y promedio 30.
    """

    values = np.array(history, dtype=float)

    day_of_week = date.dayofweek / 6.0

    month = (date.month - 1) / 11.0

    day_of_year = date.dayofyear

    sin_day = np.sin(
        2 * np.pi * day_of_year / 365.25
    )

    cos_day = np.cos(
        2 * np.pi * day_of_year / 365.25
    )

    average_7 = np.mean(values[-7:])

    average_30 = np.mean(values[-30:])

    return [
        day_of_week,
        month,
        sin_day,
        cos_day,
        average_7,
        average_30
    ]


def predict_flavor(flavor, days, models, scalers):
    """Genera predicciones recursivas con LSTM V2."""

    if flavor not in FLAVORS:
        raise ValueError(
            f"Sabor no disponible: {flavor}"
        )

    if not isinstance(days, int) or not 1 <= days <= 365:
        raise ValueError(
            "Los días deben estar entre 1 y 365"
        )

    history_df = load_historical_data(flavor)

    historical_values = (
        history_df["cantidad_vendida"]
        .astype(float)
        .tolist()
    )

    scaler = scalers[flavor]

    model = models[flavor]

    last_date = history_df["fecha"].max()

    # Crear las últimas 30 filas de variables
    recent_features = []

    for i in range(
        len(history_df) - WINDOW_SIZE,
        len(history_df)
    ):

        date = history_df.iloc[i]["fecha"]

        previous_values = historical_values[:i]

        features = get_temporal_features(
            date,
            previous_values
        )

        scaled_sale = scaler.transform(
            np.array([[historical_values[i]]])
        )[0][0]

        scaled_avg_7 = scaler.transform(
            np.array([[features[4]]])
        )[0][0]

        scaled_avg_30 = scaler.transform(
            np.array([[features[5]]])
        )[0][0]

        recent_features.append([
            scaled_sale,
            features[0],
            features[1],
            features[2],
            features[3],
            scaled_avg_7,
            scaled_avg_30
        ])

    current_window = np.array(
        recent_features,
        dtype=np.float32
    )

    predictions = []

    # Pronóstico recursivo
    for step in range(1, days + 1):

        X = current_window.reshape(
            1,
            WINDOW_SIZE,
            7
        )

        predicted_scaled = model.predict(
            X,
            verbose=0
        )[0][0]

        predicted_units = scaler.inverse_transform(
            np.array([[predicted_scaled]])
        )[0][0]

        predicted_units = max(
            0.0,
            float(predicted_units)
        )

        # Agregar predicción al historial
        historical_values.append(predicted_units)

        prediction_date = (
            last_date + timedelta(days=step)
        )

        # Calcular las variables del nuevo día
        features = get_temporal_features(
            prediction_date,
            historical_values[:-1]
        )

        scaled_avg_7 = scaler.transform(
            np.array([[features[4]]])
        )[0][0]

        scaled_avg_30 = scaler.transform(
            np.array([[features[5]]])
        )[0][0]

        new_row = np.array([[
            scaler.transform(
                np.array([[predicted_units]])
            )[0][0],
            features[0],
            features[1],
            features[2],
            features[3],
            scaled_avg_7,
            scaled_avg_30
        ]], dtype=np.float32)

        # Desplazar ventana de 30 días
        current_window = np.concatenate([
            current_window[1:],
            new_row
        ], axis=0)

        predictions.append({
            "fecha": prediction_date.strftime("%Y-%m-%d"),
            "sabor": flavor,
            "cantidad_predicha": round(
                predicted_units, 2
            )
        })

    return predictions


def predict_multiple_flavors(flavors, days):
    """Predice varios sabores."""

    if not flavors:
        raise ValueError(
            "Debes seleccionar al menos un sabor"
        )

    models, scalers = load_resources()

    results = []

    for flavor in flavors:

        results.extend(
            predict_flavor(
                flavor,
                days,
                models,
                scalers
            )
        )

    return results


def predict_all_flavors(days):
    """Predice los 13 sabores."""

    return predict_multiple_flavors(
        FLAVORS,
        days
    )


if __name__ == "__main__":

    print("\n========== PREDICCIONES LSTM V2 ==========\n")

    predictions = predict_multiple_flavors(
        ["mango", "fresa", "chocolate"],
        30
    )

    df = pd.DataFrame(predictions)

    print(df.to_string(index=False))

    print("\n========== ESTADISTICAS ==========\n")

    print(
        df.groupby("sabor")["cantidad_predicha"]
        .agg(["min", "max", "mean", "std"])
        .round(2)
    )

    print("\nPredicciones LSTM V2 generadas correctamente.")
