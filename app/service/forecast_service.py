
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from functools import lru_cache


# ==========================================
# CONFIGURACIÓN
# ==========================================

ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_FILE = ROOT_DIR / "data" / "ventas_diarias.csv"
MODELS_DIR = ROOT_DIR / "models"

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


# ==========================================
# CARGA DE MODELOS Y ESCALADORES
# ==========================================

@lru_cache(maxsize=1)
def load_resources():
    """
    Carga los 13 modelos LSTM y los escaladores
    una sola vez por proceso.
    """

    print("Cargando modelos LSTM...")

    scalers = joblib.load(
        MODELS_DIR / "lstm_scalers.pkl"
    )

    models = {}

    for flavor in FLAVORS:

        model_path = MODELS_DIR / f"lstm_{flavor}.keras"

        models[flavor] = tf.keras.models.load_model(
            model_path,
            compile=False
        )

    print("Los 13 modelos LSTM están cargados.")

    return models, scalers


# ==========================================
# CARGA DEL HISTÓRICO
# ==========================================

@lru_cache(maxsize=1)
def load_dataset():
    """
    Lee el CSV una sola vez y conserva el histórico
    en memoria.
    """

    print("Cargando histórico de ventas...")

    df = pd.read_csv(
        DATA_FILE,
        parse_dates=["fecha"]
    )

    df = df[
        (df["fecha"].dt.year >= 2021) &
        (df["fecha"].dt.year <= 2024)
    ].copy()

    print("Histórico cargado correctamente.")

    return df


@lru_cache(maxsize=13)
def load_historical_values(flavor):
    """
    Obtiene y almacena en caché los últimos 30 registros
    históricos de cada sabor.
    """

    if flavor not in FLAVORS:
        raise ValueError(f"Sabor no disponible: {flavor}")

    df = load_dataset()

    history = df.loc[
        df["sabor"] == flavor,
        ["fecha", "cantidad_vendida"]
    ]

    history = history.sort_values("fecha")

    values = history[
        "cantidad_vendida"
    ].tail(WINDOW_SIZE).to_numpy(dtype=np.float64)

    if len(values) < WINDOW_SIZE:
        raise ValueError(
            f"No hay suficientes registros históricos para {flavor}"
        )

    # El arreglo almacenado en caché no debe modificarse.
    values.setflags(write=False)

    return values


# ==========================================
# PREDICCIÓN INDIVIDUAL
# ==========================================

def predict_flavor(flavor, days, models, scalers):
    """
    Genera predicciones recursivas para un sabor.
    Conserva el modelo LSTM y su lógica de predicción.
    """

    if flavor not in FLAVORS:
        raise ValueError(
            f"Sabor no disponible: {flavor}"
        )

    if not isinstance(days, int) or days < 1 or days > 30:
        raise ValueError(
            "Los días deben estar entre 1 y 30"
        )

    # Histórico previamente almacenado en caché.
    last_values = load_historical_values(flavor)

    # Copia local para evitar modificar el histórico almacenado.
    last_values = last_values.reshape(-1, 1).copy()

    scaler = scalers[flavor]
    model = models[flavor]

    current_window = scaler.transform(last_values)

    today = datetime.now(
        ZoneInfo("America/Bogota")
    ).date()

    predictions = []

    # Predicción recursiva.
    for step in range(1, days + 1):

        X = current_window.reshape(
            1,
            WINDOW_SIZE,
            1
        ).astype(np.float32)

        predicted_scaled = float(
            model(X, training=False)[0, 0].numpy()
        )

        predicted_units = scaler.inverse_transform(
            np.array([[predicted_scaled]])
        )[0, 0]

        predicted_units = max(
            0,
            float(predicted_units)
        )

        predicted_scaled = scaler.transform(
            np.array([[predicted_units]])
        )[0, 0]

        # Desplazamiento de ventana sin crear
        # un nuevo arreglo mediante np.concatenate.
        current_window[:-1, 0] = current_window[1:, 0]
        current_window[-1, 0] = predicted_scaled

        prediction_date = today + timedelta(
            days=step
        )

        predictions.append({
            "fecha": prediction_date.strftime("%Y-%m-%d"),
            "sabor": flavor,
            "cantidad_predicha": round(
                predicted_units,
                2
            )
        })

    return predictions


# ==========================================
# PREDICCIÓN DE VARIOS SABORES
# ==========================================

def predict_multiple_flavors(flavors, days):
    """
    Predice varios sabores reutilizando los modelos
    y el histórico almacenado en memoria.
    """

    models, scalers = load_resources()

    results = []

    for flavor in flavors:

        flavor_predictions = predict_flavor(
            flavor,
            days,
            models,
            scalers
        )

        results.extend(flavor_predictions)

    return results


# ==========================================
# PREDICCIÓN GENERAL
# ==========================================

def predict_all_flavors(days):
    """
    Predice los 13 sabores disponibles.
    """

    return predict_multiple_flavors(
        FLAVORS,
        days
    )


# ==========================================
# PRUEBA
# ==========================================

if __name__ == "__main__":

    import time

    print("\n========== PRUEBA DE PREDICCIÓN LSTM ==========\n")

    start = time.perf_counter()

    predictions = predict_multiple_flavors(
        ["mango", "fresa", "chocolate"],
        7
    )

    elapsed = time.perf_counter() - start

    result_df = pd.DataFrame(predictions)

    print(result_df.to_string(index=False))

    print(
        f"\nTiempo de ejecución: {elapsed:.2f} segundos"
    )

    print("\nPredicciones generadas correctamente.")
