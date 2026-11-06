
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from functools import lru_cache


# Rutas del proyecto
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT_DIR / "data" / "ventas_diarias.csv"
MODELS_DIR = ROOT_DIR / "models"

WINDOW_SIZE = 30

# Sabores disponibles
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
# CARGA DE RECURSOS CON CACHÉ
# ==========================================

@lru_cache(maxsize=1)
def load_resources():
    """
    Carga los modelos y escaladores una sola vez.
    Los mantiene en memoria para solicitudes posteriores.
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
# CARGA DEL HISTÓRICO CON CACHÉ
# ==========================================

@lru_cache(maxsize=1)
def load_dataset():
    """
    Lee el CSV una sola vez y conserva los datos en memoria.
    """

    print("Cargando historico de ventas...")

    df = pd.read_csv(
        DATA_FILE,
        parse_dates=["fecha"]
    )

    df = df[
        (df["fecha"].dt.year >= 2021) &
        (df["fecha"].dt.year <= 2024)
    ].copy()

    print("Historico cargado correctamente.")

    return df


def load_historical_data(flavor):
    """
    Obtiene el historico de un sabor desde el CSV almacenado.
    """

    df = load_dataset()

    history = df[
        df["sabor"] == flavor
    ].copy()

    history = history.sort_values(
        "fecha"
    ).reset_index(drop=True)

    return history


# ==========================================
# PREDICCIÓN INDIVIDUAL
# ==========================================

def predict_flavor(flavor, days, models, scalers):
    """
    Genera predicciones recursivas para un sabor.
    """

    if flavor not in FLAVORS:
        raise ValueError(
            f"Sabor no disponible: {flavor}"
        )

    if not isinstance(days, int) or days < 1 or days > 30:
        raise ValueError(
            "Los dias deben estar entre 1 y 30"
        )

    # Obtener histórico desde memoria
    history = load_historical_data(flavor)

    if history.empty:
        raise ValueError(
            f"No hay datos historicos para {flavor}"
        )

    last_values = history[
        "cantidad_vendida"
    ].tail(WINDOW_SIZE).values.reshape(-1, 1)

    if len(last_values) < WINDOW_SIZE:
        raise ValueError(
            f"No hay suficientes registros historicos para {flavor}"
        )

    # Escalador y modelo ya cargados
    scaler = scalers[flavor]
    model = models[flavor]

    current_window = scaler.transform(last_values)

    today = datetime.now(
        ZoneInfo("America/Bogota")
    ).date()

    predictions = []

    # Predicción recursiva optimizada
    for step in range(1, days + 1):

        X = current_window.reshape(
            1,
            WINDOW_SIZE,
            1
        ).astype(np.float32)

        # Llamada directa al modelo: evita la sobrecarga
        # de model.predict() en cada día.
        predicted_scaled = float(
            model(X, training=False)[0, 0].numpy()
        )

        predicted_units = scaler.inverse_transform(
            np.array([[predicted_scaled]])
        )[0][0]

        predicted_units = max(
            0,
            float(predicted_units)
        )

        predicted_scaled = scaler.transform(
            np.array([[predicted_units]])
        )[0][0]

        # Desplazar ventana
        current_window = np.concatenate(
            (
                current_window[1:],
                np.array([[predicted_scaled]])
            ),
            axis=0
        )

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
    Predice varios sabores reutilizando los recursos cargados.
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
    Predice los 13 sabores.
    """

    return predict_multiple_flavors(
        FLAVORS,
        days
    )


# ==========================================
# PRUEBA
# ==========================================

if __name__ == "__main__":

    print("\n========== PRUEBA DE PREDICCIÓN LSTM ==========\n")

    predictions = predict_multiple_flavors(
        ["mango", "fresa", "chocolate"],
        7
    )

    result_df = pd.DataFrame(predictions)

    print(result_df.to_string(index=False))

    print("\nPredicciones generadas correctamente.")
