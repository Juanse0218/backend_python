
import os
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# ==========================================
# CONFIGURACIÓN
# ==========================================

DATA_PATH = "data/ventas_diarias.csv"
RESULTS_DIR = "data/backtest"

WINDOW = 30
TRAIN_END = "2024-07-01"

FORECAST_ORIGINS = [
    "2024-07-01",
    "2024-08-01",
    "2024-09-01",
    "2024-10-01",
    "2024-11-01",
    "2024-12-01",
]

HORIZONS = [7, 30]

os.makedirs(RESULTS_DIR, exist_ok=True)


# ==========================================
# CARGAR DATOS
# ==========================================

def load_data():
    df = pd.read_csv(DATA_PATH)

    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.sort_values(["sabor", "fecha"])

    return df


# ==========================================
# CARACTERÍSTICAS V2
# ==========================================

def temporal_features(date, history, scaler):

    day_of_year = date.dayofyear

    avg_7 = np.mean(history[-7:])
    avg_30 = np.mean(history[-30:])

    scaled_avg_7 = scaler.transform(
        np.array([[avg_7]])
    )[0][0]

    scaled_avg_30 = scaler.transform(
        np.array([[avg_30]])
    )[0][0]

    return [
        date.dayofweek / 6.0,
        (date.month - 1) / 11.0,
        np.sin(2 * np.pi * day_of_year / 365.25),
        np.cos(2 * np.pi * day_of_year / 365.25),
        scaled_avg_7,
        scaled_avg_30
    ]

# ==========================================
# ARQUITECTURAS
# ==========================================

def build_v1():
    model = Sequential([
        tf.keras.Input(shape=(WINDOW, 1)),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    return model


def build_v2():
    model = Sequential([
        tf.keras.Input(shape=(WINDOW, 7)),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    return model


# ==========================================
# PREPARAR ENTRENAMIENTO V1
# ==========================================


def prepare_v1(history):

    split = int(len(history) * 0.90)

    scaler = MinMaxScaler()
    scaler.fit(history[:split].reshape(-1, 1))

    scaled = scaler.transform(
        history.reshape(-1, 1)
    )

    X, y = [], []

    for i in range(WINDOW, len(scaled)):
        X.append(scaled[i-WINDOW:i])
        y.append(scaled[i])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    train_count = split - WINDOW

    return X[:train_count], y[:train_count], X[train_count:], y[train_count:], scaler



def prepare_v2(history, dates):

    split = int(len(history) * 0.90)

    scaler = MinMaxScaler()
    scaler.fit(history[:split].reshape(-1, 1))

    scaled = scaler.transform(
        history.reshape(-1, 1)
    ).flatten()

    features = []

    # Descartar los primeros 30 días,
    # igual que en el entrenamiento original V2.
    for i in range(WINDOW, len(history)):

        date = dates[i]
        day_of_year = date.dayofyear

        avg_7 = np.mean(history[i-7:i])
        avg_30 = np.mean(history[i-30:i])

        features.append([
            scaled[i],
            date.dayofweek / 6.0,
            (date.month - 1) / 11.0,
            np.sin(2 * np.pi * day_of_year / 365.25),
            np.cos(2 * np.pi * day_of_year / 365.25),
            scaler.transform([[avg_7]])[0][0],
            scaler.transform([[avg_30]])[0][0]
        ])

    features = np.array(features, dtype=np.float32)

    X, y = [], []

    # Construir secuencias de 30 días.
    for i in range(WINDOW, len(features)):

        X.append(features[i-WINDOW:i])

        # La etiqueta corresponde al día siguiente
        # a los 30 días de entrada.
        y.append(scaled[i + WINDOW])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    # Se descuentan 30 días de características
    # y otros 30 días de secuencia.
    train_count = split - (2 * WINDOW)

    return (
        X[:train_count],
        y[:train_count],
        X[train_count:],
        y[train_count:],
        scaler
    )

# ==========================================
# ENTRENAR MODELOS
# ==========================================


def train_models(df):

    models = {}

    train_end = pd.Timestamp(TRAIN_END)

    for flavor, group in df.groupby("sabor"):

        group = group.sort_values("fecha")

        train = group[group["fecha"] < train_end]

        history = train["cantidad_vendida"].values.astype(float)
        dates = train["fecha"].tolist()

        print(f"\nEntrenando {flavor}...")

        X1_train, y1_train, X1_val, y1_val, scaler1 = prepare_v1(history)

        model1 = build_v1()

        model1.fit(
            X1_train,
            y1_train,
            validation_data=(X1_val, y1_val),
            epochs=60,
            batch_size=32,
            callbacks=[
                EarlyStopping(
                    monitor="val_loss",
                    patience=8,
                    restore_best_weights=True
                )
            ],
            shuffle=False,
            verbose=0
        )

        X2_train, y2_train, X2_val, y2_val, scaler2 = prepare_v2(
            history,
            dates
        )

        model2 = build_v2()

        model2.fit(
            X2_train,
            y2_train,
            validation_data=(X2_val, y2_val),
            epochs=60,
            batch_size=32,
            callbacks=[
                EarlyStopping(
                    monitor="val_loss",
                    patience=8,
                    restore_best_weights=True
                )
            ],
            shuffle=False,
            verbose=0
        )

        models[flavor] = {
            "v1": model1,
            "v2": model2,
            "scaler_v1": scaler1,
            "scaler_v2": scaler2,
        }

    return models


# ==========================================
# PREDICCIÓN RECURSIVA V1
# ==========================================

def forecast_v1(model, scaler, history, horizon):

    window = scaler.transform(
        np.array(history[-WINDOW:]).reshape(-1, 1)
    )

    predictions = []

    for _ in range(horizon):

        pred_scaled = model.predict(
            window.reshape(1, WINDOW, 1),
            verbose=0
        )[0][0]

        pred = scaler.inverse_transform(
            np.array([[pred_scaled]])
        )[0][0]

        pred = max(0, pred)

        predictions.append(pred)

        new_value = scaler.transform(
            np.array([[pred]])
        )

        window = np.vstack([
            window[1:],
            new_value
        ])

    return predictions


# ==========================================
# PREDICCIÓN RECURSIVA V2
# ==========================================


def forecast_v2(model, scaler, history, last_date, horizon):

    history = list(history)
    predictions = []

    for step in range(1, horizon + 1):

        forecast_date = last_date + pd.Timedelta(days=step)

        window = []

        # Construir las últimas 30 observaciones
        for i in range(len(history) - WINDOW, len(history)):

            current_date = last_date + pd.Timedelta(
                days=i - (len(history) - 1)
            )

            # Variables temporales
            day_of_year = current_date.dayofyear

            avg_7 = np.mean(history[max(0, i-7):i])
            avg_30 = np.mean(history[max(0, i-30):i])

            scaled_value = scaler.transform(
                np.array([[history[i]]])
            )[0][0]

            scaled_avg_7 = scaler.transform(
                np.array([[avg_7]])
            )[0][0]

            scaled_avg_30 = scaler.transform(
                np.array([[avg_30]])
            )[0][0]

            window.append([
                scaled_value,
                current_date.dayofweek / 6.0,
                (current_date.month - 1) / 11.0,
                np.sin(2 * np.pi * day_of_year / 365.25),
                np.cos(2 * np.pi * day_of_year / 365.25),
                scaled_avg_7,
                scaled_avg_30
            ])

        X = np.array(
            window,
            dtype=np.float32
        ).reshape(1, WINDOW, 7)

        pred_scaled = model.predict(
            X,
            verbose=0
        )[0][0]

        pred = scaler.inverse_transform(
            np.array([[pred_scaled]])
        )[0][0]

        pred = max(0, float(pred))

        predictions.append(pred)

        # Incorporar la predicción al historial
        history.append(pred)

    return predictions



# ==========================================
# MODELO BASELINE
# ==========================================

def forecast_baseline(history, horizon):

    value = np.mean(history[-30:])

    return [max(0, value)] * horizon


# ==========================================
# EVALUACIÓN
# ==========================================

def evaluate():

    df = load_data()

    models = train_models(df)

    records = []

    for origin_str in FORECAST_ORIGINS:

        origin = pd.Timestamp(origin_str)

        print(f"\nEvaluando desde {origin.date()}")

        for flavor, model_data in models.items():

            group = df[df["sabor"] == flavor].sort_values("fecha")

            history_df = group[group["fecha"] < origin]

            history = history_df["cantidad_vendida"].values.astype(float)

            last_date = history_df["fecha"].max()

            if len(history) < WINDOW:
                continue

            for horizon in HORIZONS:

                future = group[
                    (group["fecha"] >= origin)
                    & (
                        group["fecha"]
                        < origin + pd.Timedelta(days=horizon)
                    )
                ]

                if len(future) != horizon:
                    continue

                actual = future["cantidad_vendida"].values.astype(float)

                pred_v1 = forecast_v1(
                    model_data["v1"],
                    model_data["scaler_v1"],
                    history,
                    horizon
                )

                pred_v2 = forecast_v2(
                    model_data["v2"],
                    model_data["scaler_v2"],
                    history,
                    last_date,
                    horizon
                )

                baseline = forecast_baseline(
                    history,
                    horizon
                )

                for i in range(horizon):

                    records.append({
                        "fecha_inicio": origin.date(),
                        "fecha": future.iloc[i]["fecha"].date(),
                        "sabor": flavor,
                        "horizonte": horizon,
                        "dia_pronostico": i + 1,
                        "real": actual[i],
                        "v1": pred_v1[i],
                        "v2": pred_v2[i],
                        "baseline": baseline[i],
                    })

    results = pd.DataFrame(records)

    results.to_csv(
        f"{RESULTS_DIR}/resultados_backtest.csv",
        index=False
    )

    metrics = []

    for horizon, group in results.groupby("horizonte"):

        for model in ["v1", "v2", "baseline"]:

            mae = mean_absolute_error(
                group["real"],
                group[model]
            )

            rmse = np.sqrt(
                mean_squared_error(
                    group["real"],
                    group[model]
                )
            )

            metrics.append({
                "horizonte": horizon,
                "modelo": model,
                "MAE": mae,
                "RMSE": rmse,
            })

    metrics_df = pd.DataFrame(metrics)

    metrics_df.to_csv(
        f"{RESULTS_DIR}/metricas_por_modelo.csv",
        index=False
    )

    print("\n========== RESULTADOS GENERALES ==========")
    print(metrics_df.to_string(index=False))

    flavor_metrics = []

    for (horizon, flavor), group in results.groupby(
        ["horizonte", "sabor"]
    ):

        for model in ["v1", "v2", "baseline"]:

            flavor_metrics.append({
                "horizonte": horizon,
                "sabor": flavor,
                "modelo": model,
                "MAE": mean_absolute_error(
                    group["real"],
                    group[model]
                ),
                "RMSE": np.sqrt(
                    mean_squared_error(
                        group["real"],
                        group[model]
                    )
                )
            })

    pd.DataFrame(flavor_metrics).to_csv(
        f"{RESULTS_DIR}/metricas_por_sabor.csv",
        index=False
    )

    print("\nEvaluación finalizada.")
    print(f"Archivos guardados en: {RESULTS_DIR}")


if __name__ == "__main__":
    evaluate()
