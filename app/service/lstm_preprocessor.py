
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.preprocessing import MinMaxScaler


ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_FILE = ROOT_DIR / "data" / "ventas_diarias.csv"

WINDOW_SIZE = 30


def prepare_lstm_data():

    df = pd.read_csv(DATA_FILE, parse_dates=["fecha"])

    df = df.sort_values(["sabor", "fecha"])

    scaler = MinMaxScaler(feature_range=(0, 1))

    X = []
    y = []
    sabores = []
    fechas = []

    for sabor, grupo in df.groupby("sabor"):

        ventas = grupo["cantidad_vendida"].values.reshape(-1, 1)

        ventas_scaled = scaler.fit_transform(ventas)

        for i in range(WINDOW_SIZE, len(ventas_scaled)):

            X.append(
                ventas_scaled[i - WINDOW_SIZE:i]
            )

            y.append(
                ventas_scaled[i, 0]
            )

            sabores.append(sabor)

            fechas.append(
                grupo.iloc[i]["fecha"]
            )

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    print("\n========== PREPARACIÓN LSTM ==========")

    print("Forma de X:", X.shape)
    print("Forma de y:", y.shape)

    print("Ventana histórica:", WINDOW_SIZE, "días")

    print("Cantidad de sabores:", len(set(sabores)))

    print("Secuencias generadas:", len(X))

    return X, y, np.array(sabores), np.array(fechas)


if __name__ == "__main__":

    prepare_lstm_data()
