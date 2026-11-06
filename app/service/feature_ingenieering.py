import pandas as pd
from pathlib import Path


FEATURE_COLUMNS = [
    "dia_semana",
    "dia_mes",
    "mes",
    "semana_anio",
    "rezago_1",
    "rezago_7",
    "rezago_14",
    "promedio_7",
    "promedio_30"
]


def create_features():

    input_file = Path("data/ventas_diarias.csv")

    df = pd.read_csv(input_file, parse_dates=["fecha"])

    df = df.sort_values(["sabor", "fecha"]).copy()

    # Variables del calendario
    df["dia_semana"] = df["fecha"].dt.dayofweek
    df["dia_mes"] = df["fecha"].dt.day
    df["mes"] = df["fecha"].dt.month
    df["semana_anio"] = df["fecha"].dt.isocalendar().week.astype(int)

# Variables de estacionalidad
    df["dia_anio"] = df["fecha"].dt.dayofyear

    import numpy as np

    df["sin_dia_anio"] = np.sin(
       2 * np.pi * df["dia_anio"] / 365.25
    )

    df["cos_dia_anio"] = np.cos(
        2 * np.pi * df["dia_anio"] / 365.25
    )

    df["sin_mes"] = np.sin(
        2 * np.pi * df["mes"] / 12
    )

    df["cos_mes"] = np.cos(
        2 * np.pi * df["mes"] / 12
    )

    df["es_fin_semana"] = (
        df["dia_semana"] >= 5
    ).astype(int)

    # Historial de ventas por sabor
    grouped = df.groupby("sabor")["cantidad_vendida"]

    df["rezago_1"] = grouped.shift(1)
    df["rezago_7"] = grouped.shift(7)
    df["rezago_14"] = grouped.shift(14)

    df["promedio_7"] = grouped.transform(
        lambda x: x.shift(1).rolling(7).mean()
    )

    df["promedio_30"] = grouped.transform(
        lambda x: x.shift(1).rolling(30).mean()
    )

    # Eliminar registros sin suficiente historial
    df = df.dropna().copy()

    output_file = Path("data/ventas_features.csv")

    df.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n========== VARIABLES PREDICTIVAS ==========")
    print("Registros preparados:", len(df))
    print("Sabores:", df["sabor"].nunique())
    print("Variables:", FEATURE_COLUMNS)
    print("Archivo generado:", output_file)

    print("\nPrimeros registros:")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    create_features()