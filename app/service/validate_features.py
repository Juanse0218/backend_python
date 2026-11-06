import pandas as pd
from pathlib import Path

def validate_features():

    file = Path("data/ventas_features.csv")

    df = pd.read_csv(file, parse_dates=["fecha"])

    print("\n========== VALIDACIÓN DEL DATASET ==========")

    print("\nTotal de registros:", len(df))

    print("Cantidad de sabores:", df["sabor"].nunique())

    print("Fecha inicial:", df["fecha"].min())

    print("Fecha final:", df["fecha"].max())

    print("\nRegistros por sabor:")
    print(df.groupby("sabor").size())

    print("\nValores nulos:")
    print(df.isnull().sum())

    print("\nEstadísticas de ventas:")
    print(df["cantidad_vendida"].describe())

    print("\nRegistros duplicados:")
    print(df.duplicated(subset=["fecha", "sabor"]).sum())

    print("\n========== VALIDACIÓN FINALIZADA ==========")


if __name__ == "__main__":
    validate_features()