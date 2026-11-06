import pandas as pd

from pathlib import Path


def analyze_dataset():

    file = Path("data/ventas_diarias.csv")

    df = pd.read_csv(file, parse_dates=["fecha"])

    print("\n========== ANÁLISIS GENERAL ==========")

    print("Total de registros:", len(df))
    print("Total de sabores:", df["sabor"].nunique())
    print("Total de fechas:", df["fecha"].nunique())

    print("\n========== ESTADÍSTICAS POR SABOR ==========")

    summary = (
        df.groupby("sabor")["cantidad_vendida"]
        .agg(
            total="sum",
            promedio_diario="mean",
            maximo_diario="max",
            dias_sin_ventas=lambda x: (x == 0).sum()
        )
        .sort_values("total", ascending=False)
    )

    print(summary.to_string())

    print("\n========== VENTAS POR AÑO ==========")

    df["año"] = df["fecha"].dt.year

    yearly = (
        df.groupby("año")["cantidad_vendida"]
        .sum()
    )

    print(yearly.to_string())

    print("\n========== VENTAS POR MES ==========")

    df["mes"] = df["fecha"].dt.month

    monthly = (
        df.groupby("mes")["cantidad_vendida"]
        .sum()
    )

    print(monthly.to_string())

    print("\n========== DISTRIBUCIÓN POR AÑO Y MES ==========")

    df["año"] = df["fecha"].dt.year
    df["mes"] = df["fecha"].dt.month

    monthly_year = df.pivot_table(
        index="mes",
        columns="año",
        values="cantidad_vendida",
        aggfunc="sum",
        fill_value=0
    )

    print(monthly_year.to_string())

    print("\n========== TOTAL DE UNIDADES POR AÑO ==========")

    print(
        df.groupby("año")["cantidad_vendida"]
        .sum()
        .to_string()
    )



if __name__ == "__main__":
    analyze_dataset()