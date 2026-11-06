
from pathlib import Path
from datetime import timezone
from zoneinfo import ZoneInfo

import pandas as pd

from app.database import sales_collection


def build_daily_dataset():

    print("Consultando ventas de MongoDB...")

    sales = sales_collection.find(
        {},
        {
            "fechaVenta": 1,
            "helados.helado.sabor": 1,
            "helados.cantidad": 1
        }
    )

    records = []
    colombia_tz = ZoneInfo("America/Bogota")

    for sale in sales:

        fecha = sale.get("fechaVenta")

        if not fecha:
            continue

        # Convertir fecha UTC a hora de Colombia
        if fecha.tzinfo is None:
            fecha = fecha.replace(tzinfo=timezone.utc)

        fecha_local = fecha.astimezone(colombia_tz).date()

        for item in sale.get("helados", []):

            helado = item.get("helado", {})
            sabor = helado.get("sabor")
            cantidad = item.get("cantidad", 0)

            if not sabor or not isinstance(cantidad, (int, float)):
                continue

            records.append({
                "fecha": fecha_local,
                "sabor": sabor.strip().lower(),
                "cantidad_vendida": cantidad
            })

    if not records:
        print("No se encontraron registros válidos.")
        return

    # Convertir los registros en DataFrame
    df = pd.DataFrame(records)

    # Conservar desde 2021 hasta la última fecha disponible
    df["año"] = pd.to_datetime(df["fecha"]).dt.year
    df = df[df["año"] >= 2021]
    df = df.drop(columns=["año"])

    # Agrupar cantidades por fecha y sabor
    daily = (
        df.groupby(["fecha", "sabor"])["cantidad_vendida"]
        .sum()
        .reset_index()
    )

    # Crear calendario completo para cada sabor
    fechas = pd.date_range(
        start=daily["fecha"].min(),
        end=daily["fecha"].max(),
        freq="D"
    )

    sabores = sorted(daily["sabor"].unique())

    calendario = pd.MultiIndex.from_product(
        [fechas.date, sabores],
        names=["fecha", "sabor"]
    ).to_frame(index=False)

    # Incorporar días sin ventas con cantidad cero
    daily = calendario.merge(
        daily,
        on=["fecha", "sabor"],
        how="left"
    )

    daily["cantidad_vendida"] = (
        daily["cantidad_vendida"].fillna(0)
    )

    # Ordenar información
    daily = daily.sort_values(["fecha", "sabor"])

    # Crear carpeta de datos
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "ventas_diarias.csv"

    daily.to_csv(output_file, index=False, encoding="utf-8-sig")

    print("\n========== DATASET GENERADO ==========")
    print("Archivo:", output_file)
    print("Registros:", len(daily))
    print("Sabores:", len(sabores))
    print("Primera fecha:", daily["fecha"].min())
    print("Última fecha:", daily["fecha"].max())

    print("\nVentas por año:")
    print(
        daily.assign(
            año=pd.to_datetime(daily["fecha"]).dt.year
        )
        .groupby("año")["cantidad_vendida"]
        .sum()
        .to_string()
    )

    print("\nPrimeros registros:")
    print(daily.head(15).to_string(index=False))


if __name__ == "__main__":
    build_daily_dataset()
