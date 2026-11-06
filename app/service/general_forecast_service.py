
import pandas as pd

from app.service.forecast_service import (
    FLAVORS,
    predict_all_flavors
)


def generate_general_forecast(days: int):
    """
    Genera predicciones para todos los sabores y
    agrupa la demanda semanal y mensual.
    """
    if not isinstance(days, int) or days < 1 or days > 30:
        raise ValueError("Los dias deben estar entre 1 y 30")

    # Generar predicciones de los 13 sabores.
    predictions = predict_all_flavors(days)

    if not predictions:
        raise ValueError("No se generaron predicciones.")

    df = pd.DataFrame(predictions)

    df["fecha"] = pd.to_datetime(df["fecha"])
    df["cantidad_predicha"] = pd.to_numeric(
        df["cantidad_predicha"],
        errors="coerce"
    ).fillna(0)

    # Agrupación semanal: lunes como inicio de semana.
    df["semana_inicio"] = (
        df["fecha"] -
        pd.to_timedelta(df["fecha"].dt.weekday, unit="D")
    )

    weekly = (
        df.groupby(
            ["semana_inicio", "sabor"],
            as_index=False
        )["cantidad_predicha"]
        .sum()
    )

    # Agrupación mensual.
    df["mes_inicio"] = (
        df["fecha"].dt.to_period("M").dt.to_timestamp()
    )

    monthly = (
        df.groupby(
            ["mes_inicio", "sabor"],
            as_index=False
        )["cantidad_predicha"]
        .sum()
    )

    # Construir respuesta semanal por sabor.
    weekly_data = []

    for date, group in weekly.groupby("semana_inicio"):
        values = {
            flavor: 0.0
            for flavor in FLAVORS
        }

        for _, row in group.iterrows():
            values[row["sabor"]] = round(
                float(row["cantidad_predicha"]), 2
            )

        weekly_data.append({
            "fecha_inicio": date.strftime("%Y-%m-%d"),
            "demanda_total": round(sum(values.values()), 2),
            "sabores": values
        })

    # Construir respuesta mensual por sabor.
    monthly_data = []

    for date, group in monthly.groupby("mes_inicio"):
        values = {
            flavor: 0.0
            for flavor in FLAVORS
        }

        for _, row in group.iterrows():
            values[row["sabor"]] = round(
                float(row["cantidad_predicha"]), 2
            )

        monthly_data.append({
            "mes": date.strftime("%Y-%m"),
            "demanda_total": round(sum(values.values()), 2),
            "sabores": values
        })

    return {
        "total_sabores": len(FLAVORS),
        "dias_pronosticados": days,
        "predicciones_diarias": len(predictions),
        "sabores": FLAVORS,
        "semanal": weekly_data,
        "mensual": monthly_data
    }
