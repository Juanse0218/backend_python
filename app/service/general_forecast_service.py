
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

    if not isinstance(days, int) or isinstance(days, bool) or days < 1 or days > 30:
        raise ValueError(
            "Los días deben estar entre 1 y 30"
        )

    # ==========================================
    # GENERAR PREDICCIONES
    # ==========================================

    predictions = predict_all_flavors(days)

    if not predictions:
        raise ValueError(
            "No se generaron predicciones."
        )

    df = pd.DataFrame(predictions)

    df["fecha"] = pd.to_datetime(
        df["fecha"]
    )

    df["cantidad_predicha"] = pd.to_numeric(
        df["cantidad_predicha"],
        errors="coerce"
    ).fillna(0)

    # ==========================================
    # AGRUPACIÓN SEMANAL
    # ==========================================

    df["semana_inicio"] = (
        df["fecha"]
        - pd.to_timedelta(
            df["fecha"].dt.weekday,
            unit="D"
        )
    )

    weekly = (
        df.groupby(
            ["semana_inicio", "sabor"],
            as_index=False,
            sort=True
        )["cantidad_predicha"]
        .sum()
    )

    weekly_pivot = (
        weekly.pivot(
            index="semana_inicio",
            columns="sabor",
            values="cantidad_predicha"
        )
        .reindex(columns=FLAVORS, fill_value=0)
        .fillna(0)
        .round(2)
    )

    weekly_data = []

    for date, row in weekly_pivot.iterrows():

        values = {
            flavor: float(row[flavor])
            for flavor in FLAVORS
        }

        weekly_data.append({
            "fecha_inicio": date.strftime("%Y-%m-%d"),
            "demanda_total": round(
                sum(values.values()), 2
            ),
            "sabores": values
        })

    # ==========================================
    # AGRUPACIÓN MENSUAL
    # ==========================================

    df["mes_inicio"] = (
        df["fecha"]
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    monthly = (
        df.groupby(
            ["mes_inicio", "sabor"],
            as_index=False,
            sort=True
        )["cantidad_predicha"]
        .sum()
    )

    monthly_pivot = (
        monthly.pivot(
            index="mes_inicio",
            columns="sabor",
            values="cantidad_predicha"
        )
        .reindex(columns=FLAVORS, fill_value=0)
        .fillna(0)
        .round(2)
    )

    monthly_data = []

    for date, row in monthly_pivot.iterrows():

        values = {
            flavor: float(row[flavor])
            for flavor in FLAVORS
        }

        monthly_data.append({
            "mes": date.strftime("%Y-%m"),
            "demanda_total": round(
                sum(values.values()), 2
            ),
            "sabores": values
        })

    # ==========================================
    # RESPUESTA FINAL
    # ==========================================

    return {
        "total_sabores": len(FLAVORS),
        "dias_pronosticados": days,
        "predicciones_diarias": len(predictions),
        "sabores": FLAVORS,
        "semanal": weekly_data,
        "mensual": monthly_data
    }
