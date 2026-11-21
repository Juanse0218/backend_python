
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List
from datetime import datetime
from zoneinfo import ZoneInfo

from app.service.forecast_service import (
    FLAVORS,
    predict_multiple_flavors
)

from app.service.general_forecast_service import (
    generate_general_forecast
)

from app.core.prediction_cache import (
    forecast_cache,
    general_forecast_cache
)


router = APIRouter(
    prefix="/api/predictions",
    tags=["Predicciones"]
)


class ForecastRequest(BaseModel):
    sabores: List[str] = Field(
        ...,
        min_length=1,
        description="Lista de sabores que se desean predecir"
    )

    dias: int = Field(
        ...,
        ge=1,
        le=30,
        description="Cantidad de días de predicción"
    )


class GeneralForecastRequest(BaseModel):
    dias: int = Field(
        ...,
        ge=1,
        le=30,
        description="Cantidad de días de predicción general"
    )


def get_local_date():
    return datetime.now(
        ZoneInfo("America/Bogota")
    ).strftime("%Y-%m-%d")


@router.get("/flavors")
def get_flavors():
    return {
        "sabores": FLAVORS,
        "total": len(FLAVORS)
    }


@router.post("/forecast")
def generate_forecast(request: ForecastRequest):
    try:
        sabores = [
            sabor.lower().strip()
            for sabor in request.sabores
        ]

        invalidos = [
            sabor for sabor in sabores
            if sabor not in FLAVORS
        ]

        if invalidos:
            raise HTTPException(
                status_code=400,
                detail=f"Sabores no disponibles: {invalidos}"
            )

        if len(sabores) != len(set(sabores)):
            raise HTTPException(
                status_code=400,
                detail="No se permiten sabores repetidos"
            )

        fecha = get_local_date()

        # Se conserva el orden de sabores porque también
        # determina el orden de la respuesta.
        cache_key = (
            f"forecast:{fecha}:{','.join(sabores)}:{request.dias}"
        )

        result, cache_hit = forecast_cache.get_or_compute(
            cache_key,
            lambda: {
                "total_predicciones": len(sabores) * request.dias,
                "dias": request.dias,
                "sabores": sabores,
                "predicciones": predict_multiple_flavors(
                    sabores,
                    request.dias
                )
            }
        )

        return result

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Error generando predicciones: {str(error)}"
        )


@router.post("/general-forecast")
def generate_general_forecast_endpoint(
    request: GeneralForecastRequest
):
    try:
        fecha = get_local_date()

        cache_key = (
            f"general_forecast:{fecha}:{request.dias}"
        )

        result, cache_hit = (
            general_forecast_cache.get_or_compute(
                cache_key,
                lambda: generate_general_forecast(
                    request.dias
                )
            )
        )

        return result

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Error generando demanda general: {str(error)}"
        )
