
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List

from app.service.forecast_service import (
    FLAVORS,
    predict_multiple_flavors
)

from app.service.general_forecast_service import (
    generate_general_forecast
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
        le=365,
        description="Cantidad de días de predicción"
    )


class GeneralForecastRequest(BaseModel):
    dias: int = Field(
        ...,
        ge=1,
        le=365,
        description="Cantidad de días de predicción general"
    )


@router.get("/flavors")
def get_flavors():
    return {
        "sabores": FLAVORS,
        "total": len(FLAVORS)
    }


@router.post("/forecast")
def generate_forecast(request: ForecastRequest):

    try:
        sabores = [sabor.lower().strip() for sabor in request.sabores]

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

        predictions = predict_multiple_flavors(
            sabores,
            request.dias
        )

        return {
            "total_predicciones": len(predictions),
            "dias": request.dias,
            "sabores": sabores,
            "predicciones": predictions
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Error generando predicciones: {str(error)}"
        )


# NUEVO ENDPOINT: DEMANDA GENERAL
@router.post("/general-forecast")
def generate_general_forecast_endpoint(
    request: GeneralForecastRequest
):

    try:
        result = generate_general_forecast(
            request.dias
        )

        return result

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Error generando demanda general: {str(error)}"
        )
