
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.prediction_routes import router as prediction_router

app = FastAPI(
    title="ICE CREAM - API de Predicciones",
    description="API para consultar predicciones de demanda de helados mediante modelos LSTM.",
    version="1.0.0"
)

# Permitir solicitudes desde Flutter Web y aplicaciones móviles.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Registrar rutas de predicción.
app.include_router(prediction_router)


@app.get("/")
def root():
    return {
        "mensaje": "API ICE CREAM funcionando correctamente",
        "estado": "activo"
    }


@app.get("/health")
def health():
    return {
        "estado": "ok",
        "servicio": "ICE CREAM API"
    }
