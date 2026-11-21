
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.prediction_routes import router as prediction_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Controla el inicio y cierre de la aplicación.
    No carga los modelos LSTM anticipadamente.
    """

    start_time = time.perf_counter()

    print("======================================")
    print("Iniciando ICE CREAM API...")
    print("======================================")

    yield

    elapsed = time.perf_counter() - start_time

    print("======================================")
    print("Cerrando ICE CREAM API...")
    print(f"Tiempo de vida: {elapsed:.2f} segundos")
    print("======================================")


app = FastAPI(
    title="ICE CREAM - API de Predicciones",
    description=(
        "API para consultar predicciones de demanda "
        "de helados mediante modelos LSTM."
    ),
    version="1.0.0",
    lifespan=lifespan
)


# ==========================================
# CONFIGURACIÓN CORS
# ==========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# REGISTRO DE RUTAS
# ==========================================

app.include_router(prediction_router)


# ==========================================
# RUTA PRINCIPAL
# ==========================================

@app.get("/")
def root():
    return {
        "mensaje": "API ICE CREAM funcionando correctamente",
        "estado": "activo"
    }


# ==========================================
# VERIFICACIÓN DE SALUD
# ==========================================

@app.get("/health")
def health():
    return {
        "estado": "ok",
        "servicio": "ICE CREAM API"
    }
