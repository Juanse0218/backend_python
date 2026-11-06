
import os
from pymongo import MongoClient
from dotenv import load_dotenv

# Cargar variables del archivo .env
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DATABASE = os.getenv("MONGO_DATABASE")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION")

# Crear conexión
client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=10000)

# Seleccionar base de datos y colección
database = client[MONGO_DATABASE]
sales_collection = database[MONGO_COLLECTION]


def test_connection():
    try:
        client.admin.command("ping")
        print("Conexión exitosa con MongoDB Atlas")

        print(f"Base de datos: {MONGO_DATABASE}")
        print(f"Colección: {MONGO_COLLECTION}")

        return True

    except Exception as error:
        print(f"Error de conexión: {error}")
        return False
