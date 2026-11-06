import pandas as pd

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import HistGradientBoostingRegressor


FEATURES = [
    "dia_semana",
    "dia_mes",
    "mes",
    "semana_anio",
    "dia_anio",
    "sin_dia_anio",
    "cos_dia_anio",
    "sin_mes",
    "cos_mes",
    "es_fin_semana",
    "rezago_1",
    "rezago_7",
    "rezago_14",
    "promedio_7",
    "promedio_30",
]


def train_model():

    file = Path("data/ventas_features.csv")

    df = pd.read_csv(file, parse_dates=["fecha"])

    # Codificar los sabores
    encoder = LabelEncoder()
    df["sabor_codigo"] = encoder.fit_transform(df["sabor"])

    features = FEATURES + ["sabor_codigo"]

    # Separar entrenamiento y prueba cronológicamente
    fecha_corte = pd.Timestamp("2024-07-01")

    train = df[df["fecha"] < fecha_corte].copy()
    test = df[df["fecha"] >= fecha_corte].copy()

    X_train = train[features]
    y_train = train["cantidad_vendida"]

    X_test = test[features]
    y_test = test["cantidad_vendida"]

    # Crear modelo
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )

    print("Entrenando modelo...")

    model.fit(X_train, y_train)

    # Predicciones
    predictions = model.predict(X_test)

    # Evitar valores negativos
    predictions = predictions.clip(min=0)

    # Métricas
    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5

        # Comparar con modelos de referencia
    baselines = {
        "Ayer (rezago_1)": test["rezago_1"],
        "Semana anterior (rezago_7)": test["rezago_7"],
        "Promedio 30 días": test["promedio_30"]
    }

    print("\n========== COMPARACIÓN DE MODELOS ==========")

    print(f"{'Modelo':<30} {'MAE':<12} {'RMSE':<12}")

    print(
        f"{'Random Forest':<30} "
        f"{mae:<12.4f} {rmse:<12.4f}"
    )

    for nombre, prediccion_base in baselines.items():

        error_mae = mean_absolute_error(
            y_test,
            prediccion_base
        )

        error_rmse = mean_squared_error(
            y_test,
            prediccion_base
        ) ** 0.5

        print(
            f"{nombre:<30} "
            f"{error_mae:<12.4f} {error_rmse:<12.4f}"
        )

    print("\n========== RESULTADOS DEL MODELO ==========")

    print("Registros de entrenamiento:", len(train))
    print("Registros de prueba:", len(test))

    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")

    # Comparación de resultados
    resultados = test[["fecha", "sabor", "cantidad_vendida"]].copy()

    resultados["prediccion"] = predictions.round(2)

    print("\nPrimeras predicciones:")
    print(resultados.head(15).to_string(index=False))

    # Importancia de características
    importancia = pd.DataFrame({
        "caracteristica": features,
        "importancia": model.feature_importances_
    }).sort_values(
        by="importancia",
        ascending=False
    )

    print("\nImportancia de características:")
    print(importancia.to_string(index=False))

        # Segundo modelo: HistGradientBoosting
    boosting = HistGradientBoostingRegressor(
        max_iter=200,
        learning_rate=0.05,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=42
    )

    print("\nEntrenando HistGradientBoosting...")

    boosting.fit(X_train, y_train)

    pred_boosting = boosting.predict(X_test)
    pred_boosting = pred_boosting.clip(min=0)

        # Evaluación individual por sabor
    resultados = test[["fecha", "sabor", "cantidad_vendida"]].copy()

    resultados["prediccion"] = pred_boosting

    print("\n========== EVALUACIÓN POR SABOR ==========")

    metricas_sabor = []

    for sabor, grupo in resultados.groupby("sabor"):

        mae_sabor = mean_absolute_error(
            grupo["cantidad_vendida"],
            grupo["prediccion"]
        )

        rmse_sabor = mean_squared_error(
            grupo["cantidad_vendida"],
            grupo["prediccion"]
        ) ** 0.5

        metricas_sabor.append({
            "sabor": sabor,
            "MAE": round(mae_sabor, 4),
            "RMSE": round(rmse_sabor, 4)
        })

    metricas_df = pd.DataFrame(metricas_sabor)

    print(
        metricas_df.sort_values("MAE").to_string(index=False)
    )

    mae_boosting = mean_absolute_error(y_test, pred_boosting)

    rmse_boosting = mean_squared_error(
        y_test,
        pred_boosting
    ) ** 0.5

    print("\n========== COMPARACIÓN DE ALGORITMOS ==========")

    print(f"Random Forest MAE: {mae:.4f}")
    print(f"Random Forest RMSE: {rmse:.4f}")

    print(f"HistGradientBoosting MAE: {mae_boosting:.4f}")
    print(f"HistGradientBoosting RMSE: {rmse_boosting:.4f}")



if __name__ == "__main__":
    train_model()

