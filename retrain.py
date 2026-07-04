"""
PAIOL — Re-entrenamiento Automático del Modelo

Flujo mensual:
1. Recopila datos reales (precios + sentiment) de los últimos N días
2. Genera features y labels reales (¿subió o bajó al día siguiente?)
3. Entrena modelo nuevo con los datos acumulados
4. Compara nuevo vs actual en un test set reciente
5. Si el nuevo es mejor → reemplaza. Si no → mantiene.

Aprende de:
- Errores del paper trading
- Sentiment real de Gemini (no sintético)
- Nuevos patrones del mercado

Uso:
    from retrain import AutoRetrainer

    ar = AutoRetrainer()
    ar.run()  # Evalúa si hay que re-entrenar y lo hace

    # Forzar re-entrenamiento
    ar.run(forzar=True)

    # Solo evaluar sin reemplazar
    ar.evaluar()
"""

import os
import json
import logging
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
import xgboost as xgb
from sklearn.metrics import f1_score, accuracy_score, classification_report

from features import PaiolFeatures, FEATURE_COLS, TICKERS, UMBRAL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PAIOL-Retrain")

# Configuración
MODEL_DIR = Path("model")
MODEL_PATH = MODEL_DIR / "xgboost_model.json"
MODEL_BACKUP_DIR = MODEL_DIR / "backups"
SENTIMENT_DIR = Path("outputs")
RETRAIN_LOG = Path("retrain_log.json")
MIN_DIAS_PARA_REENTRENAR = 30
MIN_OPERACIONES = 20


class AutoRetrainer:
    """Re-entrenamiento automático del modelo PAIOL."""

    def __init__(
        self,
        model_path: str = None,
        lookback_days: int = 365,
        test_days: int = 30,
        min_mejora_f1: float = 0.02,
    ):
        """
        Args:
            model_path: Ruta al modelo actual
            lookback_days: Días de historia para entrenar
            test_days: Últimos N días como test set
            min_mejora_f1: Mejora mínima en F1 para reemplazar (0.02 = 2%)
        """
        self.model_path = Path(model_path or MODEL_PATH)
        self.lookback_days = lookback_days
        self.test_days = test_days
        self.min_mejora_f1 = min_mejora_f1
        self.pf = PaiolFeatures(lookback_days=lookback_days + 60)

        MODEL_DIR.mkdir(exist_ok=True)
        MODEL_BACKUP_DIR.mkdir(exist_ok=True)

        # Cargar historial de re-entrenamientos
        if RETRAIN_LOG.exists():
            with open(RETRAIN_LOG, "r") as f:
                self.log = json.load(f)
        else:
            self.log = {"entrenamientos": [], "ultimo": None}

    # ================================================================
    # RECOPILACIÓN DE DATOS REALES
    # ================================================================

    def recopilar_datos(self) -> pd.DataFrame:
        """
        Descarga precios históricos y genera features + labels reales
        para todos los tickers PAIOL.

        Returns:
            DataFrame con features y labels listo para entrenar
        """
        logger.info(f"Recopilando {self.lookback_days} días de datos...")
        all_data = []

        for ticker, config in TICKERS.items():
            try:
                # Descargar precios
                df = self.pf.descargar_precios(config["yfinance"])

                # Features técnicos
                df = self.pf.calcular_tecnicos(df)

                # Agregar sentiment real acumulado (si existe)
                df = self._agregar_sentiment_historico(df, ticker)

                # Labels
                df = PaiolFeatures.generar_labels(df, umbral=UMBRAL)

                # Agregar ticker
                df["ticker"] = ticker

                # Limpiar
                df = df.dropna(subset=FEATURE_COLS + ["label"])

                all_data.append(df)
                logger.info(f"  {ticker}: {len(df)} días con features completos")

            except Exception as e:
                logger.error(f"  {ticker}: error — {e}")

        if not all_data:
            raise ValueError("No se pudieron recopilar datos para ningún ticker")

        dataset = pd.concat(all_data, ignore_index=True)
        logger.info(f"Dataset total: {len(dataset)} filas, {dataset['ticker'].nunique()} tickers")

        return dataset

    def _agregar_sentiment_historico(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Busca archivos de sentiment guardados y los mergea con el DataFrame.
        Si no hay sentiment para una fecha, usa 0.0.
        """
        df = df.copy()

        # Buscar archivos de resultado con sentiment real
        sentiment_por_fecha = {}
        for f in SENTIMENT_DIR.glob("resultado_*.json"):
            try:
                with open(f, "r") as fh:
                    data = json.load(fh)
                sent = data.get("sentiment", {})
                fecha = sent.get("fecha", "")
                if fecha:
                    macro = sent.get("macro", {})
                    instr = sent.get("instrumentos", {}).get(ticker, {})
                    sentiment_por_fecha[fecha] = {
                        "dolar_sentiment": macro.get("dolar_sentiment", 0.0),
                        "reservas_sentiment": macro.get("reservas_sentiment", 0.0),
                        "riesgo_pais_sentiment": macro.get("riesgo_pais_sentiment", 0.0),
                        "ticker_sentiment": instr.get("sentimiento", 0.0),
                        "ticker_relevancia": instr.get("relevancia", 0.0),
                    }
            except:
                continue

        # Mergear con el dataframe
        for col in ["dolar_sentiment", "reservas_sentiment", "riesgo_pais_sentiment",
                     "ticker_sentiment", "ticker_relevancia"]:
            df[col] = 0.0

        for idx, row in df.iterrows():
            fecha_str = pd.Timestamp(row["date"]).strftime("%Y-%m-%d")
            if fecha_str in sentiment_por_fecha:
                for col, val in sentiment_por_fecha[fecha_str].items():
                    df.at[idx, col] = val

        n_con_sent = len([f for f in df["dolar_sentiment"] if f != 0.0])
        logger.info(f"    {ticker}: {n_con_sent}/{len(df)} días con sentiment real")

        return df

    # ================================================================
    # ENTRENAMIENTO
    # ================================================================

    def entrenar_modelo(self, dataset: pd.DataFrame) -> tuple:
        """
        Entrena un nuevo modelo con split temporal.

        Returns:
            (modelo_nuevo, metricas_nuevo, metricas_viejo)
        """
        # Split temporal
        dataset = dataset.sort_values("date")
        cutoff = dataset["date"].max() - pd.Timedelta(days=self.test_days)

        train = dataset[dataset["date"] < cutoff]
        test = dataset[dataset["date"] >= cutoff]

        X_train = train[FEATURE_COLS].astype(float)
        y_train = train["label"].astype(int)
        X_test = test[FEATURE_COLS].astype(float)
        y_test = test["label"].astype(int)

        logger.info(f"Train: {len(train)} filas ({train['date'].min().date()} → {train['date'].max().date()})")
        logger.info(f"Test:  {len(test)} filas ({test['date'].min().date()} → {test['date'].max().date()})")
        logger.info(f"Distribución test: {dict(y_test.value_counts().sort_index())}")

        # Entrenar modelo nuevo con Optuna-like params (mejores del TP)
        modelo_nuevo = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            random_state=42,
            use_label_encoder=False,
        )
        modelo_nuevo.fit(X_train, y_train)

        # Evaluar modelo nuevo
        y_pred_nuevo = modelo_nuevo.predict(X_test)
        f1_nuevo = f1_score(y_test, y_pred_nuevo, average="weighted")
        acc_nuevo = accuracy_score(y_test, y_pred_nuevo)

        metricas_nuevo = {
            "f1_weighted": round(f1_nuevo, 4),
            "accuracy": round(acc_nuevo, 4),
            "train_size": len(train),
            "test_size": len(test),
        }

        logger.info(f"\n📊 Modelo NUEVO: F1={f1_nuevo:.4f}, Acc={acc_nuevo:.4f}")
        logger.info(classification_report(y_test, y_pred_nuevo,
                    target_names=["Baja", "Lateral", "Sube"]))

        # Evaluar modelo actual
        metricas_viejo = {"f1_weighted": 0.0, "accuracy": 0.0}
        try:
            if self.model_path.exists():
                modelo_viejo = xgb.XGBClassifier()
                modelo_viejo.load_model(str(self.model_path))
                y_pred_viejo = modelo_viejo.predict(X_test)
                f1_viejo = f1_score(y_test, y_pred_viejo, average="weighted")
                acc_viejo = accuracy_score(y_test, y_pred_viejo)
                metricas_viejo = {
                    "f1_weighted": round(f1_viejo, 4),
                    "accuracy": round(acc_viejo, 4),
                }
                logger.info(f"📊 Modelo ACTUAL: F1={f1_viejo:.4f}, Acc={acc_viejo:.4f}")
        except Exception as e:
            logger.warning(f"No se pudo evaluar modelo actual: {e}")

        return modelo_nuevo, metricas_nuevo, metricas_viejo

    # ================================================================
    # DECISIÓN: ¿REEMPLAZAR?
    # ================================================================

    def decidir_reemplazo(self, metricas_nuevo: dict, metricas_viejo: dict) -> bool:
        """
        Decide si el modelo nuevo es suficientemente mejor.

        Criterios:
        - F1 nuevo > F1 viejo + min_mejora_f1
        - O F1 viejo es 0 (no hay modelo actual)
        """
        f1_nuevo = metricas_nuevo["f1_weighted"]
        f1_viejo = metricas_viejo["f1_weighted"]

        if f1_viejo == 0:
            logger.info("No hay modelo previo — reemplazando")
            return True

        mejora = f1_nuevo - f1_viejo
        logger.info(f"\nComparación: nuevo={f1_nuevo:.4f} vs actual={f1_viejo:.4f} "
                    f"(mejora={mejora:+.4f}, mínima requerida={self.min_mejora_f1})")

        if mejora >= self.min_mejora_f1:
            logger.info(f"✅ Modelo nuevo es mejor — reemplazando")
            return True
        else:
            logger.info(f"❌ Modelo nuevo no supera al actual — manteniendo")
            return False

    def reemplazar_modelo(self, modelo_nuevo: xgb.XGBClassifier, metricas: dict):
        """Backupea el modelo actual y guarda el nuevo."""
        # Backup del actual
        if self.model_path.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            backup_path = MODEL_BACKUP_DIR / f"xgboost_model_{timestamp}.json"
            shutil.copy(self.model_path, backup_path)
            logger.info(f"Modelo anterior backupeado en {backup_path}")

        # Guardar nuevo
        modelo_nuevo.save_model(str(self.model_path))
        logger.info(f"Modelo nuevo guardado en {self.model_path}")

        # Registrar en log
        entrada = {
            "fecha": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "metricas": metricas,
            "modelo_path": str(self.model_path),
        }
        self.log["entrenamientos"].append(entrada)
        self.log["ultimo"] = entrada
        with open(RETRAIN_LOG, "w") as f:
            json.dump(self.log, f, indent=2)

    # ================================================================
    # ANÁLISIS DE ERRORES
    # ================================================================

    def analizar_errores(self, dataset: pd.DataFrame) -> str:
        """
        Analiza en qué se equivoca el modelo actual.
        Ayuda a entender qué features faltan o qué ajustar.
        """
        if not self.model_path.exists():
            return "No hay modelo para analizar."

        model = xgb.XGBClassifier()
        model.load_model(str(self.model_path))

        dataset = dataset.sort_values("date")
        cutoff = dataset["date"].max() - pd.Timedelta(days=self.test_days)
        test = dataset[dataset["date"] >= cutoff].copy()

        X_test = test[FEATURE_COLS].astype(float)
        y_test = test["label"].astype(int)
        y_pred = model.predict(X_test)

        test["pred"] = y_pred
        test["acierto"] = test["label"] == test["pred"]

        lines = [
            f"\n{'='*60}",
            f"  ANÁLISIS DE ERRORES",
            f"{'='*60}",
        ]

        # Por ticker
        lines.append("\n📊 Aciertos por ticker:")
        for ticker in test["ticker"].unique():
            t = test[test["ticker"] == ticker]
            wr = t["acierto"].mean()
            lines.append(f"  {ticker}: {wr:.1%} ({len(t)} ops)")

        # Por clase
        lines.append("\n📊 Aciertos por clase predicha:")
        labels = {0: "BAJA", 1: "LATERAL", 2: "SUBE"}
        for clase in [0, 1, 2]:
            c = test[test["pred"] == clase]
            if len(c) > 0:
                wr = c["acierto"].mean()
                lines.append(f"  {labels[clase]}: {wr:.1%} ({len(c)} ops)")

        # Por volatilidad
        lines.append("\n📊 Aciertos por volatilidad:")
        test["vol_bucket"] = pd.qcut(test["volatility_5d"], q=3,
                                      labels=["baja", "media", "alta"],
                                      duplicates="drop")
        for bucket in ["baja", "media", "alta"]:
            b = test[test["vol_bucket"] == bucket]
            if len(b) > 0:
                wr = b["acierto"].mean()
                lines.append(f"  Vol {bucket}: {wr:.1%} ({len(b)} ops)")

        # Features más importantes
        importance = model.get_booster().get_score(importance_type="gain")
        top_features = sorted(importance, key=importance.get, reverse=True)[:5]
        lines.append("\n📊 Top 5 features por importancia:")
        for f in top_features:
            lines.append(f"  {f}: {importance[f]:.2f}")

        # Recomendaciones
        lines.append("\n💡 RECOMENDACIONES:")

        overall_wr = test["acierto"].mean()
        if overall_wr < 0.40:
            lines.append("  ⚠️ Win rate muy bajo — considerar más features o ajustar umbral")

        sent_features = [f for f in top_features if "sentiment" in f]
        if not sent_features:
            lines.append("  ⚠️ Sentiment no aparece en top features — puede que los datos"
                        " sintéticos no aporten. Re-entrenar con sentiment real debería mejorar")

        if test[test["pred"] == 1]["acierto"].mean() > 0.7:
            lines.append("  ℹ️ LATERAL tiene alto acierto — el modelo es conservador."
                        " Podés bajar el umbral de 1% a 0.5% para más señales")

        lines.append(f"\n{'='*60}")
        return "\n".join(lines)

    # ================================================================
    # PIPELINE PRINCIPAL
    # ================================================================

    def debe_reentrenar(self) -> bool:
        """Verifica si pasó suficiente tiempo desde el último re-entrenamiento."""
        if not self.log.get("ultimo"):
            return True

        ultimo = datetime.strptime(self.log["ultimo"]["fecha"], "%Y-%m-%d %H:%M")
        dias = (datetime.now() - ultimo).days
        logger.info(f"Último re-entrenamiento: hace {dias} días")

        return dias >= MIN_DIAS_PARA_REENTRENAR

    def run(self, forzar: bool = False):
        """
        Pipeline completo de re-entrenamiento.

        Args:
            forzar: True para forzar re-entrenamiento sin importar el tiempo
        """
        logger.info("=" * 60)
        logger.info("  PAIOL Auto-Retrainer")
        logger.info("=" * 60)

        # Verificar si toca re-entrenar
        if not forzar and not self.debe_reentrenar():
            logger.info("No toca re-entrenar todavía. Usá --forzar para forzar.")
            return

        # 1. Recopilar datos
        logger.info("\n📥 PASO 1: Recopilando datos reales...")
        dataset = self.recopilar_datos()

        # 2. Analizar errores del modelo actual
        logger.info("\n🔍 PASO 2: Analizando errores del modelo actual...")
        print(self.analizar_errores(dataset))

        # 3. Entrenar modelo nuevo
        logger.info("\n🏋️ PASO 3: Entrenando modelo nuevo...")
        modelo_nuevo, metricas_nuevo, metricas_viejo = self.entrenar_modelo(dataset)

        # 4. Decidir si reemplazar
        logger.info("\n⚖️ PASO 4: Comparando modelos...")
        if self.decidir_reemplazo(metricas_nuevo, metricas_viejo):
            self.reemplazar_modelo(modelo_nuevo, metricas_nuevo)
            logger.info("\n🎉 Modelo actualizado exitosamente")
        else:
            logger.info("\n📌 Modelo actual mantenido")

        logger.info("\n✅ Re-entrenamiento completado")

    def evaluar(self):
        """Solo evalúa el modelo actual sin re-entrenar."""
        dataset = self.recopilar_datos()
        print(self.analizar_errores(dataset))


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PAIOL Auto-Retrainer")
    parser.add_argument("--forzar", action="store_true",
                       help="Forzar re-entrenamiento sin esperar 30 días")
    parser.add_argument("--evaluar", action="store_true",
                       help="Solo evaluar modelo actual, sin re-entrenar")
    parser.add_argument("--analisis", action="store_true",
                       help="Solo mostrar análisis de errores")
    args = parser.parse_args()

    ar = AutoRetrainer()

    if args.evaluar or args.analisis:
        ar.evaluar()
    else:
        ar.run(forzar=args.forzar)
