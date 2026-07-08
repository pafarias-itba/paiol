"""
PAIOL — Pipeline Diario de Trading

Orquesta todo el flujo diario:
1. Scrape noticias RSS → Gemini → sentiment JSON
2. yfinance → features técnicos (pandas)
3. Modelo XGBoost → predicción por ticker
4. Señal → IOL API → orden de compra/venta
5. Reporte por email

Corre cada mañana antes de la apertura del mercado argentino (10:00 ART).

Uso:
    # Ejecución manual
    python pipeline.py

    # Con modo dry-run (no opera en real)
    python pipeline.py --dry-run

    # Solo un ticker
    python pipeline.py --ticker GGAL
"""

import os
import sys
import json
import logging
import argparse
import smtplib
import pickle
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb

from scraper_sentiment import PaiolSentiment
from features import PaiolFeatures, FEATURE_COLS, TICKERS
from iol_client import IOLClient

from dotenv import load_dotenv
load_dotenv(override=True)  # el .env pisa exports viejos del shell (ej: keys vencidas)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("PAIOL-Pipeline")


# ================================================================
# CONFIGURACIÓN
# ================================================================

class Config:
    """Configuración centralizada desde variables de entorno."""

    # IOL
    IOL_USERNAME = os.environ.get("IOL_USERNAME")
    IOL_PASSWORD = os.environ.get("IOL_PASSWORD")
    IOL_SANDBOX = os.environ.get("IOL_SANDBOX", "true").lower() == "true"

    # Gemini
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

    # Email para reportes
    EMAIL_FROM = os.environ.get("PAIOL_EMAIL_FROM")
    EMAIL_TO = os.environ.get("PAIOL_EMAIL_TO")
    EMAIL_PASSWORD = os.environ.get("PAIOL_EMAIL_PASSWORD")
    EMAIL_SMTP_HOST = os.environ.get("PAIOL_SMTP_HOST", "smtp.gmail.com")
    EMAIL_SMTP_PORT = int(os.environ.get("PAIOL_SMTP_PORT", "587"))

    # Modelo
    MODEL_PATH = os.environ.get("PAIOL_MODEL_PATH", "model/xgboost_model.json")

    # Trading
    CANTIDAD_DEFAULT = int(os.environ.get("PAIOL_CANTIDAD_DEFAULT", "10"))
    CONFIANZA_MINIMA = float(os.environ.get("PAIOL_CONFIANZA_MINIMA", "0.5"))

    # Outputs
    OUTPUT_DIR = os.environ.get("PAIOL_OUTPUT_DIR", "outputs")

    @classmethod
    def validar(cls, dry_run: bool = False):
        """Valida que las credenciales necesarias estén configuradas."""
        errores = []

        if not cls.GEMINI_API_KEY:
            errores.append("GEMINI_API_KEY")
        if not dry_run:
            if not cls.IOL_USERNAME:
                errores.append("IOL_USERNAME")
            if not cls.IOL_PASSWORD:
                errores.append("IOL_PASSWORD")

        if errores:
            raise EnvironmentError(
                f"Variables de entorno faltantes: {', '.join(errores)}\n"
                "Configuralas en tu archivo .env o en el entorno."
            )


# ================================================================
# CARGA DEL MODELO
# ================================================================

def cargar_modelo(path: str) -> xgb.XGBClassifier:
    """
    Carga el modelo entrenado exportado desde Databricks/Kaggle.

    Soporta:
    - .json (XGBoost nativo, recomendado)
    - .pkl (pickle, legacy)
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Modelo no encontrado en {path}\n"
            "Exportalo desde Databricks con:\n"
            "  model.save_model('xgboost_model.json')"
        )

    model = xgb.XGBClassifier()

    if path.suffix == ".json":
        model.load_model(str(path))
        logger.info(f"Modelo cargado desde {path} (XGBoost JSON)")
    elif path.suffix == ".pkl":
        with open(path, "rb") as f:
            model = pickle.load(f)
        logger.info(f"Modelo cargado desde {path} (pickle)")
    else:
        raise ValueError(f"Formato no soportado: {path.suffix}. Usá .json o .pkl")

    return model


# ================================================================
# PREDICCIÓN
# ================================================================

LABELS = {0: "BAJA", 1: "LATERAL", 2: "SUBE"}
ACCIONES = {0: "VENDER", 1: "HOLD", 2: "COMPRAR"}


def predecir(model: xgb.XGBClassifier, features: pd.Series) -> dict:
    """
    Predice con el modelo y devuelve señal + probabilidades.

    Returns:
        {
            "prediccion": 0|1|2,
            "label": "BAJA"|"LATERAL"|"SUBE",
            "accion": "VENDER"|"HOLD"|"COMPRAR",
            "probabilidades": {"BAJA": 0.2, "LATERAL": 0.3, "SUBE": 0.5},
            "confianza": 0.5  # probabilidad de la clase ganadora
        }
    """
    X = features[FEATURE_COLS].astype(float).values.reshape(1, -1)
    X_df = pd.DataFrame(X, columns=FEATURE_COLS)

    pred = model.predict(X_df)[0]
    proba = model.predict_proba(X_df)[0]

    return {
        "prediccion": int(pred),
        "label": LABELS[int(pred)],
        "accion": ACCIONES[int(pred)],
        "probabilidades": {
            "BAJA": round(float(proba[0]), 4),
            "LATERAL": round(float(proba[1]), 4),
            "SUBE": round(float(proba[2]), 4),
        },
        "confianza": round(float(max(proba)), 4),
    }


# ================================================================
# REPORTE POR EMAIL
# ================================================================

def generar_reporte(resultados: dict, sentiment: dict, cuenta: dict = None) -> str:
    """Genera el reporte diario en texto plano."""
    fecha = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"{'='*60}",
        f"  PAIOL — Reporte Diario  {fecha}",
        f"{'='*60}",
        "",
    ]

    # Sentiment macro
    macro = sentiment.get("macro", {})
    lines.append("📊 SENTIMENT MACRO")
    lines.append(f"  Dólar:       {macro.get('dolar_sentiment', 'N/A'):>6}")
    lines.append(f"  Reservas:    {macro.get('reservas_sentiment', 'N/A'):>6}")
    lines.append(f"  Riesgo País: {macro.get('riesgo_pais_sentiment', 'N/A'):>6}")
    lines.append("")

    # Eventos clave
    eventos = sentiment.get("eventos_clave", [])
    if eventos:
        lines.append("📰 EVENTOS CLAVE")
        for ev in eventos:
            lines.append(f"  • {ev}")
        lines.append("")

    # Predicciones y acciones
    lines.append("🎯 PREDICCIONES")
    lines.append(f"  {'Ticker':<8} {'Señal':<10} {'Acción':<10} {'Confianza':<10} {'Ejecutada'}")
    lines.append(f"  {'-'*55}")

    for ticker, res in resultados.items():
        pred = res.get("prediccion", {})
        orden = res.get("orden", None)
        ejecutada = "✅ SÍ" if orden else "❌ NO"
        if pred.get("accion") == "HOLD":
            ejecutada = "➖ N/A"

        lines.append(
            f"  {ticker:<8} {pred.get('label', 'N/A'):<10} "
            f"{pred.get('accion', 'N/A'):<10} "
            f"{pred.get('confianza', 0):<10.1%} {ejecutada}"
        )

    lines.append("")

    # Detalle de probabilidades
    lines.append("📈 DETALLE DE PROBABILIDADES")
    for ticker, res in resultados.items():
        pred = res.get("prediccion", {})
        proba = pred.get("probabilidades", {})
        lines.append(
            f"  {ticker}: ↓{proba.get('BAJA', 0):.1%}  "
            f"→{proba.get('LATERAL', 0):.1%}  "
            f"↑{proba.get('SUBE', 0):.1%}"
        )

    lines.append("")

    # Estado de cuenta
    if cuenta:
        lines.append("💰 CUENTA")
        lines.append(f"  {json.dumps(cuenta, indent=4, default=str)}")
        lines.append("")

    lines.append(f"{'='*60}")
    lines.append(f"  PAIOL v0.1 | Sandbox: {Config.IOL_SANDBOX}")
    lines.append(f"{'='*60}")

    return "\n".join(lines)


def enviar_email(reporte: str):
    """Envía el reporte por email."""
    if not all([Config.EMAIL_FROM, Config.EMAIL_TO, Config.EMAIL_PASSWORD]):
        logger.warning("Email no configurado, saltando envío")
        return

    msg = MIMEMultipart()
    msg["From"] = Config.EMAIL_FROM
    msg["To"] = Config.EMAIL_TO
    msg["Subject"] = f"PAIOL Reporte — {datetime.now().strftime('%Y-%m-%d')}"
    msg.attach(MIMEText(reporte, "plain"))

    try:
        # timeout corto: muchos hosts cloud bloquean SMTP saliente y sin esto
        # el envío cuelga el job ~2 min antes de fallar
        with smtplib.SMTP(Config.EMAIL_SMTP_HOST, Config.EMAIL_SMTP_PORT,
                          timeout=15) as server:
            server.starttls()
            server.login(Config.EMAIL_FROM, Config.EMAIL_PASSWORD)
            server.send_message(msg)
        logger.info("Reporte enviado por email")
    except Exception as e:
        logger.error(f"Error enviando email: {e}")


# ================================================================
# PIPELINE PRINCIPAL
# ================================================================

def run_pipeline(
    dry_run: bool = False,
    ticker_filter: Optional[str] = None,
    skip_sentiment: bool = False,
):
    """
    Ejecuta el pipeline completo de PAIOL.

    Args:
        dry_run: True = no ejecuta órdenes reales
        ticker_filter: Operar solo un ticker específico
        skip_sentiment: Saltear scraping/Gemini (usa sentiment vacío)
    """
    start_time = datetime.now()
    logger.info("=" * 60)
    logger.info("  PAIOL Pipeline — Inicio")
    logger.info(f"  Modo: {'DRY RUN' if dry_run else 'PRODUCCIÓN'}")
    logger.info(f"  Sandbox IOL: {Config.IOL_SANDBOX}")
    logger.info("=" * 60)

    # Validar configuración
    Config.validar(dry_run=dry_run)

    # Determinar tickers a operar
    tickers_a_operar = {ticker_filter: TICKERS[ticker_filter]} if ticker_filter else TICKERS

    resultados = {}

    # ----- PASO 1: SENTIMENT -----
    logger.info("\n📰 PASO 1: Análisis de sentimiento")
    if skip_sentiment:
        logger.info("   Saltando sentiment (--skip-sentiment)")
        sentiment = {
            "macro": {},
            "instrumentos": {},
            "eventos_clave": [],
        }
    else:
        try:
            ps = PaiolSentiment()
            sentiment = ps.run()
        except Exception as e:
            logger.error(f"   Error en sentiment: {e}")
            logger.info("   Continuando sin sentiment...")
            sentiment = {"macro": {}, "instrumentos": {}, "eventos_clave": []}

    # ----- PASO 2: FEATURES -----
    logger.info("\n📊 PASO 2: Feature engineering")
    pf = PaiolFeatures()
    all_features = pf.build_all(sentiment)

    # ----- PASO 3: MODELO -----
    logger.info("\n🎯 PASO 3: Predicción")
    model = cargar_modelo(Config.MODEL_PATH)

    for ticker in tickers_a_operar:
        if ticker not in all_features:
            logger.warning(f"   {ticker}: sin features, saltando")
            resultados[ticker] = {"prediccion": {}, "orden": None, "error": "sin features"}
            continue

        features = all_features[ticker]
        pred = predecir(model, features)

        logger.info(
            f"   {ticker}: {pred['label']} ({pred['accion']}) "
            f"confianza={pred['confianza']:.1%}"
        )

        resultados[ticker] = {"prediccion": pred, "orden": None}

    # ----- PASO 4: EJECUTAR ÓRDENES -----
    logger.info("\n💰 PASO 4: Ejecución de órdenes")
    cuenta = None

    if dry_run:
        logger.info("   DRY RUN — no se ejecutan órdenes")
        for ticker, res in resultados.items():
            pred = res["prediccion"]
            if pred.get("accion") in ("COMPRAR", "VENDER"):
                logger.info(
                    f"   [DRY] {ticker}: {pred['accion']} x{Config.CANTIDAD_DEFAULT}"
                )
    else:
        try:
            iol = IOLClient(sandbox=Config.IOL_SANDBOX)
            cuenta = iol.resumen_cuenta()

            for ticker, res in resultados.items():
                pred = res["prediccion"]

                # Solo operar si la confianza supera el mínimo
                if pred.get("confianza", 0) < Config.CONFIANZA_MINIMA:
                    logger.info(
                        f"   {ticker}: confianza {pred['confianza']:.1%} < "
                        f"{Config.CONFIANZA_MINIMA:.0%} — no opera"
                    )
                    continue

                try:
                    orden = iol.ejecutar_senal(
                        ticker=TICKERS[ticker]["iol"],
                        senal=pred["prediccion"],
                        cantidad=Config.CANTIDAD_DEFAULT,
                    )
                    res["orden"] = orden
                except Exception as e:
                    logger.error(f"   Error operando {ticker}: {e}")
                    res["error"] = str(e)

        except Exception as e:
            logger.error(f"   Error conectando a IOL: {e}")

    # ----- PASO 5: REPORTE -----
    logger.info("\n📧 PASO 5: Reporte")
    reporte = generar_reporte(resultados, sentiment, cuenta)
    print(reporte)

    # Guardar reporte
    output_dir = Path(Config.OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)

    fecha_str = datetime.now().strftime("%Y%m%d_%H%M")
    reporte_path = output_dir / f"reporte_{fecha_str}.txt"
    with open(reporte_path, "w", encoding="utf-8") as f:
        f.write(reporte)
    logger.info(f"   Reporte guardado en {reporte_path}")

    # Guardar resultados JSON
    resultado_path = output_dir / f"resultado_{fecha_str}.json"
    with open(resultado_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": fecha_str,
                "sentiment": sentiment,
                "predicciones": {
                    t: r["prediccion"] for t, r in resultados.items()
                },
                "ordenes": {
                    t: str(r.get("orden")) for t, r in resultados.items()
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    logger.info(f"   Resultados guardados en {resultado_path}")

    # Enviar email
    enviar_email(reporte)

    # Fin
    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(f"\n✅ Pipeline completado en {elapsed:.1f}s")

    return resultados


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PAIOL — Pipeline Diario de Trading")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="No ejecutar órdenes reales (solo predecir y reportar)",
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default=None,
        choices=list(TICKERS.keys()),
        help="Operar solo un ticker específico",
    )
    parser.add_argument(
        "--skip-sentiment",
        action="store_true",
        help="Saltear scraping de noticias y análisis de sentimiento",
    )

    args = parser.parse_args()

    try:
        run_pipeline(
            dry_run=args.dry_run,
            ticker_filter=args.ticker,
            skip_sentiment=args.skip_sentiment,
        )
    except Exception as e:
        logger.critical(f"Pipeline falló: {e}", exc_info=True)
        sys.exit(1)
