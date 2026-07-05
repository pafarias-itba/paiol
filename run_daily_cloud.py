"""
PAIOL — Runner Diario para Cloud (Railway/Render)

Versión cloud del script diario. Corre como cron job:
1. Corre el pipeline (sentiment + features + predicción)
2. Persiste todo en Supabase (predicciones, sentiment, operaciones)
3. Cierra posiciones del día anterior con precio real
4. Re-entrenamiento mensual (día 1)

En Railway se programa como cron job a las 9:30 ART (12:30 UTC).
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path

import yfinance as yf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("PAIOL-Cloud")


def main():
    logger.info("=" * 60)
    logger.info("  PAIOL Daily Run — Cloud + Supabase")
    logger.info(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    # Verificar día de semana (no correr sábado/domingo)
    # dia = datetime.now().weekday()
    # if dia >= 5:
    #     logger.info("Fin de semana — no se opera. Saliendo.")
    #     return

    # Conectar a Supabase
    try:
        from db import PaiolDB
        db = PaiolDB()
        db.log_pipeline("inicio", "OK", "Pipeline iniciado")
        logger.info("✅ Conectado a Supabase")
    except Exception as e:
        logger.error(f"❌ No se pudo conectar a Supabase: {e}")
        logger.info("Continuando sin persistencia...")
        db = None

    # ---- PASO 1: CERRAR POSICIONES DEL DÍA ANTERIOR ----
    logger.info("\n📊 PASO 1: Cerrando posiciones del día anterior...")
    try:
        if db:
            from features import TICKERS
            abiertas = db.get_operaciones_abiertas()

            if abiertas:
                for op in abiertas:
                    ticker = op["ticker"]
                    config = TICKERS.get(ticker)
                    if not config:
                        continue

                    try:
                        df = yf.download(config["yfinance"], period="2d", progress=False)
                        if df.empty or len(df) < 1:
                            continue

                        precio_salida = float(df["Close"].iloc[-1])
                        precio_entrada = op["precio_entrada"]
                        accion = op["accion"]

                        if accion == "COMPRAR":
                            pnl = (precio_salida - precio_entrada) * op["cantidad"]
                            pnl_pct = (precio_salida / precio_entrada) - 1
                            correcta = precio_salida > precio_entrada
                        else:
                            pnl = (precio_entrada - precio_salida) * op["cantidad"]
                            pnl_pct = (precio_entrada / precio_salida) - 1
                            correcta = precio_salida < precio_entrada

                        db.cerrar_operacion(op["id"], precio_salida, pnl, pnl_pct, correcta)

                        emoji = "✅" if pnl > 0 else "❌"
                        logger.info(f"  {emoji} {ticker}: ${precio_entrada:,.0f} → ${precio_salida:,.0f} P&L: ${pnl:+,.0f}")

                    except Exception as e:
                        logger.error(f"  Error cerrando {ticker}: {e}")

                db.log_pipeline("cerrar_posiciones", "OK", f"Cerradas {len(abiertas)} posiciones")
            else:
                logger.info("  Sin posiciones abiertas")
    except Exception as e:
        logger.error(f"  Error cerrando posiciones: {e}")
        if db:
            db.log_pipeline("cerrar_posiciones", "ERROR", str(e))

    # ---- PASO 2: CORRER PIPELINE ----
    logger.info("\n🚀 PASO 2: Corriendo pipeline...")
    try:
        from pipeline import run_pipeline
        resultados = run_pipeline(dry_run=True)
        if db:
            db.log_pipeline("pipeline", "OK", "Pipeline completado")
    except Exception as e:
        logger.error(f"  Error en pipeline: {e}")
        if db:
            db.log_pipeline("pipeline", "ERROR", str(e))
        return

    # ---- PASO 3: PERSISTIR EN SUPABASE ----
    logger.info("\n💾 PASO 3: Guardando en Supabase...")
    try:
        if db:
            # Buscar último resultado
            outputs_dir = Path("outputs")
            resultados_files = sorted(outputs_dir.glob("resultado_*.json"), reverse=True)

            if resultados_files:
                with open(resultados_files[0], "r") as f:
                    resultado = json.load(f)

                predicciones = resultado.get("predicciones", {})
                sentiment = resultado.get("sentiment", {})

                # Guardar sentiment
                db.guardar_sentiment(sentiment)
                logger.info("  ✅ Sentiment guardado")

                # Guardar predicciones
                db.guardar_predicciones_batch(predicciones)
                logger.info("  ✅ Predicciones guardadas")

                # Obtener cotizaciones y guardar operaciones
                from features import TICKERS
                from position_sizing import PositionManager

                cotizaciones = {}
                for ticker in predicciones:
                    config = TICKERS.get(ticker)
                    if config:
                        try:
                            df = yf.download(config["yfinance"], period="1d", progress=False)
                            if not df.empty:
                                cotizaciones[ticker] = {
                                    "ultimoPrecio": float(df["Close"].iloc[-1])
                                }
                        except:
                            pass

                # Position sizing
                pm = PositionManager(capital_total=100_000, riesgo_pct=0.02)
                ordenes = pm.calcular_todas(predicciones, cotizaciones)

                # Guardar operaciones abiertas
                for ticker, orden in ordenes.items():
                    if orden.accion in ("COMPRAR", "VENDER") and orden.cantidad > 0:
                        precio = cotizaciones.get(ticker, {}).get("ultimoPrecio", 0)
                        db.guardar_operacion(
                            ticker=ticker,
                            accion=orden.accion,
                            cantidad=orden.cantidad,
                            precio_entrada=precio,
                            confianza=predicciones[ticker].get("confianza", 0),
                        )
                        logger.info(f"  📝 {ticker}: {orden.accion} x{orden.cantidad} @ ${precio:,.0f}")

                db.log_pipeline("persistencia", "OK", "Todo guardado en Supabase")
            else:
                logger.warning("  No se encontró archivo de resultado")
    except Exception as e:
        logger.error(f"  Error persistiendo: {e}")
        if db:
            db.log_pipeline("persistencia", "ERROR", str(e))

    # ---- PASO 4: REPORTE ----
    logger.info("\n📈 PASO 4: Reporte")
    try:
        if db:
            metricas = db.get_metricas_paper_trading()
            if "mensaje" not in metricas:
                logger.info(f"  Operaciones: {metricas['total_operaciones']}")
                logger.info(f"  Win rate: {metricas['win_rate']:.1%}")
                logger.info(f"  P&L total: ${metricas['pnl_total']:,.0f}")
                logger.info(f"  Profit factor: {metricas['profit_factor']:.2f}x")
            else:
                logger.info(f"  {metricas['mensaje']}")
    except Exception as e:
        logger.error(f"  Error en reporte: {e}")

    # ---- PASO 5: RE-ENTRENAMIENTO MENSUAL ----
    if datetime.now().day == 1:
        logger.info("\n🏋️ PASO 5: Re-entrenamiento mensual")
        try:
            from retrain import AutoRetrainer
            ar = AutoRetrainer()
            ar.run()
            if db:
                db.log_pipeline("retrain", "OK", "Re-entrenamiento completado")
        except Exception as e:
            logger.error(f"  Error en re-entrenamiento: {e}")
            if db:
                db.log_pipeline("retrain", "ERROR", str(e))

    logger.info("\n✅ PAIOL Daily Run completado")
    if db:
        db.log_pipeline("fin", "OK", "Pipeline completado exitosamente")


if __name__ == "__main__":
    main()
