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
import time
import logging
from datetime import datetime
from pathlib import Path

import yfinance as yf
import holidays

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("PAIOL-Cloud")


# Caché de último cierre por ticker, para toda la corrida.
# Antes cada paso (1, 1.5, 3) descargaba los mismos precios por separado:
# ~15 llamadas a yfinance en 2 min → Yahoo devolvía vacío (rate-limit) y los
# tickers quedaban "sin cotización". Ahora se descarga una vez y se reusa.
_precio_cache: dict = {}
_feriados_ar = holidays.Argentina()


def precio_cierre(ticker_yf: str, reintentos: int = 3, espera: float = 3.0):
    """Último cierre REAL de yfinance, cacheado por corrida y con reintentos.

    Descarta barras de días sin rueda antes de tomar el cierre:
    - volumen 0 (fines de semana / feriados "limpios")
    - para tickers de BYMA (.BA), fechas que son feriado de Argentina (yfinance
      a veces trae una barra CON volumen del ADR en EEUU en días que BYMA no
      operó, ej. feriados con fines turísticos).

    Devuelve float o None. Solo cachea resultados exitosos.
    """
    if _precio_cache.get(ticker_yf) is not None:
        return _precio_cache[ticker_yf]

    es_byma = ticker_yf.endswith(".BA")
    for intento in range(reintentos):
        try:
            df = yf.download(ticker_yf, period="10d", progress=False)
            if not df.empty:
                # yfinance devuelve columns MultiIndex (('Close','GGAL.BA')):
                # aplanar para que df["Close"] sea una Serie y float() funcione
                if df.columns.nlevels > 1:
                    df.columns = df.columns.get_level_values(0)
                # descartar barras sin rueda real
                if "Volume" in df.columns:
                    df = df[df["Volume"].fillna(0) > 0]
                if es_byma and not df.empty:
                    df = df[[d.date() not in _feriados_ar for d in df.index]]
                if not df.empty:
                    precio = float(df["Close"].iloc[-1])
                    _precio_cache[ticker_yf] = precio
                    return precio
        except Exception as e:
            logger.warning(f"  yfinance {ticker_yf} intento {intento + 1}: {e}")
        if intento < reintentos - 1:
            time.sleep(espera)

    logger.warning(f"  yfinance {ticker_yf}: sin dato tras {reintentos} intentos")
    return None


def main():
    logger.info("=" * 60)
    logger.info("  PAIOL Daily Run — Cloud + Supabase")
    logger.info(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    # No correr en fines de semana ni feriados: BYMA cerrado (incluidos los
    # "feriados con fines turísticos", que también cierran el mercado) → no hay
    # rueda ese día → las predicciones quedarían sin poder evaluarse nunca.
    hoy = datetime.now().date()
    feriados_ar = holidays.Argentina()
    if hoy.weekday() >= 5 or hoy in feriados_ar:
        motivo = "Fin de semana" if hoy.weekday() >= 5 else f"Feriado ({feriados_ar.get(hoy)})"
        logger.info(f"{motivo} — no se opera. Saliendo.")
        return

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
                        precio_salida = precio_cierre(config["yfinance"])
                        if precio_salida is None:
                            continue

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

    # ---- PASO 1.5: EVALUAR ACIERTO DE PREDICCIONES ANTERIORES ----
    logger.info("\n🎯 PASO 1.5: Evaluando aciertos de predicciones anteriores...")
    try:
        if db:
            from features import TICKERS, UMBRAL

            # Precio de cierre actual para cada ticker con predicción pendiente
            precios_actuales = {}
            for ticker, config in TICKERS.items():
                precio = precio_cierre(config["yfinance"])
                if precio is not None:
                    precios_actuales[ticker] = precio

            res = db.evaluar_predicciones_pendientes(precios_actuales, umbral=UMBRAL)
            if res["evaluadas"]:
                logger.info(
                    f"  Evaluadas {res['evaluadas']} predicciones — "
                    f"aciertos {res['aciertos']} ({res['accuracy']:.1%})"
                )
                db.log_pipeline("evaluar_predicciones", "OK",
                                f"{res['aciertos']}/{res['evaluadas']} aciertos")
            else:
                logger.info("  Sin predicciones pendientes de evaluar")
    except Exception as e:
        logger.error(f"  Error evaluando predicciones: {e}")
        if db:
            db.log_pipeline("evaluar_predicciones", "ERROR", str(e))

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

                # Obtener cotizaciones actuales (precio de referencia de la predicción)
                from features import TICKERS
                from position_sizing import PositionManager

                cotizaciones = {}
                for ticker in predicciones:
                    config = TICKERS.get(ticker)
                    if config:
                        precio = precio_cierre(config["yfinance"])
                        if precio is not None:
                            cotizaciones[ticker] = {"ultimoPrecio": precio}

                # Guardar predicciones con su precio de referencia
                db.guardar_predicciones_batch(predicciones, cotizaciones=cotizaciones)
                logger.info("  ✅ Predicciones guardadas")

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
            # Accuracy del modelo (predicción de clase, incluye HOLD/LATERAL)
            acc = db.get_accuracy_modelo()
            if "mensaje" not in acc:
                logger.info(f"  Accuracy modelo: {acc['accuracy']:.1%} "
                            f"({acc['aciertos']}/{acc['total_evaluadas']})")
            else:
                logger.info(f"  {acc['mensaje']}")

            # Performance del paper trading (rentabilidad de las operaciones)
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
