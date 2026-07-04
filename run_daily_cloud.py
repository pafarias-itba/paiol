"""
PAIOL — Runner Diario para Cloud (Railway/Render)

Versión cloud del script diario. Corre como cron job:
1. Cierra posiciones del día anterior
2. Corre el pipeline
3. Registra en paper trading
4. Guarda logs

En Railway se programa como cron job a las 9:30 ART (12:30 UTC).
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("PAIOL-Cloud")


def main():
    logger.info("=" * 60)
    logger.info("  PAIOL Daily Run — Cloud")
    logger.info(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    # Verificar día de semana (no correr sábado/domingo)
    dia = datetime.now().weekday()
    if dia >= 5:
        logger.info("Fin de semana — no se opera. Saliendo.")
        return

    # PASO 1: Cerrar posiciones del día anterior
    logger.info("\n📊 PASO 1: Cerrando posiciones del día anterior...")
    try:
        from paper_trading import PaperTrader
        pt = PaperTrader(capital_inicial=100_000)
        pt.cerrar_posiciones_abiertas()
    except Exception as e:
        logger.error(f"Error cerrando posiciones: {e}")

    # PASO 2: Correr pipeline
    logger.info("\n🚀 PASO 2: Corriendo pipeline...")
    try:
        from pipeline import run_pipeline
        resultados = run_pipeline(dry_run=True)
    except Exception as e:
        logger.error(f"Error en pipeline: {e}")
        return

    # PASO 3: Registrar en paper trading
    logger.info("\n📝 PASO 3: Registrando en paper trading...")
    try:
        # Buscar último resultado
        outputs_dir = Path("outputs")
        resultados_files = sorted(outputs_dir.glob("resultado_*.json"), reverse=True)

        if resultados_files:
            ultimo = resultados_files[0]
            with open(ultimo, "r") as f:
                resultado = json.load(f)

            predicciones = resultado.get("predicciones", {})

            # Obtener cotizaciones
            import yfinance as yf
            from features import TICKERS
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
            from position_sizing import PositionManager
            pm = PositionManager(capital_total=100_000, riesgo_pct=0.02)
            ordenes = pm.calcular_todas(predicciones, cotizaciones)

            ordenes_dict = {}
            for ticker, orden in ordenes.items():
                ordenes_dict[ticker] = {
                    "accion": orden.accion,
                    "cantidad": orden.cantidad,
                }

            pt.registrar_dia(predicciones, ordenes_dict, cotizaciones)
        else:
            logger.warning("No se encontró archivo de resultado")
    except Exception as e:
        logger.error(f"Error registrando paper trading: {e}")

    # PASO 4: Reporte
    logger.info("\n📈 PASO 4: Reporte")
    try:
        print(pt.reporte())
    except Exception as e:
        logger.error(f"Error generando reporte: {e}")

    logger.info("\n✅ PAIOL Daily Run completado")
    
    # PASO 5: Re-entrenamiento mensual (día 1 de cada mes)
    if datetime.now().day == 1:
        logger.info("\n🏋️ PASO 5: Re-entrenamiento mensual")
        try:
            from retrain import AutoRetrainer
            ar = AutoRetrainer()
            ar.run()
        except Exception as e:
            logger.error(f"Error en re-entrenamiento: {e}")
    else:
        logger.info(f"\n🏋️ PASO 5: Re-entrenamiento — próximo el día 1 del mes")


if __name__ == "__main__":
    main()
