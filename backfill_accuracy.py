"""
PAIOL — Backfill de accuracy para predicciones viejas

Las predicciones guardadas antes de agregar la columna `precio_pred` quedaron
sin precio de referencia, así que el PASO 1.5 del job las saltea. Este script
las recupera de una sola vez usando el histórico de yfinance:

  - precio_pred     = cierre del día de la predicción (`fecha`)
  - precio_actual   = cierre del día hábil siguiente
  - retorno_real    = precio_actual / precio_pred - 1
  - senal_real      = clase realizada con el mismo UMBRAL del modelo
  - acierto         = senal == senal_real

Es idempotente: solo toca filas con `precio_pred` NULL. Si el día siguiente
todavía no cerró, guarda solo `precio_pred` y deja que el PASO 1.5 del job
termine de evaluarla mañana.

Uso:
    python3 backfill_accuracy.py           # aplica los cambios
    python3 backfill_accuracy.py --dry-run # muestra qué haría, sin escribir
"""

import argparse
import logging

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

from db import PaiolDB
from features import TICKERS, UMBRAL

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("PAIOL-Backfill")


def clasificar(retorno: float, umbral: float) -> int:
    """0=Baja, 1=Lateral, 2=Sube (mismo criterio que features.generar_labels)."""
    if retorno < -umbral:
        return 0
    if retorno > umbral:
        return 2
    return 1


def descargar_historico(symbol: str, fecha_min: str) -> pd.DataFrame:
    """Baja el histórico diario desde antes de fecha_min hasta hoy."""
    start = (pd.Timestamp(fecha_min) - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    end = (pd.Timestamp.today() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    df = yf.download(symbol, start=start, end=end, progress=False)
    if df.empty:
        return df
    # yfinance a veces devuelve columnas MultiIndex para un solo ticker
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def cierres_para_fecha(hist: pd.DataFrame, fecha: str):
    """Devuelve (precio_pred, precio_actual) para una predicción en `fecha`.

    precio_pred   = último cierre <= fecha
    precio_actual = primer cierre estrictamente posterior a esa fecha de cierre
    Cualquiera de los dos puede ser None si no hay dato.
    """
    f = pd.Timestamp(fecha)

    hasta_fecha = hist.loc[:f]
    if hasta_fecha.empty:
        return None, None
    precio_pred = float(hasta_fecha["Close"].iloc[-1])
    fecha_cierre = hasta_fecha.index[-1]

    posteriores = hist.loc[hist.index > fecha_cierre]
    precio_actual = float(posteriores["Close"].iloc[0]) if not posteriores.empty else None

    return precio_pred, precio_actual


def main():
    parser = argparse.ArgumentParser(description="Backfill de accuracy de predicciones")
    parser.add_argument("--dry-run", action="store_true",
                        help="Mostrar qué haría sin escribir en la DB")
    args = parser.parse_args()

    load_dotenv()
    db = PaiolDB()

    pendientes = db._get("predicciones", {
        "precio_pred": "is.null",
        "select": "id,fecha,ticker,senal",
        "order": "fecha.asc",
    })

    if not pendientes:
        logger.info("No hay predicciones sin precio_pred. Nada que hacer.")
        return

    logger.info(f"Predicciones a recuperar: {len(pendientes)}")

    # Agrupar por ticker para bajar el histórico una sola vez por símbolo
    por_ticker = {}
    for p in pendientes:
        por_ticker.setdefault(p["ticker"], []).append(p)

    total_eval = total_parcial = total_skip = aciertos = 0

    for ticker, preds in por_ticker.items():
        config = TICKERS.get(ticker)
        if not config:
            logger.warning(f"  {ticker}: no está en TICKERS, se saltea ({len(preds)})")
            total_skip += len(preds)
            continue

        fecha_min = min(p["fecha"] for p in preds)
        hist = descargar_historico(config["yfinance"], fecha_min)
        if hist.empty:
            logger.warning(f"  {ticker}: sin histórico de yfinance, se saltea")
            total_skip += len(preds)
            continue

        for p in preds:
            precio_pred, precio_actual = cierres_para_fecha(hist, p["fecha"])
            if precio_pred is None:
                logger.warning(f"  {ticker} {p['fecha']}: sin cierre de referencia, skip")
                total_skip += 1
                continue

            # Caso 1: todavía no cerró el día siguiente → solo guardar precio_pred
            if precio_actual is None:
                if args.dry_run:
                    logger.info(f"  [DRY] {ticker} {p['fecha']}: precio_pred={precio_pred:.2f} (sin evaluar aún)")
                else:
                    db._patch("predicciones", {"id": f"eq.{p['id']}"},
                              {"precio_pred": round(precio_pred, 4)})
                total_parcial += 1
                continue

            # Caso 2: evaluación completa
            retorno = precio_actual / precio_pred - 1
            senal_real = clasificar(retorno, UMBRAL)
            acierto = (p["senal"] == senal_real)

            if args.dry_run:
                marca = "✅" if acierto else "❌"
                logger.info(
                    f"  [DRY] {marca} {ticker} {p['fecha']}: "
                    f"{precio_pred:.2f}→{precio_actual:.2f} ret={retorno:+.2%} "
                    f"senal={p['senal']} real={senal_real}"
                )
            else:
                db._patch("predicciones", {"id": f"eq.{p['id']}"}, {
                    "precio_pred": round(precio_pred, 4),
                    "retorno_real": round(retorno, 6),
                    "senal_real": senal_real,
                    "acierto": acierto,
                })
            total_eval += 1
            aciertos += int(acierto)

    logger.info("-" * 50)
    accuracy = aciertos / total_eval if total_eval else 0.0
    logger.info(f"Evaluadas por completo: {total_eval}  (aciertos {aciertos} = {accuracy:.1%})")
    logger.info(f"Solo precio_pred (falta día siguiente): {total_parcial}")
    logger.info(f"Salteadas: {total_skip}")
    if args.dry_run:
        logger.info("DRY RUN — no se escribió nada en la DB.")


if __name__ == "__main__":
    main()
