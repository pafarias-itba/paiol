"""
PAIOL — Backfill de accuracy para predicciones viejas

Las predicciones guardadas antes de agregar la columna `precio_pred` quedaron
sin precio de referencia, así que el PASO 1.5 del job las saltea. Este script
las recupera de una sola vez usando el histórico de yfinance.

El job corre 9:30 ART (pre-mercado), así que una predicción con `fecha=D` se
calculó con la barra de D-1 y predice el movimiento D-1 → D:

  - precio_pred     = cierre del día hábil ANTERIOR a `fecha` (barra del modelo)
  - precio_actual   = cierre EN `fecha`
  - retorno_real    = precio_actual / precio_pred - 1
  - senal_real      = clase realizada con el mismo UMBRAL del modelo
  - acierto         = senal == senal_real

Re-ejecutable: toma filas con `acierto` NULL. Si `fecha` todavía no cerró
(es hoy), guarda solo `precio_pred` y la deja parcial para más adelante.

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

    El job corre 9:30 ART (pre-mercado), así que una predicción con `fecha=D`
    se calculó con la barra de D-1 y predice el movimiento D-1 → D. Por eso:

    precio_pred   = cierre del día hábil ANTERIOR a fecha (la barra que usó el
                    modelo)
    precio_actual = cierre EN fecha, SOLO si fecha ya cerró (es anterior a hoy);
                    si fecha es hoy/futuro devuelve None y la fila queda parcial.
    Cualquiera de los dos puede ser None si no hay dato.
    """
    f = pd.Timestamp(fecha).normalize()
    hoy = pd.Timestamp.today().normalize()

    # Referencia = último cierre estrictamente anterior a fecha (D-1)
    antes = hist.loc[hist.index.normalize() < f]
    if antes.empty:
        return None, None
    precio_pred = float(antes["Close"].iloc[-1])

    # Target = cierre en fecha (D), solo si ese día ya cerró
    en_fecha = hist.loc[hist.index.normalize() == f]
    if not en_fecha.empty and f < hoy:
        precio_actual = float(en_fecha["Close"].iloc[-1])
    else:
        precio_actual = None

    return precio_pred, precio_actual


def main():
    parser = argparse.ArgumentParser(description="Backfill de accuracy de predicciones")
    parser.add_argument("--dry-run", action="store_true",
                        help="Mostrar qué haría sin escribir en la DB")
    parser.add_argument("--reset", action="store_true",
                        help="Limpiar (NULL) acierto/senal_real/retorno_real de las "
                             "filas ya evaluadas, para recomputarlas contra cierres finales")
    args = parser.parse_args()

    load_dotenv(override=True)
    db = PaiolDB()

    if args.reset:
        ok = db._patch("predicciones", {"acierto": "not.is.null"},
                       {"acierto": None, "senal_real": None, "retorno_real": None})
        logger.info("Reset de filas evaluadas: " + ("OK" if ok else "FALLÓ"))
        return

    # Re-ejecutable: toma todo lo que aún no tiene acierto resuelto
    pendientes = db._get("predicciones", {
        "acierto": "is.null",
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
