"""
¿PAIOL le gana a comprar y mantener?

Compara, TRADE POR TRADE, el retorno que obtuvo PAIOL (su elección) contra el de
tener la canasta equiponderada de todas las acciones que opera, en la MISMA
ventana de cada trade. Así aísla si el modelo ELIGE bien, sin que lo ensucien:
  - el split de YPFD (se usan cierres ajustados, auto_adjust=True)
  - el hueco de ~2 meses que PAIOL estuvo caído (sólo compara ventanas de trades reales)

Interpretación: si el "alpha" (PAIOL − canasta) es ~0 o negativo, el modelo no
está aportando sobre comprar-y-mantener; si es claramente positivo Y se sostiene
en muchas ventanas (incluyendo mercados a la baja), ahí sí agrega valor.

Corre:  venv/bin/python paiol_vs_hold.py
Sólo lee (Supabase + yfinance). No modifica nada.
"""

from dotenv import load_dotenv
load_dotenv(override=True)

import requests
import numpy as np
import pandas as pd
import yfinance as yf

from db import PaiolDB
from features import TICKERS


def cierres_ajustados(tickers_yf, start, end):
    """Panel de cierres ajustados (maneja splits) indexado por fecha 'YYYY-MM-DD'."""
    px = {}
    for tk, y in tickers_yf.items():
        df = yf.download(y, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty:
            continue
        c = df["Close"]
        c = c.iloc[:, 0] if hasattr(c, "columns") else c
        c.index = [str(d)[:10] for d in c.index]
        px[tk] = c
    return px


def ret_ventana(serie, d0, d1):
    """Retorno de una serie entre dos fechas (alinea a la rueda más cercana)."""
    i0 = min(serie.index.searchsorted(d0), len(serie) - 1)
    i1 = min(serie.index.searchsorted(d1), len(serie) - 1)
    return float(serie.iloc[i1] / serie.iloc[i0] - 1)


def main():
    db = PaiolDB()
    ops = requests.get(
        f"{db.url}/rest/v1/operaciones",
        headers=db.headers,
        params={"cerrada": "eq.true",
                "select": "ticker,fecha,fecha_cierre,pnl_pct",
                "order": "fecha.asc"},
    ).json()
    if not ops:
        print("No hay operaciones cerradas todavía.")
        return

    # Universo = tickers que PAIOL efectivamente operó (mapeados a yfinance)
    universo = sorted({o["ticker"] for o in ops})
    tickers_yf = {t: TICKERS[t]["yfinance"] for t in universo if t in TICKERS}

    start = min(o["fecha"] for o in ops)
    end = max(o["fecha_cierre"] for o in ops)
    # buffer para tener la rueda siguiente disponible
    end = (pd.to_datetime(end) + pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    px = cierres_ajustados(tickers_yf, start, end)

    filas = []
    for o in ops:
        rets = [ret_ventana(px[t], o["fecha"], o["fecha_cierre"]) for t in px]
        filas.append({
            "fecha": o["fecha"],
            "mes": o["fecha"][:7],
            "ticker": o["ticker"],
            "paiol": o["pnl_pct"],
            "canasta": float(np.mean(rets)) if rets else np.nan,
        })
    df = pd.DataFrame(filas)
    df["alpha"] = df["paiol"] - df["canasta"]

    def linea(sub, nombre):
        gano = (sub["alpha"] > 0).mean()
        print(f"  {nombre:14s} n={len(sub):2d} | PAIOL {sub['paiol'].mean():+.2%} | "
              f"canasta {sub['canasta'].mean():+.2%} | alpha {sub['alpha'].mean():+.2%} | "
              f"le gana a la canasta {gano:.0%}")

    print(f"PAIOL vs COMPRAR Y MANTENER   (canasta = {', '.join(universo)})")
    print(f"Ventanas: {len(df)} trades cerrados, {start} → {max(o['fecha_cierre'] for o in ops)}")
    print("=" * 78)
    linea(df, "TOTAL")
    print("  " + "-" * 72)
    for mes in sorted(df["mes"].unique()):
        linea(df[df["mes"] == mes], mes)

    print()
    if len(df) < 30:
        print(f"⚠️  Son {len(df)} trades: muestra chica. Un alpha positivo acá todavía")
        print("    puede ser suerte/una buena racha. Hace falta más historia —")
        print("    sobre todo meses a la baja— para concluir que el modelo aporta.")


if __name__ == "__main__":
    main()
