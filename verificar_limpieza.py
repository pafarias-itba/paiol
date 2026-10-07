"""
Verificación de la limpieza del hueco de agosto 2026 (reavivar PAIOL).

Chequea contra yfinance de forma independiente que las operaciones y
predicciones del 2026-08-04 quedaron cerradas/evaluadas con los precios
correctos:
  - entrada de la operación  = cierre del 2026-08-03 (día D-1)
  - salida  de la operación  = cierre del 2026-08-04 (día D)
  - predicciones evaluadas contra el cierre del 2026-08-04

Corre:  venv/bin/python verificar_limpieza.py
Sólo lee (yfinance + Supabase). No modifica nada.
"""

from dotenv import load_dotenv
load_dotenv(override=True)

import yfinance as yf
from db import PaiolDB

FECHA = "2026-08-04"
D_1 = "2026-08-03"   # cierre que debería ser la ENTRADA
D = "2026-08-04"     # cierre que debería ser la SALIDA / referencia de evaluación
UMBRAL = 0.01
TICKERS = {"GGAL": "GGAL.BA", "YPFD": "YPFD.BA", "MELI": "MELI.BA",
           "PAMP": "PAMP.BA", "SPY": "SPY"}

ok_all = True


def check(cond, msg):
    global ok_all
    print(("  ✓ " if cond else "  ✗ ") + msg)
    if not cond:
        ok_all = False


def cierre_de(ticker_yf, fecha):
    """Cierre real de yfinance para una fecha (None si no hubo rueda)."""
    df = yf.download(ticker_yf, start="2026-07-29", end="2026-08-07",
                     progress=False, auto_adjust=False)
    if df.empty:
        return None
    c = df["Close"]
    if hasattr(c, "columns"):
        c = c.iloc[:, 0]
    for idx, val in c.items():
        if str(idx)[:10] == fecha:
            return round(float(val), 2)
    return None


def aprox(a, b, tol=0.5):
    return a is not None and b is not None and abs(a - b) <= tol


db = PaiolDB()

# --- precios reales de yfinance ---
print(f"Precios yfinance (referencia independiente):")
cierres = {}
for tk, yf_sym in TICKERS.items():
    c3 = cierre_de(yf_sym, D_1)
    c4 = cierre_de(yf_sym, D)
    cierres[tk] = (c3, c4)
    print(f"  {tk:5s}  3/8={c3}  4/8={c4}")

# --- 1. operaciones del 4/8 ---
print(f"\n1) OPERACIONES del {FECHA} (entrada=cierre 3/8, salida=cierre 4/8):")
ops = db._get("operaciones", {"fecha": f"eq.{FECHA}", "select": "*", "order": "ticker"})
for op in ops:
    tk = op["ticker"]
    c3, c4 = cierres.get(tk, (None, None))
    ent, sal, cant, pnl = op["precio_entrada"], op["precio_salida"], op["cantidad"], op["pnl"]
    pnl_esp = round((sal - ent) * cant, 2)
    print(f"  · {tk}: entrada={ent} salida={sal} cant={cant} pnl={pnl:+,.0f}")
    check(aprox(ent, c3), f"{tk} entrada {ent} == cierre 3/8 {c3}")
    check(aprox(sal, c4), f"{tk} salida  {sal} == cierre 4/8 {c4}")
    check(abs(pnl - pnl_esp) < 1, f"{tk} pnl {pnl:+,.0f} == (salida-entrada)*cant {pnl_esp:+,.0f}")
    check(op["cerrada"] is True, f"{tk} marcada como cerrada")

# --- 2. predicciones del 4/8 ---
print(f"\n2) PREDICCIONES del {FECHA} (evaluadas contra cierre 4/8, umbral {UMBRAL}):")
preds = db._get("predicciones", {"fecha": f"eq.{FECHA}", "select": "*", "order": "ticker"})
for p in preds:
    tk = p["ticker"]
    _, c4 = cierres.get(tk, (None, None))
    pp = p["precio_pred"]
    ret = c4 / pp - 1 if (pp and c4) else None
    sr_esp = 0 if ret < -UMBRAL else (2 if ret > UMBRAL else 1)
    nom = {0: "BAJA", 1: "LATERAL", 2: "SUBE"}
    print(f"  · {tk}: senal={nom[p['senal']]:7s} senal_real={nom[p['senal_real']]:7s} "
          f"ret={ret:+.2%} acierto={p['acierto']}")
    check(p["senal_real"] == sr_esp, f"{tk} senal_real {nom[p['senal_real']]} == esperado {nom[sr_esp]}")
    check(p["acierto"] == (p["senal"] == sr_esp), f"{tk} acierto coherente")

# --- 3. nada colgado ---
print("\n3) Nada pendiente:")
abiertas = db._get("operaciones", {"cerrada": "eq.false", "select": "id"})
pendientes = db._get("predicciones", {"acierto": "is.null", "select": "id"})
check(len(abiertas) == 0, f"operaciones abiertas = {len(abiertas)} (esperado 0)")
check(len(pendientes) == 0, f"predicciones sin evaluar = {len(pendientes)} (esperado 0)")

# --- 4. totales ---
print("\n4) Totales de la cuenta:")
cerradas = db._get("operaciones", {"cerrada": "eq.true", "select": "pnl"})
pnl_total = sum(o["pnl"] or 0 for o in cerradas)
gan = sum(1 for o in cerradas if (o["pnl"] or 0) > 0)
ev = db._get("predicciones", {"acierto": "not.is.null", "select": "acierto"})
acc = sum(1 for x in ev if x["acierto"]) / len(ev) if ev else 0
print(f"  P&L total: {pnl_total:+,.0f} ARS | {len(cerradas)} cerradas | "
      f"{gan} ganadoras ({gan/len(cerradas):.1%})")
print(f"  Accuracy modelo: {acc:.1%} ({len(ev)} evaluadas)")

print("\n" + ("✅ TODO OK" if ok_all else "❌ HAY DISCREPANCIAS — revisar arriba"))
