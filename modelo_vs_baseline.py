"""
¿El modelo de PAIOL le gana al azar / a un baseline trivial?

Compara la accuracy del modelo (sobre predicciones ya evaluadas en Supabase)
contra varios baselines tontos, y le pone un intervalo de confianza del 95%
para no sacar conclusiones apuradas con pocas muestras.

Corre:  venv/bin/python modelo_vs_baseline.py
Sólo lee Supabase. No modifica nada.
"""

from dotenv import load_dotenv
load_dotenv(override=True)

import math
from collections import Counter
from db import PaiolDB

NOM = {0: "BAJA", 1: "LATERAL", 2: "SUBE"}


def wilson(k, n, z=1.96):
    """Intervalo de confianza de Wilson para una proporción (robusto con n chico)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centro = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0, centro - half), min(1, centro + half))


db = PaiolDB()
preds = db._get("predicciones", {
    "acierto": "not.is.null",
    "select": "fecha,senal,senal_real,acierto",
    "order": "fecha.asc",
})

n = len(preds)
if n == 0:
    print("Todavía no hay predicciones evaluadas.")
    raise SystemExit

aciertos = sum(p["acierto"] for p in preds)
acc = aciertos / n
lo, hi = wilson(aciertos, n)

real = Counter(p["senal_real"] for p in preds)
pred = Counter(p["senal"] for p in preds)

# --- baselines ---
clase_mayor, freq_mayor = real.most_common(1)[0]
base_mayor = freq_mayor / n                       # siempre la clase real más común
base_sube = real.get(2, 0) / n                    # siempre SUBE
base_azar3 = 1 / 3                                # tirar una moneda de 3 caras
base_pond = sum((c / n) ** 2 for c in real.values())  # adivinar según frecuencias

print(f"MODELO vs BASELINE   (n={n}, del {preds[0]['fecha']} al {preds[-1]['fecha']})")
print("=" * 56)
print(f"\nAccuracy del modelo:   {acc:5.1%}   (IC95%: {lo:.1%} – {hi:.1%})")
print("\nBaselines (lo que hay que superar):")
rows = [
    (f"siempre {NOM[clase_mayor]}", base_mayor, "el más duro"),
    ("siempre SUBE", base_sube, ""),
    ("azar ponderado", base_pond, "según frecuencias"),
    ("azar puro (3 clases)", base_azar3, ""),
]
for nombre, val, nota in sorted(rows, key=lambda r: -r[1]):
    marca = "  ← " + nota if nota else ""
    print(f"   {nombre:22s} {val:5.1%}{marca}")

# --- veredicto honesto según el IC ---
duro = base_mayor
print("\nVeredicto:")
if lo > duro:
    print(f"   ✅ El modelo SUPERA el baseline duro ({duro:.1%}) con significancia")
    print(f"      (hasta el piso del IC queda por encima).")
elif hi < duro:
    print(f"   ⚠️  El modelo está POR DEBAJO del baseline duro ({duro:.1%}),")
    print(f"      incluso en el techo del IC. Revisar.")
else:
    print(f"   🤔 Empate técnico: el IC del modelo ({lo:.1%}–{hi:.1%}) incluye")
    print(f"      el baseline duro ({duro:.1%}). Con n={n} todavía no se puede")
    print(f"      afirmar que le gana. Hace falta más data.")

# --- señales accionables (lo que importa para operar) ---
accionables = [p for p in preds if p["senal"] in (0, 2)]  # SUBE o BAJA, no LATERAL
if accionables:
    m = len(accionables)
    ok = sum(p["acierto"] for p in accionables)
    lo2, hi2 = wilson(ok, m)
    print(f"\nSeñales accionables (el modelo dijo SUBE o BAJA): {m} de {n}")
    print(f"   Aciertos: {ok/m:5.1%}   (IC95%: {lo2:.1%} – {hi2:.1%})")
    print(f"   [Un 'siempre LATERAL' nunca opera, así que acá es donde el")
    print(f"    modelo tiene que demostrar que sabe algo.]")

print()
