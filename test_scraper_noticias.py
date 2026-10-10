"""
Tests del recolector de noticias.

Corre sin dependencias:  venv/bin/python test_scraper_noticias.py
(también compatible con pytest si lo instalás)
"""

from scraper_sentiment import Noticia
from scraper_noticias import construir_fila, detectar_empresas, es_cotizacion


def _fila(titulo, resumen="", link="http://x/n", fuente="ambito_negocios", fecha=""):
    n = Noticia(titulo=titulo, resumen=resumen, fuente=fuente, fecha=fecha, link=link)
    return construir_fila(n, sentimiento=0.1, fecha_scrape="2026-10-10")


# ---- Parte 1: empresas se envía como LISTA, no string (bug doble-codificación) ----

def test_empresas_es_lista_no_string():
    fila = _fila("YPF anuncia inversión en Vaca Muerta")
    assert isinstance(fila["empresas"], list), \
        f"empresas debe ser list, es {type(fila['empresas']).__name__}"
    assert fila["empresas"] == ["YPFD"]


def test_empresas_vacia_es_lista_vacia():
    fila = _fila("La inflación de la zona euro se aceleró en agosto")
    assert fila["empresas"] == []          # lista vacía
    assert not isinstance(fila["empresas"], str)   # NUNCA el string "[]"


def test_empresas_multiples():
    fila = _fila("Galicia y Pampa Energía lideran la suba del Merval")
    assert set(fila["empresas"]) == {"GGAL", "PAMP"}


def test_detectar_empresas_sin_falso_positivo():
    # "argentina" no debe matchear ninguna empresa
    assert detectar_empresas("La selección argentina ganó en Brasil") == []


# ---- Parte 3: es_cotizacion con ejemplos reales del corpus ----

COTIZACIONES = [
    "Dólar hoy: a cuánto cotiza este viernes 2 de octubre",
    "Dólar blue hoy: a cuánto cerró este lunes 5 de octubre",
    "Euro hoy y Euro blue hoy: a cuánto cotizó este lunes",
    "Real blue: a cuánto operó este lunes 5 de octubre",
    "Dólar hoy y dólar blue hoy minuto a minuto: a cuánto opera el billete",
]

NO_COTIZACIONES = [
    "Efecto Brasil: las acciones argentinas treparon hasta 10%",
    "Cuánto se gana con plazo fijo Banco Galicia si se invierten $100.000",
    "YPF anuncia nueva inversión en Vaca Muerta",
    "Flybondi se presentó en concurso preventivo",
]


def test_es_cotizacion_detecta_precios():
    for t in COTIZACIONES:
        assert es_cotizacion(t), f"debería ser cotización: {t!r}"


def test_es_cotizacion_no_marca_noticias_reales():
    for t in NO_COTIZACIONES:
        assert not es_cotizacion(t), f"NO debería ser cotización: {t!r}"


def test_es_cotizacion_por_url():
    assert es_cotizacion("Nota cualquiera", link="https://ambito.com/cotizacion/dolar")


def test_fila_marca_es_cotizacion():
    fila = _fila("Dólar hoy: a cuánto cotiza este viernes")
    assert fila["es_cotizacion"] is True
    fila2 = _fila("YPF descubre un nuevo yacimiento")
    assert fila2["es_cotizacion"] is False


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    fallos = 0
    for t in tests:
        try:
            t()
            print(f"  ✓ {t.__name__}")
        except AssertionError as e:
            fallos += 1
            print(f"  ✗ {t.__name__}: {e}")
    print(f"\n{'❌' if fallos else '✅'} {len(tests)-fallos}/{len(tests)} tests OK")
    raise SystemExit(1 if fallos else 0)
