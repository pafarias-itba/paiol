"""
Tests del recolector de noticias.

Corre sin dependencias:  venv/bin/python test_scraper_noticias.py
(también compatible con pytest si lo instalás)
"""

from scraper_sentiment import Noticia
from scraper_noticias import (
    construir_fila, detectar_empresas, es_cotizacion, es_patrocinado,
    medio_y_seccion,
)

TODAS = frozenset({"seccion", "es_cotizacion", "es_patrocinado"})


def _fila(titulo, resumen="", link="http://x/n", fuente="ambito_negocios",
          fecha="", extras=TODAS):
    n = Noticia(titulo=titulo, resumen=resumen, fuente=fuente, fecha=fecha, link=link)
    return construir_fila(n, sentimiento=0.1, fecha_scrape="2026-10-10", extras=extras)


# ---- Parte 1: empresas se envía como LISTA, no string (bug doble-codificación) ----

def test_empresas_es_lista_no_string():
    fila = _fila("YPF anuncia inversión en Vaca Muerta")
    assert isinstance(fila["empresas"], list)
    assert fila["empresas"] == ["YPFD"]


def test_empresas_vacia_es_lista_vacia():
    fila = _fila("La inflación de la zona euro se aceleró en agosto")
    assert fila["empresas"] == []
    assert not isinstance(fila["empresas"], str)


def test_empresas_multiples():
    fila = _fila("Galicia y Pampa Energía lideran la suba del Merval")
    assert set(fila["empresas"]) == {"GGAL", "PAMP"}


def test_detectar_empresas_sin_falso_positivo():
    assert detectar_empresas("La selección argentina ganó en Brasil") == []


# ---- medio / seccion ----

def test_medio_y_seccion():
    assert medio_y_seccion("ambito_negocios") == ("ambito", "negocios")
    assert medio_y_seccion("cronista") == ("cronista", "general")
    assert medio_y_seccion("iprofesional") == ("iprofesional", "economia")
    assert medio_y_seccion("cronista_economia-politica") == ("cronista", "economia-politica")


def test_fila_separa_medio_y_seccion():
    fila = _fila("Nota", fuente="eleconomista_energia")
    assert fila["medio"] == "eleconomista"
    assert fila["seccion"] == "energia"


def test_fila_sin_columna_seccion_preserva_feed_key():
    # Si la migración de seccion no corrió, se guarda el feed_key en medio (no se pierde info)
    fila = _fila("Nota", fuente="eleconomista_energia", extras=frozenset())
    assert fila["medio"] == "eleconomista_energia"
    assert "seccion" not in fila
    assert "es_cotizacion" not in fila


# ---- Parte 3: es_cotizacion con ejemplos reales del corpus ----

COTIZACIONES = [
    "Dólar hoy: a cuánto cotiza este viernes 2 de octubre",
    "Dólar blue hoy: a cuánto cerró este lunes 5 de octubre",
    "Euro hoy y Euro blue hoy: a cuánto cotizó este lunes",
    "Real blue: a cuánto operó este lunes 5 de octubre",
]

NO_COTIZACIONES = [
    "Efecto Brasil: las acciones argentinas treparon hasta 10%",
    "Cuánto se gana con plazo fijo Banco Galicia si se invierten $100.000",
    "YPF anuncia nueva inversión en Vaca Muerta",
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
    assert _fila("Dólar hoy: a cuánto cotiza este viernes")["es_cotizacion"] is True
    assert _fila("YPF descubre un nuevo yacimiento")["es_cotizacion"] is False


# ---- Parte 2: es_patrocinado (contenido pagado) ----

def test_es_patrocinado_brand_strategy():
    assert es_patrocinado("https://www.cronista.com/brand-strategy/nota-pagada/")
    assert es_patrocinado("https://x.com/contenido-patrocinado/y/")


def test_es_patrocinado_no_marca_editorial():
    assert not es_patrocinado("https://www.cronista.com/economia-politica/ypf-deuda/")
    assert not es_patrocinado("https://www.ambito.com/energia/ypf-luz/")


def test_fila_marca_es_patrocinado():
    assert _fila("Nota pagada", link="https://cronista.com/brand-strategy/x/")["es_patrocinado"] is True
    assert _fila("YPF compra", link="https://cronista.com/negocios/ypf/")["es_patrocinado"] is False


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
