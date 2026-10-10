"""
PAIOL — Recolector de NOTICIAS individuales (corpus para la tesis)

A diferencia de `scraper_sentiment.py` (que manda un lote a Gemini y guarda
sólo el sentimiento AGREGADO del día), este módulo guarda CADA noticia con:
  - titulo, resumen, medio, link, fecha_publicacion
  - empresas mencionadas (GGAL/YPFD/PAMP/MELI)         -> H2/H3
  - flag de jerga económica local (cepo/brecha/CCL...) -> H1
  - sentimiento del titular según Gemini (-1..1)       -> H1
  - es_cotizacion: marca notas de precio ("a cuánto cotiza...") para excluirlas
    del análisis causal (describen el precio -> serían circulares)

Reusa el RSSscraper existente. Suma secciones de negocios/energía (donde aparecen
las empresas) además de las fuentes del pipeline de trading.

Eficiencia: antes de clasificar con Gemini saltea los links ya guardados, así
correr seguido no re-clasifica lo mismo (permite subir la frecuencia barato).

Uso:
    python scraper_noticias.py            # scrapea, clasifica y persiste
    python scraper_noticias.py --sin-sent # sólo corpus, sin sentimiento Gemini
"""

import os
import re
import sys
import json
import logging
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime

import requests
from dotenv import load_dotenv
load_dotenv(override=True)

try:
    from google import genai  # noqa: F401 (algunas instalaciones exponen el namespace)
    import google.generativeai as genai
except ImportError:
    import google.generativeai as genai

from scraper_sentiment import RSSscraper, RSS_FEEDS
from db import PaiolDB

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("PAIOL-Noticias")


# ================================================================
# FUENTES — las del trading (RSS_FEEDS) + secciones de negocios/energía
# ================================================================
# Las empresas (GGAL/YPFD/PAMP) aparecen sobre todo en negocios y energía, que
# el pipeline de trading no lee. Se suman sólo acá para enriquecer el corpus.
# Todas verificadas (responden y traen ítems) el 2026-10-10.

NOTICIAS_FEEDS = {
    **RSS_FEEDS,
    "ambito_negocios":       "https://www.ambito.com/rss/pages/negocios.xml",
    "ambito_energia":        "https://www.ambito.com/rss/pages/energia.xml",
    "cronista_negocios":     "https://www.cronista.com/files/rss/negocios.xml",
    "iprofesional_negocios": "https://www.iprofesional.com/rss/negocios",
    "iprofesional_finanzas": "https://www.iprofesional.com/rss/finanzas",
    "iprofesional_energia":  "https://www.iprofesional.com/rss/energia",
    "eleconomista_negocios": "https://eleconomista.com.ar/negocios/feed/",
    "eleconomista_energia":  "https://eleconomista.com.ar/energia/feed/",
}


# ================================================================
# TAGGEO DETERMINÍSTICO (empresa + jerga + cotización)
# ================================================================
# Se hace por keywords (no con el LLM) a propósito: es reproducible y auditable,
# algo deseable para la tesis. La lista se puede refinar sin re-scrapear.

EMPRESAS_KEYWORDS = {
    "GGAL": [r"ggal", r"galicia", r"grupo financiero galicia", r"banco galicia"],
    "YPFD": [r"ypfd?", r"y\.?p\.?f\.?"],
    "PAMP": [r"pamp", r"pampa energ[ií]a"],
    "MELI": [r"meli", r"mercado\s?libre"],
}

JERGA_LOCAL = [
    "cepo", "brecha", "contado con liqui", "ccl", "dólar blue", "dolar blue",
    "dólar mep", "dolar mep", "dólar tarjeta", "dolar tarjeta", "cedear",
    "leliq", "lefi", "bopreal", "blanqueo", "carry trade", "crawling peg",
    "riesgo país", "riesgo pais", "pase pasivo", "dólar soja", "dolar soja",
]

# Notas automáticas de cotización: describen el precio ("a cuánto cotiza/abre/cerró
# el dólar hoy", URLs de /cotizacion/). Usarlas para explicar retornos sería circular.
COTIZACION_TITULO = re.compile(
    r"a\s+cu[áa]nto\s+(cotiza|cotiz[óo]|abre|abri[óo]|cierra|cerr[óo]|opera|oper[óo])",
    re.IGNORECASE,
)


def detectar_empresas(texto: str) -> list:
    """Devuelve la lista de tickers mencionados en el texto (word-boundary)."""
    t = texto.lower()
    encontradas = []
    for ticker, patrones in EMPRESAS_KEYWORDS.items():
        for p in patrones:
            if re.search(r"\b" + p + r"\b", t):
                encontradas.append(ticker)
                break
    return encontradas


def tiene_jerga(texto: str) -> bool:
    t = texto.lower()
    return any(j in t for j in JERGA_LOCAL)


def es_cotizacion(titulo: str, link: str = "") -> bool:
    """True si la nota es una cotización automática de precio (a excluir del análisis)."""
    if titulo and COTIZACION_TITULO.search(titulo):
        return True
    if link and "/cotizacion/" in link.lower():
        return True
    return False


def parsear_fecha(fecha_str: str):
    """Convierte la fecha RSS a ISO 8601 con timezone (o None si no se puede)."""
    if not fecha_str:
        return None
    try:
        return parsedate_to_datetime(fecha_str).isoformat()
    except (TypeError, ValueError):
        pass
    try:
        return datetime.fromisoformat(fecha_str.replace("Z", "+00:00")).isoformat()
    except ValueError:
        return None


def construir_fila(noticia, sentimiento, fecha_scrape: str, incluir_cotizacion: bool = True) -> dict:
    """Arma el dict de una noticia listo para insertar en Supabase.

    Clave: `empresas` queda como LISTA de Python (no string). requests serializa
    el body a JSON, así que llega como array JSONB. Hacerle json.dumps() acá la
    guardaría doble-codificada (bug corregido 2026-10-10).
    """
    texto = f"{noticia.titulo} {noticia.resumen}"
    fila = {
        "link": noticia.link,
        "titulo": noticia.titulo,
        "resumen": noticia.resumen,
        "medio": noticia.fuente,
        "fecha_publicacion": parsear_fecha(noticia.fecha),
        "fecha_scrape": fecha_scrape,
        "empresas": detectar_empresas(texto),   # LISTA, no json.dumps
        "jerga_local": tiene_jerga(texto),
        "sentimiento_gemini": sentimiento,
    }
    if incluir_cotizacion:
        fila["es_cotizacion"] = es_cotizacion(noticia.titulo, noticia.link)
    return fila


# ================================================================
# HELPERS DE BASE
# ================================================================

def links_ya_guardados(db, dias: int = 3) -> set:
    """Links ya en la tabla en los últimos N días (para no re-clasificar)."""
    desde = (datetime.now() - timedelta(days=dias)).strftime("%Y-%m-%d")
    try:
        r = requests.get(f"{db.url}/rest/v1/noticias", headers=db.headers,
                         params={"select": "link", "fecha_scrape": f"gte.{desde}",
                                 "limit": "10000"})
        return {row["link"] for row in r.json()} if r.ok else set()
    except Exception as e:
        logger.warning(f"No se pudo leer links existentes: {e}")
        return set()


def columna_existe(db, col: str) -> bool:
    """True si la columna existe en `noticias` (para degradar sin romper)."""
    try:
        r = requests.get(f"{db.url}/rest/v1/noticias", headers=db.headers,
                         params={"select": col, "limit": "0"})
        return r.status_code == 200
    except Exception:
        return False


# ================================================================
# SENTIMIENTO POR TITULAR (Gemini, en lotes)
# ================================================================

PROMPT_SENT = """Sos analista financiero del mercado argentino. Te paso un lote de
noticias numeradas. Para CADA una, estimá el tono financiero hacia las empresas o el
mercado argentino que menciona, de -1 (muy negativo) a 1 (muy positivo); 0 si es
neutral o no afecta a los precios.

Devolvé SOLAMENTE un JSON array (sin markdown, sin backticks, sin texto extra), con un
objeto por noticia, en el mismo orden:
[{{"n": 1, "sentimiento": -0.4}}, {{"n": 2, "sentimiento": 0.2}}]

NOTICIAS:
{bloque}
"""


class SentimentPorTitular:
    """Clasifica el sentimiento de cada titular con Gemini, en lotes."""

    def __init__(self, api_key: str = None, lote: int = 20):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.lote = lote
        self.activo = bool(self.api_key)
        if self.activo:
            genai.configure(api_key=self.api_key, transport="rest")
            self.model = genai.GenerativeModel("gemini-2.5-flash-lite")
        else:
            logger.warning("Sin GEMINI_API_KEY: se guarda el corpus sin sentimiento.")

    def clasificar(self, noticias: list) -> dict:
        """Devuelve {indice_en_la_lista: sentimiento}."""
        if not self.activo:
            return {}
        out = {}
        for inicio in range(0, len(noticias), self.lote):
            chunk = noticias[inicio:inicio + self.lote]
            bloque = "\n".join(
                f"{i}. {n.titulo} — {n.resumen[:160]}"
                for i, n in enumerate(chunk, 1)
            )
            try:
                resp = self.model.generate_content(PROMPT_SENT.format(bloque=bloque))
                for item in self._parse(resp.text):
                    idx = inicio + int(item["n"]) - 1
                    if 0 <= idx < len(noticias):
                        out[idx] = max(-1.0, min(1.0, float(item["sentimiento"])))
            except Exception as e:
                logger.error(f"  Gemini lote {inicio}-{inicio+len(chunk)}: {e}")
        return out

    @staticmethod
    def _parse(texto: str) -> list:
        texto = texto.strip()
        if texto.startswith("```"):
            texto = texto.split("\n", 1)[1]
        if texto.endswith("```"):
            texto = texto.rsplit("```", 1)[0]
        try:
            data = json.loads(texto.strip())
            return data if isinstance(data, list) else []
        except json.JSONDecodeError:
            logger.error("  No se pudo parsear la respuesta de Gemini")
            return []


# ================================================================
# ORQUESTACIÓN
# ================================================================

def recolectar(con_sentimiento: bool = True) -> dict:
    """Scrapea noticias relevantes nuevas, las clasifica y las persiste."""
    logger.info("=== PAIOL — Recolector de noticias ===")
    db = PaiolDB()

    # 1. Scrape (fuentes del trading + secciones de negocios/energía)
    noticias = RSSscraper(feeds=NOTICIAS_FEEDS).scrape_relevantes()
    if not noticias:
        logger.warning("Sin noticias relevantes en esta corrida.")
        return {"scrapeadas": 0, "nuevas": 0, "guardadas": 0}

    # 2. Pre-dedup: saltear las que ya están (ahorra Gemini y habilita alta frecuencia)
    ya = links_ya_guardados(db)
    nuevas = [n for n in noticias if n.link and n.link not in ya]
    logger.info(f"{len(noticias)} relevantes | {len(nuevas)} nuevas (no vistas antes)")
    if not nuevas:
        return {"scrapeadas": len(noticias), "nuevas": 0, "guardadas": 0,
                "corpus_total": db.contar_noticias()}

    # 3. Sentimiento por titular, sólo de las nuevas
    sentimientos = {}
    if con_sentimiento:
        logger.info(f"Clasificando sentimiento de {len(nuevas)} titulares...")
        sentimientos = SentimentPorTitular().clasificar(nuevas)

    # 4. Armar filas (degradar si falta la columna es_cotizacion)
    incluir_cotiz = columna_existe(db, "es_cotizacion")
    if not incluir_cotiz:
        logger.warning("Falta la columna es_cotizacion — corré migracion_es_cotizacion.sql. "
                       "Guardo sin esa marca por ahora.")
    hoy = datetime.now().strftime("%Y-%m-%d")
    filas = [construir_fila(n, sentimientos.get(i), hoy, incluir_cotiz)
             for i, n in enumerate(nuevas)]

    # 5. Persistir (ignora duplicados por link, por las dudas)
    enviadas = db.guardar_noticias_batch(filas)
    con_empresa = sum(1 for f in filas if f["empresas"])
    con_jerga = sum(1 for f in filas if f["jerga_local"])
    cotiz = sum(1 for f in filas if f.get("es_cotizacion")) if incluir_cotiz else 0
    total = db.contar_noticias()

    logger.info(f"Guardadas {enviadas} noticias nuevas "
                f"({con_empresa} con empresa, {con_jerga} con jerga, {cotiz} cotización)")
    logger.info(f"Corpus total acumulado: {total} noticias")
    return {"scrapeadas": len(noticias), "nuevas": len(nuevas), "guardadas": enviadas,
            "con_empresa": con_empresa, "con_jerga": con_jerga,
            "cotizacion": cotiz, "corpus_total": total}


if __name__ == "__main__":
    con_sent = "--sin-sent" not in sys.argv
    recolectar(con_sentimiento=con_sent)
