"""
PAIOL — Recolector de NOTICIAS individuales (corpus para la tesis)

A diferencia de `scraper_sentiment.py` (que manda un lote a Gemini y guarda
sólo el sentimiento AGREGADO del día), este módulo guarda CADA noticia con:
  - titulo, resumen, medio, seccion, link, fecha_publicacion
  - empresas mencionadas (GGAL/YPFD/PAMP/MELI)         -> H2/H3
  - flag de jerga económica local (cepo/brecha/CCL...) -> H1
  - sentimiento del titular según Gemini (-1..1)       -> H1
  - es_cotizacion: notas de precio ("a cuánto cotiza...") -> excluir (circular)
  - es_patrocinado: contenido pagado (cronista brand-strategy, etc.) -> excluir

Las secciones se eligieron con evidencia GDELT 2024 (ver datos_gdelt): se leen las
secciones donde aparecieron las notas sobre GGAL/YPF/Pampa (negocios, energía,
economía-política), apuntando a ~80% de cobertura por medio.

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
# FUENTES — elegidas por evidencia GDELT 2024 (secciones con notas de empresa)
# ================================================================
# Objetivo: ~80% de las notas sobre GGAL/YPF/Pampa por medio. Las empresas
# aparecen sobre todo en energía, negocios y economía-política, que el pipeline
# de trading no lee. Se suman sólo acá. Todas verificadas (responden + ítems) 2026-10-10.

NOTICIAS_FEEDS = {
    **RSS_FEEDS,  # trading: ambito economia/finanzas/politica, cronista (news.xml),
                  # infobae_economia, eleconomista economia/finanzas, iprofesional (economia)
    # ámbito — clave: energia (63/104); + negocios, economia (ya en RSS_FEEDS)
    "ambito_negocios":           "https://www.ambito.com/rss/pages/negocios.xml",
    "ambito_energia":            "https://www.ambito.com/rss/pages/energia.xml",
    # cronista — economia-politica (81) y negocios (60); finanzas-mercados no tiene RSS
    "cronista_negocios":         "https://www.cronista.com/files/rss/negocios.xml",
    "cronista_economia-politica": "https://www.cronista.com/files/rss/economia-politica.xml",
    # eleconomista — clave: energia (35/67); + economia (RSS_FEEDS), politica
    "eleconomista_negocios":     "https://eleconomista.com.ar/negocios/feed/",
    "eleconomista_energia":      "https://eleconomista.com.ar/energia/feed/",
    "eleconomista_politica":     "https://eleconomista.com.ar/politica/feed/",
    # iprofesional — negocios (44), economia (RSS_FEEDS), actualidad (21), finanzas (16)
    "iprofesional_negocios":     "https://www.iprofesional.com/rss/negocios",
    "iprofesional_finanzas":     "https://www.iprofesional.com/rss/finanzas",
    "iprofesional_actualidad":   "https://www.iprofesional.com/rss/actualidad",
    # iprofesional_energia: GDELT 2024 no lo registra, pero en vivo trae notas de YPF.
    "iprofesional_energia":      "https://www.iprofesional.com/rss/energia",
}

# feed_key -> (medio, seccion). Para guardarlos separados y poder medir secciones.
FEED_META = {
    "ambito_economia": ("ambito", "economia"),
    "ambito_finanzas": ("ambito", "finanzas"),
    "ambito_politica": ("ambito", "politica"),
    "ambito_negocios": ("ambito", "negocios"),
    "ambito_energia": ("ambito", "energia"),
    "cronista": ("cronista", "general"),
    "cronista_negocios": ("cronista", "negocios"),
    "cronista_economia-politica": ("cronista", "economia-politica"),
    "infobae_economia": ("infobae", "economia"),
    "eleconomista_economia": ("eleconomista", "economia"),
    "eleconomista_finanzas": ("eleconomista", "finanzas"),
    "eleconomista_negocios": ("eleconomista", "negocios"),
    "eleconomista_energia": ("eleconomista", "energia"),
    "eleconomista_politica": ("eleconomista", "politica"),
    "iprofesional": ("iprofesional", "economia"),
    "iprofesional_negocios": ("iprofesional", "negocios"),
    "iprofesional_finanzas": ("iprofesional", "finanzas"),
    "iprofesional_actualidad": ("iprofesional", "actualidad"),
    "iprofesional_energia": ("iprofesional", "energia"),
}


def medio_y_seccion(feed_key: str):
    """(medio, seccion) desde la clave del feed. Fallback: split por el primer '_'."""
    if feed_key in FEED_META:
        return FEED_META[feed_key]
    if "_" in feed_key:
        m, s = feed_key.split("_", 1)
        return m, s
    return feed_key, "general"


# ================================================================
# TAGGEO DETERMINÍSTICO (empresa / jerga / cotización / patrocinado)
# ================================================================
# Por keywords (no LLM) a propósito: reproducible y auditable para la tesis.

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

# Cotizaciones automáticas: describen el precio -> usarlas para explicar retornos
# sería circular.
COTIZACION_TITULO = re.compile(
    r"a\s+cu[áa]nto\s+(cotiza|cotiz[óo]|abre|abri[óo]|cierra|cerr[óo]|opera|oper[óo])",
    re.IGNORECASE,
)

# Contenido patrocinado (pagado por las empresas, no periodístico). Se detecta por la
# sección en la URL. Según GDELT 2024 el único caso real es cronista "brand-strategy";
# el resto queda como red defensiva por si aparecen en algún feed general.
PATROCINADO_URL = re.compile(
    r"/(brand-strategy|contenido-patrocinado|branded|publinota|"
    r"espacio-no-editorial|sponsor|publicidad)/",
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
    return any(j in texto.lower() for j in JERGA_LOCAL)


def es_cotizacion(titulo: str, link: str = "") -> bool:
    """True si la nota es una cotización automática de precio (a excluir del análisis)."""
    if titulo and COTIZACION_TITULO.search(titulo):
        return True
    if link and "/cotizacion/" in link.lower():
        return True
    return False


def es_patrocinado(link: str = "") -> bool:
    """True si la nota es contenido patrocinado/pagado (por la sección en la URL)."""
    return bool(link and PATROCINADO_URL.search(link))


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


def construir_fila(noticia, sentimiento, fecha_scrape: str, extras=frozenset()) -> dict:
    """Arma el dict de una noticia listo para insertar en Supabase.

    `extras` = columnas opcionales que EXISTEN en la tabla (seccion, es_cotizacion,
    es_patrocinado); se incluyen sólo si están, para no romper si falta la migración.

    Clave: `empresas` queda como LISTA (no json.dumps) -> array JSONB. Si todavía no
    existe la columna `seccion`, se guarda el feed_key completo en `medio` para no
    perder info (la migración después lo separa).
    """
    texto = f"{noticia.titulo} {noticia.resumen}"
    medio, seccion = medio_y_seccion(noticia.fuente)
    fila = {
        "link": noticia.link,
        "titulo": noticia.titulo,
        "resumen": noticia.resumen,
        "fecha_publicacion": parsear_fecha(noticia.fecha),
        "fecha_scrape": fecha_scrape,
        "empresas": detectar_empresas(texto),   # LISTA, no json.dumps
        "jerga_local": tiene_jerga(texto),
        "sentimiento_gemini": sentimiento,
    }
    if "seccion" in extras:
        fila["medio"] = medio
        fila["seccion"] = seccion
    else:
        fila["medio"] = noticia.fuente          # preserva info si falta la columna
    if "es_cotizacion" in extras:
        fila["es_cotizacion"] = es_cotizacion(noticia.titulo, noticia.link)
    if "es_patrocinado" in extras:
        fila["es_patrocinado"] = es_patrocinado(noticia.link)
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


def columnas_opcionales(db) -> set:
    """Cuáles de las columnas opcionales existen ya en `noticias`."""
    presentes = set()
    for col in ("seccion", "es_cotizacion", "es_patrocinado"):
        try:
            r = requests.get(f"{db.url}/rest/v1/noticias", headers=db.headers,
                             params={"select": col, "limit": "0"})
            if r.status_code == 200:
                presentes.add(col)
        except Exception:
            pass
    return presentes


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

    # 1. Scrape (fuentes del trading + secciones de negocios/energía/economía-política)
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

    # 4. Armar filas (incluye sólo las columnas opcionales que existan)
    extras = columnas_opcionales(db)
    faltan = {"seccion", "es_cotizacion", "es_patrocinado"} - extras
    if faltan:
        logger.warning(f"Faltan columnas {sorted(faltan)} — corré sus migraciones. "
                       "Guardo sin esas marcas por ahora.")
    hoy = datetime.now().strftime("%Y-%m-%d")
    filas = [construir_fila(n, sentimientos.get(i), hoy, extras)
             for i, n in enumerate(nuevas)]

    # 5. Persistir (ignora duplicados por link, por las dudas)
    enviadas = db.guardar_noticias_batch(filas)
    con_empresa = sum(1 for f in filas if f["empresas"])
    total = db.contar_noticias()
    logger.info(f"Guardadas {enviadas} noticias nuevas ({con_empresa} con empresa)")
    logger.info(f"Corpus total acumulado: {total} noticias")
    return {"scrapeadas": len(noticias), "nuevas": len(nuevas), "guardadas": enviadas,
            "con_empresa": con_empresa, "corpus_total": total}


if __name__ == "__main__":
    con_sent = "--sin-sent" not in sys.argv
    recolectar(con_sentimiento=con_sent)
