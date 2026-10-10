"""
PAIOL — Recolector de NOTICIAS individuales (corpus para la tesis)

A diferencia de `scraper_sentiment.py` (que manda un lote a Gemini y guarda
sólo el sentimiento AGREGADO del día), este módulo guarda CADA noticia con:
  - titulo, resumen, medio, link, fecha_publicacion
  - empresas mencionadas (GGAL/YPFD/PAMP/MELI)         -> H2/H3
  - flag de jerga económica local (cepo/brecha/CCL...) -> H1
  - sentimiento del titular según Gemini (-1..1)       -> H1

Reusa el RSSscraper existente (misma lista de fuentes y filtro de relevancia).
Pensado para correr como cron 2x/día, todos los días, y acumular el corpus.

Uso:
    python scraper_noticias.py            # scrapea, clasifica y persiste
    python scraper_noticias.py --sin-sent # sólo corpus, sin sentimiento Gemini
"""

import os
import re
import sys
import json
import logging
from datetime import datetime
from email.utils import parsedate_to_datetime

from dotenv import load_dotenv
load_dotenv(override=True)

try:
    from google import genai  # noqa: F401 (algunas instalaciones exponen el namespace)
    import google.generativeai as genai
except ImportError:
    import google.generativeai as genai

from scraper_sentiment import RSSscraper
from db import PaiolDB

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("PAIOL-Noticias")


# ================================================================
# TAGGEO DETERMINÍSTICO (empresa + jerga)
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
            # transport="rest": mismo motivo que en scraper_sentiment.py
            genai.configure(api_key=self.api_key, transport="rest")
            self.model = genai.GenerativeModel("gemini-2.5-flash-lite")
        else:
            logger.warning("Sin GEMINI_API_KEY: se guarda el corpus sin sentimiento.")

    def clasificar(self, noticias: list) -> dict:
        """Devuelve {indice_global: sentimiento} para la lista de noticias."""
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
    """Scrapea noticias relevantes, las clasifica y las persiste en Supabase."""
    logger.info("=== PAIOL — Recolector de noticias ===")
    db = PaiolDB()

    # 1. Scrape (reusa fuentes y filtro de relevancia del scraper existente)
    noticias = RSSscraper().scrape_relevantes()
    if not noticias:
        logger.warning("Sin noticias relevantes en esta corrida.")
        return {"scrapeadas": 0, "guardadas": 0}

    # 2. Sentimiento por titular (opcional)
    sentimientos = {}
    if con_sentimiento:
        logger.info(f"Clasificando sentimiento de {len(noticias)} titulares...")
        sentimientos = SentimentPorTitular().clasificar(noticias)

    # 3. Armar filas con taggeo
    hoy = datetime.now().strftime("%Y-%m-%d")
    filas = []
    for i, n in enumerate(noticias):
        if not n.link:
            continue  # sin link no hay clave de dedup
        texto = f"{n.titulo} {n.resumen}"
        filas.append({
            "link": n.link,
            "titulo": n.titulo,
            "resumen": n.resumen,
            "medio": n.fuente,
            "fecha_publicacion": parsear_fecha(n.fecha),
            "fecha_scrape": hoy,
            "empresas": detectar_empresas(texto),
            "jerga_local": tiene_jerga(texto),
            "sentimiento_gemini": sentimientos.get(i),
        })

    # 4. Persistir (ignora duplicados por link)
    enviadas = db.guardar_noticias_batch(filas)
    con_empresa = sum(1 for f in filas if f["empresas"])
    con_jerga = sum(1 for f in filas if f["jerga_local"])
    total = db.contar_noticias()

    logger.info(f"Enviadas {enviadas} noticias "
                f"({con_empresa} mencionan empresas, {con_jerga} con jerga local)")
    logger.info(f"Corpus total acumulado: {total} noticias")
    return {"scrapeadas": len(noticias), "guardadas": enviadas,
            "con_empresa": con_empresa, "con_jerga": con_jerga, "corpus_total": total}


if __name__ == "__main__":
    con_sent = "--sin-sent" not in sys.argv
    recolectar(con_sentimiento=con_sent)
