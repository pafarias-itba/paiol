"""
PAIOL — Scraper de Noticias RSS + Análisis de Sentimiento con Gemini

Flujo:
1. Parsea RSS feeds de medios financieros argentinos
2. Filtra noticias relevantes al mercado
3. Envía batch a Gemini 2.0 Flash para análisis de sentimiento
4. Retorna JSON estructurado con sentiment por instrumento y macro

Uso:
    from scraper_sentiment import PaiolSentiment

    ps = PaiolSentiment()
    resultado = ps.run()
    print(resultado)
    # {
    #   "fecha": "2026-07-04",
    #   "macro": {"dolar_sentiment": 0.3, ...},
    #   "instrumentos": {"GGAL": {"sentimiento": 0.5, "relevancia": 0.8}, ...},
    #   "eventos_clave": ["BCRA sube tasa...", ...]
    # }
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass, field

import feedparser
import ssl
import certifi
os.environ["SSL_CERT_FILE"] = certifi.where()
try:
    from google import genai
except ImportError:
    import google.generativeai as genai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PAIOL-Sentiment")


# ================================================================
# CONFIGURACIÓN DE FUENTES RSS
# ================================================================

RSS_FEEDS = {
    # Ámbito Financiero — principal para mercado argentino
    "ambito_economia": "https://www.ambito.com/rss/pages/economia.xml",
    "ambito_finanzas": "https://www.ambito.com/rss/pages/finanzas.xml",
    "ambito_politica": "https://www.ambito.com/rss/pages/politica.xml",

    # El Cronista — noticias económicas/financieras
    "cronista": "https://www.cronista.com/files/rss/news.xml",

    # Infobae — sección economía
    "infobae_economia": "https://www.infobae.com/feeds/rss/economia/",

    # El Economista Argentina — finanzas
    "eleconomista_economia": "https://eleconomista.com.ar/economia/feed/",
    "eleconomista_finanzas": "https://eleconomista.com.ar/finanzas/feed/",

    # iProfesional — negocios y finanzas
    "iprofesional": "https://www.iprofesional.com/rss/economia",
}

# Instrumentos que PAIOL opera
TICKERS_PAIOL = ["GGAL", "YPFD", "YPF", "MELI", "PAMP", "SPY", "GALICIA", "PAMPA"]

# Keywords para filtrar noticias relevantes al mercado
KEYWORDS_MERCADO = [
    # Instrumentos
    "galicia", "ggal", "ypf", "ypfd", "mercadolibre", "meli", "pampa", "pamp",
    "cedear", "cedears", "spy", "s&p",
    # Mercado
    "merval", "bolsa", "acciones", "bonos", "riesgo país", "riesgo pais",
    "mercado", "wall street", "nasdaq", "byma",
    # Macro Argentina
    "dólar", "dolar", "blue", "mep", "ccl", "contado con liqui",
    "inflación", "inflacion", "ipc", "indec",
    "bcra", "banco central", "tasa", "tasas", "reservas",
    "deuda", "fmi", "cepo", "devaluación", "devaluacion",
    "presupuesto", "recaudación", "recaudacion", "deficit",
    # Política económica
    "milei", "caputo", "economía", "economia", "ley bases",
    "desregulación", "desregulacion", "privatización", "privatizacion",
]


# ================================================================
# SCRAPER RSS
# ================================================================

@dataclass
class Noticia:
    titulo: str
    resumen: str
    fuente: str
    fecha: str
    link: str


class RSSscraper:
    """Parsea y filtra noticias de feeds RSS argentinos."""

    def __init__(self, feeds: dict = None, max_horas: int = 24):
        """
        Args:
            feeds: dict de {nombre: url} de feeds RSS
            max_horas: solo noticias de las últimas N horas
        """
        self.feeds = feeds or RSS_FEEDS
        self.max_horas = max_horas
        self.cutoff = datetime.now() - timedelta(hours=max_horas)

    def scrape_all(self) -> list[Noticia]:
        """Descarga y parsea todos los feeds RSS."""
        todas = []
        for nombre, url in self.feeds.items():
            try:
                noticias = self._parse_feed(nombre, url)
                todas.extend(noticias)
                logger.info(f"{nombre}: {len(noticias)} noticias")
            except Exception as e:
                logger.error(f"Error en {nombre}: {e}")

        # Deduplicar por título
        seen = set()
        unique = []
        for n in todas:
            key = n.titulo.lower().strip()
            if key not in seen:
                seen.add(key)
                unique.append(n)

        logger.info(f"Total noticias únicas: {len(unique)}")
        return unique

    def scrape_relevantes(self) -> list[Noticia]:
        """Devuelve solo noticias relevantes al mercado financiero."""
        todas = self.scrape_all()
        relevantes = [n for n in todas if self._es_relevante(n)]
        logger.info(f"Noticias relevantes al mercado: {len(relevantes)}")
        return relevantes

    def _parse_feed(self, nombre: str, url: str) -> list[Noticia]:
        """Parsea un feed RSS individual."""
        feed = feedparser.parse(
            url,
            agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        )
        noticias = []

        for entry in feed.entries:
            # Extraer fecha
            fecha_str = ""
            if hasattr(entry, "published"):
                fecha_str = entry.published
            elif hasattr(entry, "updated"):
                fecha_str = entry.updated

            # Extraer resumen/contenido
            resumen = ""
            if hasattr(entry, "summary"):
                resumen = entry.summary
            elif hasattr(entry, "description"):
                resumen = entry.description

            # Limpiar HTML básico del resumen
            resumen = self._limpiar_html(resumen)

            noticia = Noticia(
                titulo=entry.get("title", ""),
                resumen=resumen[:500],  # Truncar para no pasarse del contexto
                fuente=nombre,
                fecha=fecha_str,
                link=entry.get("link", ""),
            )
            noticias.append(noticia)

        return noticias

    def _es_relevante(self, noticia: Noticia) -> bool:
        """Filtra noticias por keywords del mercado."""
        texto = (noticia.titulo + " " + noticia.resumen).lower()
        return any(kw in texto for kw in KEYWORDS_MERCADO)

    @staticmethod
    def _limpiar_html(texto: str) -> str:
        """Limpieza básica de tags HTML."""
        import re
        texto = re.sub(r"<[^>]+>", " ", texto)
        texto = re.sub(r"\s+", " ", texto).strip()
        return texto


# ================================================================
# ANÁLISIS DE SENTIMIENTO CON GEMINI
# ================================================================

# Prompt para Gemini
GEMINI_PROMPT = """Sos un analista financiero especializado en el mercado argentino.
Te voy a pasar un lote de noticias del día. Necesito que analices el sentimiento
y devuelvas SOLAMENTE un JSON (sin markdown, sin explicación, sin backticks) con esta estructura:

{{
  "fecha": "{fecha}",
  "macro": {{
    "dolar_sentiment": float entre -1 y 1,
    "reservas_sentiment": float entre -1 y 1,
    "riesgo_pais_sentiment": float entre -1 y 1,
    "inflacion_sentiment": float entre -1 y 1
  }},
  "instrumentos": {{
    "GGAL": {{ "sentimiento": float, "relevancia": float entre 0 y 1 }},
    "YPFD": {{ "sentimiento": float, "relevancia": float entre 0 y 1 }},
    "MELI": {{ "sentimiento": float, "relevancia": float entre 0 y 1 }},
    "PAMP": {{ "sentimiento": float, "relevancia": float entre 0 y 1 }},
    "SPY":  {{ "sentimiento": float, "relevancia": float entre 0 y 1 }}
  }},
  "eventos_clave": ["lista de strings con los 3-5 eventos más importantes del día"]
}}

Criterios:
- Sentimiento: -1 = muy negativo, 0 = neutral, 1 = muy positivo
- Relevancia: 0 = no se mencionó, 0.5 = mención indirecta, 1 = noticia directa
- Si un instrumento no aparece en ninguna noticia, sentimiento=0 y relevancia=0
- Para SPY considerá noticias sobre mercados internacionales / Wall Street
- El dolar_sentiment es sobre el peso: positivo = peso fuerte, negativo = devaluación
- riesgo_pais: positivo = baja del riesgo, negativo = sube
- Si no hay noticias suficientes, usá valores cercanos a 0

NOTICIAS DEL DÍA:
{noticias}
"""


class GeminiSentiment:
    """Analiza sentimiento de noticias usando Gemini 2.0 Flash."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Args:
            api_key: Gemini API key (o env var GEMINI_API_KEY)
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY requerida. Seteala como variable de entorno "
                "o pasala al constructor."
            )

        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel("gemini-2.5-flash-lite")

    def analizar(self, noticias: list[Noticia]) -> dict:
        """
        Envía lote de noticias a Gemini y obtiene análisis de sentimiento.

        Args:
            noticias: Lista de noticias del día

        Returns:
            dict con estructura de sentimiento PAIOL
        """
        if not noticias:
            logger.warning("Sin noticias para analizar")
            return self._default_output()

        # Formatear noticias para el prompt
        noticias_texto = self._formatear_noticias(noticias)

        # Armar prompt
        fecha_hoy = datetime.now().strftime("%Y-%m-%d")
        prompt = GEMINI_PROMPT.format(fecha=fecha_hoy, noticias=noticias_texto)

        # Llamar a Gemini
        try:
            response = self.model.generate_content(prompt)
            resultado = self._parse_response(response.text)
            logger.info("Análisis de sentimiento completado")
            return resultado
        except Exception as e:
            logger.error(f"Error en Gemini: {e}")
            return self._default_output()

    def _formatear_noticias(self, noticias: list[Noticia]) -> str:
        """Formatea noticias para incluir en el prompt."""
        lineas = []
        for i, n in enumerate(noticias[:50], 1):  # Max 50 noticias por request
            lineas.append(
                f"{i}. [{n.fuente}] {n.titulo}\n   {n.resumen[:200]}"
            )
        return "\n\n".join(lineas)

    def _parse_response(self, texto: str) -> dict:
        """Parsea la respuesta JSON de Gemini."""
        # Limpiar posibles backticks o markdown
        texto = texto.strip()
        if texto.startswith("```"):
            texto = texto.split("\n", 1)[1]
        if texto.endswith("```"):
            texto = texto.rsplit("```", 1)[0]
        texto = texto.strip()

        try:
            return json.loads(texto)
        except json.JSONDecodeError as e:
            logger.error(f"Error parseando JSON de Gemini: {e}")
            logger.debug(f"Respuesta raw: {texto[:500]}")
            return self._default_output()

    def _default_output(self) -> dict:
        """Output por defecto cuando no hay datos o hay error."""
        return {
            "fecha": datetime.now().strftime("%Y-%m-%d"),
            "macro": {
                "dolar_sentiment": 0.0,
                "reservas_sentiment": 0.0,
                "riesgo_pais_sentiment": 0.0,
                "inflacion_sentiment": 0.0,
            },
            "instrumentos": {
                ticker: {"sentimiento": 0.0, "relevancia": 0.0}
                for ticker in ["GGAL", "YPFD", "MELI", "PAMP", "SPY"]
            },
            "eventos_clave": [],
        }


# ================================================================
# CLASE PRINCIPAL — ORQUESTA TODO
# ================================================================

class PaiolSentiment:
    """
    Pipeline completo: RSS → filtrado → Gemini → JSON de sentimiento.

    Uso:
        ps = PaiolSentiment()
        resultado = ps.run()
    """

    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        feeds: dict = None,
        max_horas: int = 24,
    ):
        self.scraper = RSSscraper(feeds=feeds, max_horas=max_horas)
        self.gemini = GeminiSentiment(api_key=gemini_api_key)

    def run(self) -> dict:
        """Ejecuta el pipeline completo."""
        logger.info("=== PAIOL Sentiment Pipeline ===")

        # 1. Scrape
        logger.info("1. Scrapeando noticias...")
        noticias = self.scraper.scrape_relevantes()

        # 2. Analizar sentimiento
        logger.info(f"2. Analizando {len(noticias)} noticias con Gemini...")
        resultado = self.gemini.analizar(noticias)

        # 3. Log resumen
        logger.info(f"   Fecha: {resultado.get('fecha')}")
        macro = resultado.get("macro", {})
        logger.info(f"   Dólar sentiment: {macro.get('dolar_sentiment', 'N/A')}")
        logger.info(f"   Riesgo país: {macro.get('riesgo_pais_sentiment', 'N/A')}")

        eventos = resultado.get("eventos_clave", [])
        if eventos:
            logger.info("   Eventos clave:")
            for ev in eventos:
                logger.info(f"     • {ev}")

        return resultado

    def run_y_guardar(self, path: str = None) -> dict:
        """Ejecuta y guarda resultado como JSON."""
        resultado = self.run()

        if not path:
            fecha = datetime.now().strftime("%Y%m%d")
            path = f"sentiment_{fecha}.json"

        with open(path, "w", encoding="utf-8") as f:
            json.dump(resultado, f, ensure_ascii=False, indent=2)

        logger.info(f"Resultado guardado en {path}")
        return resultado


# ================================================================
# USO RÁPIDO
# ================================================================

if __name__ == "__main__":
    # Configurar:
    #   export GEMINI_API_KEY="tu_api_key"

    ps = PaiolSentiment()
    resultado = ps.run_y_guardar()
    print(json.dumps(resultado, indent=2, ensure_ascii=False))
