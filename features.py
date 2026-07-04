"""
PAIOL — Feature Engineering para Producción

Versión pandas del feature engineering del TP (que usa PySpark).
En producción procesamos pocos datos (1 ticker, ~60 días de historia para ventanas),
no necesitamos Spark.

Features:
- Técnicos: RSI, MACD, SMAs, volatilidad, volumen relativo, retornos
- Sentiment: del módulo scraper_sentiment.py (Gemini)

Uso:
    from features import PaiolFeatures

    pf = PaiolFeatures()

    # Generar features para hoy (descarga precios con yfinance)
    features = pf.build_features_hoy("GGAL.BA", sentiment_data)

    # Generar features para todos los tickers PAIOL
    all_features = pf.build_all(sentiment_data)
"""

import numpy as np
import pandas as pd
import yfinance as yf
import logging
from datetime import datetime, timedelta
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PAIOL-Features")


# Tickers PAIOL con su símbolo en yfinance y en IOL
TICKERS = {
    "GGAL": {"yfinance": "GGAL.BA", "iol": "GGAL", "mercado": "bcba"},
    "YPFD": {"yfinance": "YPFD.BA", "iol": "YPFD", "mercado": "bcba"},
    "MELI": {"yfinance": "MELI.BA", "iol": "MELI", "mercado": "bcba"},
    "PAMP": {"yfinance": "PAMP.BA", "iol": "PAMP", "mercado": "bcba"},
    "SPY":  {"yfinance": "SPY",     "iol": "SPY",  "mercado": "bcba"},
}

# Features que usa el modelo (mismo orden que en el TP)
FEATURE_COLS = [
    # Técnicos
    "returns", "returns_2d", "returns_5d",
    "rsi", "macd", "macd_signal", "macd_histogram",
    "sma_5_20_ratio", "sma_20_50_ratio", "price_sma20_ratio",
    "volatility_5d", "volatility_20d",
    "volume_ratio",
    # Sentiment
    "dolar_sentiment", "reservas_sentiment", "riesgo_pais_sentiment",
    "ticker_sentiment", "ticker_relevancia",
]

# Umbral para clasificación (mismo que el TP)
UMBRAL = 0.01


class PaiolFeatures:
    """Calcula features técnicos + sentiment para el modelo PAIOL."""

    def __init__(self, lookback_days: int = 120):
        """
        Args:
            lookback_days: Días de historia a descargar para calcular
                          ventanas (50 días de SMA + margen). Default 120.
        """
        self.lookback_days = lookback_days

    # ================================================================
    # DESCARGA DE PRECIOS
    # ================================================================

    def descargar_precios(self, ticker_yf: str) -> pd.DataFrame:
        """
        Descarga precios históricos recientes con yfinance.

        Args:
            ticker_yf: Símbolo de yfinance (ej: "GGAL.BA")

        Returns:
            DataFrame con columnas: date, open, high, low, close, volume
        """
        end = datetime.now()
        start = end - timedelta(days=self.lookback_days)

        df = yf.download(ticker_yf, start=start, end=end, progress=False)

        if df.empty:
            raise ValueError(f"No se pudieron descargar datos para {ticker_yf}")

        df = df.reset_index()

        # Aplanar MultiIndex si viene así
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] if col[1] == "" else col[0] for col in df.columns]

        df = df.rename(columns={
            "Date": "date", "Open": "open", "High": "high",
            "Low": "low", "Close": "close", "Volume": "volume",
        })
        df = df[["date", "open", "high", "low", "close", "volume"]]
        df = df.sort_values("date").reset_index(drop=True)

        logger.info(f"{ticker_yf}: {len(df)} días descargados ({df['date'].iloc[0].date()} → {df['date'].iloc[-1].date()})")
        return df

    # ================================================================
    # FEATURES TÉCNICOS
    # ================================================================

    def calcular_tecnicos(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula todos los features técnicos sobre un DataFrame de precios.
        Equivalente a la función de PySpark del TP pero en pandas.

        Args:
            df: DataFrame con columnas date, open, high, low, close, volume

        Returns:
            DataFrame con features técnicos agregados
        """
        df = df.copy().sort_values("date").reset_index(drop=True)

        # --- Retornos ---
        df["returns"] = df["close"].pct_change(1)
        df["returns_2d"] = df["close"].pct_change(2)
        df["returns_5d"] = df["close"].pct_change(5)

        # --- Medias Móviles ---
        df["sma_5"] = df["close"].rolling(5).mean()
        df["sma_20"] = df["close"].rolling(20).mean()
        df["sma_50"] = df["close"].rolling(50).mean()

        # Ratios de SMAs (más útiles que el SMA absoluto)
        df["sma_5_20_ratio"] = df["sma_5"] / df["sma_20"]
        df["sma_20_50_ratio"] = df["sma_20"] / df["sma_50"]
        df["price_sma20_ratio"] = df["close"] / df["sma_20"]

        # --- RSI (14 períodos) ---
        df["rsi"] = self._calcular_rsi(df["close"], 14)

        # --- MACD (12, 26, signal 9) ---
        ema_12 = df["close"].ewm(span=12, adjust=False).mean()
        ema_26 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = ema_12 - ema_26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_histogram"] = df["macd"] - df["macd_signal"]

        # --- Volatilidad ---
        df["volatility_5d"] = df["returns"].rolling(5).std()
        df["volatility_20d"] = df["returns"].rolling(20).std()

        # --- Volumen relativo ---
        df["vol_sma_20"] = df["volume"].rolling(20).mean()
        df["volume_ratio"] = df["volume"] / df["vol_sma_20"]

        # Limpiar columnas auxiliares
        df = df.drop(columns=["sma_5", "sma_20", "sma_50", "vol_sma_20"], errors="ignore")

        return df

    @staticmethod
    def _calcular_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Calcula RSI con el método EMA (más preciso que SMA)."""
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)

        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()

        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi

    # ================================================================
    # MERGE CON SENTIMENT
    # ================================================================

    def merge_sentiment(self, df: pd.DataFrame, sentiment: dict, ticker: str) -> pd.DataFrame:
        """
        Agrega features de sentiment al DataFrame de features técnicos.
        Solo agrega al último día (hoy), que es el que predecimos.

        Args:
            df: DataFrame con features técnicos
            sentiment: Output de PaiolSentiment.run() (JSON de Gemini)
            ticker: Ticker PAIOL ("GGAL", "YPFD", etc.)

        Returns:
            DataFrame con columnas de sentiment agregadas
        """
        df = df.copy()

        # Extraer valores de sentiment
        macro = sentiment.get("macro", {})
        instrumento = sentiment.get("instrumentos", {}).get(ticker, {})

        # Valores default si no hay datos
        dolar_sent = macro.get("dolar_sentiment", 0.0)
        reservas_sent = macro.get("reservas_sentiment", 0.0)
        riesgo_sent = macro.get("riesgo_pais_sentiment", 0.0)
        ticker_sent = instrumento.get("sentimiento", 0.0)
        ticker_rel = instrumento.get("relevancia", 0.0)

        # Agregar a todas las filas (para mantener el shape)
        # pero solo la última fila importa para la predicción
        df["dolar_sentiment"] = dolar_sent
        df["reservas_sentiment"] = reservas_sent
        df["riesgo_pais_sentiment"] = riesgo_sent
        df["ticker_sentiment"] = ticker_sent
        df["ticker_relevancia"] = ticker_rel

        return df

    # ================================================================
    # PIPELINE COMPLETO PARA UN TICKER
    # ================================================================

    def build_features_hoy(self, ticker: str, sentiment: dict = None) -> Optional[pd.Series]:
        """
        Pipeline completo para un ticker: descarga → features → sentiment → última fila.

        Args:
            ticker: Ticker PAIOL ("GGAL", "YPFD", etc.)
            sentiment: Output de PaiolSentiment.run()

        Returns:
            pd.Series con los features del día para pasar al modelo
        """
        config = TICKERS.get(ticker)
        if not config:
            raise ValueError(f"Ticker {ticker} no está en TICKERS PAIOL")

        try:
            # 1. Descargar precios
            df = self.descargar_precios(config["yfinance"])

            # 2. Calcular features técnicos
            df = self.calcular_tecnicos(df)

            # 3. Agregar sentiment
            if sentiment:
                df = self.merge_sentiment(df, sentiment, ticker)
            else:
                # Sin sentiment, poner 0s
                for col in ["dolar_sentiment", "reservas_sentiment",
                            "riesgo_pais_sentiment", "ticker_sentiment",
                            "ticker_relevancia"]:
                    df[col] = 0.0

            # 4. Tomar última fila (hoy) con features completos
            df = df.dropna(subset=FEATURE_COLS)
            if df.empty:
                logger.error(f"No hay features completos para {ticker}")
                return None

            ultima_fila = df.iloc[-1]

            logger.info(
                f"{ticker} features: RSI={ultima_fila['rsi']:.1f}, "
                f"MACD={ultima_fila['macd']:.4f}, "
                f"Vol5d={ultima_fila['volatility_5d']:.4f}, "
                f"Sent={ultima_fila['ticker_sentiment']:.2f}"
            )

            return ultima_fila[FEATURE_COLS]

        except Exception as e:
            logger.error(f"Error generando features para {ticker}: {e}")
            return None

    def build_all(self, sentiment: dict = None) -> dict:
        """
        Genera features para TODOS los tickers PAIOL.

        Args:
            sentiment: Output de PaiolSentiment.run()

        Returns:
            dict: {ticker: pd.Series de features} para cada ticker
        """
        logger.info("=== Generando features para todos los tickers ===")
        resultados = {}

        for ticker in TICKERS:
            features = self.build_features_hoy(ticker, sentiment)
            if features is not None:
                resultados[ticker] = features
            else:
                logger.warning(f"Sin features para {ticker}")

        logger.info(f"Features generados para {len(resultados)}/{len(TICKERS)} tickers")
        return resultados

    # ================================================================
    # LABEL (para re-entrenamiento)
    # ================================================================

    @staticmethod
    def generar_labels(df: pd.DataFrame, umbral: float = UMBRAL) -> pd.DataFrame:
        """
        Genera labels para re-entrenamiento del modelo.
        0=Baja, 1=Lateral, 2=Sube (basado en retorno del día siguiente).

        Args:
            df: DataFrame con columna 'close'
            umbral: % de movimiento para definir sube/baja

        Returns:
            DataFrame con columna 'label' agregada
        """
        df = df.copy()
        df["next_return"] = df["close"].shift(-1) / df["close"] - 1

        df["label"] = 1  # Lateral por defecto
        df.loc[df["next_return"] < -umbral, "label"] = 0  # Baja
        df.loc[df["next_return"] > umbral, "label"] = 2   # Sube

        # Última fila no tiene label (no sabemos el cierre de mañana)
        df.loc[df.index[-1], "label"] = np.nan

        return df


# ================================================================
# USO RÁPIDO
# ================================================================

if __name__ == "__main__":
    pf = PaiolFeatures()

    # Ejemplo sin sentiment (para testing rápido)
    print("=== Features GGAL ===")
    features_ggal = pf.build_features_hoy("GGAL")
    if features_ggal is not None:
        print(features_ggal)

    # Ejemplo con sentiment mock
    sentiment_mock = {
        "fecha": "2026-07-04",
        "macro": {
            "dolar_sentiment": 0.3,
            "reservas_sentiment": -0.1,
            "riesgo_pais_sentiment": -0.2,
            "inflacion_sentiment": 0.1,
        },
        "instrumentos": {
            "GGAL": {"sentimiento": 0.5, "relevancia": 0.8},
            "YPFD": {"sentimiento": -0.2, "relevancia": 0.6},
            "MELI": {"sentimiento": 0.1, "relevancia": 0.3},
            "PAMP": {"sentimiento": 0.0, "relevancia": 0.1},
            "SPY":  {"sentimiento": 0.4, "relevancia": 0.5},
        },
        "eventos_clave": ["BCRA mantiene tasa", "Dólar estable"],
    }

    print("\n=== Todos los tickers con sentiment ===")
    all_features = pf.build_all(sentiment_mock)
    for ticker, feat in all_features.items():
        print(f"\n{ticker}:")
        print(feat.to_string())
