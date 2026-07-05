"""
PAIOL — Persistencia en Supabase

Guarda todo el historial en la nube para no perder datos en redeploys:
- Operaciones de paper trading
- Sentiment diario (Gemini)
- Predicciones del modelo
- Métricas de re-entrenamiento
- Logs de ejecución

Setup:
    1. Crear proyecto en https://supabase.com
    2. Correr el SQL de create_tables() en el SQL Editor de Supabase
    3. Agregar SUPABASE_URL y SUPABASE_KEY al .env

Uso:
    from db import PaiolDB

    db = PaiolDB()
    db.guardar_prediccion("GGAL", 2, 0.57, {"BAJA": 0.28, "LATERAL": 0.14, "SUBE": 0.57})
    db.guardar_sentiment(sentiment_dict)
    db.guardar_operacion("GGAL", "COMPRAR", 3, 5500, 5650, 450)
"""

import os
import json
import logging
from datetime import datetime
from typing import Optional

import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PAIOL-DB")


# ================================================================
# SQL PARA CREAR LAS TABLAS (correr una vez en Supabase SQL Editor)
# ================================================================

CREATE_TABLES_SQL = """
-- Predicciones diarias del modelo
CREATE TABLE IF NOT EXISTS predicciones (
    id BIGSERIAL PRIMARY KEY,
    fecha DATE NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    senal INTEGER NOT NULL,          -- 0=Baja, 1=Lateral, 2=Sube
    accion VARCHAR(10) NOT NULL,     -- COMPRAR, VENDER, HOLD
    confianza FLOAT NOT NULL,
    prob_baja FLOAT,
    prob_lateral FLOAT,
    prob_sube FLOAT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(fecha, ticker)
);

-- Sentiment diario de Gemini
CREATE TABLE IF NOT EXISTS sentiment (
    id BIGSERIAL PRIMARY KEY,
    fecha DATE NOT NULL,
    dolar_sentiment FLOAT DEFAULT 0,
    reservas_sentiment FLOAT DEFAULT 0,
    riesgo_pais_sentiment FLOAT DEFAULT 0,
    inflacion_sentiment FLOAT DEFAULT 0,
    eventos_clave JSONB DEFAULT '[]',
    instrumentos JSONB DEFAULT '{}',
    raw_response JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(fecha)
);

-- Operaciones de paper trading
CREATE TABLE IF NOT EXISTS operaciones (
    id BIGSERIAL PRIMARY KEY,
    fecha DATE NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    accion VARCHAR(10) NOT NULL,     -- COMPRAR, VENDER
    cantidad INTEGER NOT NULL,
    precio_entrada FLOAT NOT NULL,
    precio_salida FLOAT,
    pnl FLOAT,
    pnl_pct FLOAT,
    confianza FLOAT,
    senal_correcta BOOLEAN,
    cerrada BOOLEAN DEFAULT FALSE,
    fecha_cierre DATE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Cotizaciones diarias (para no depender de yfinance retroactivo)
CREATE TABLE IF NOT EXISTS cotizaciones (
    id BIGSERIAL PRIMARY KEY,
    fecha DATE NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    open_price FLOAT,
    high_price FLOAT,
    low_price FLOAT,
    close_price FLOAT,
    volume BIGINT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(fecha, ticker)
);

-- Historial de modelos entrenados
CREATE TABLE IF NOT EXISTS modelos (
    id BIGSERIAL PRIMARY KEY,
    fecha DATE NOT NULL,
    f1_weighted FLOAT,
    accuracy FLOAT,
    train_size INTEGER,
    test_size INTEGER,
    reemplazo BOOLEAN DEFAULT FALSE,
    parametros JSONB,
    notas TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Logs de ejecución del pipeline
CREATE TABLE IF NOT EXISTS pipeline_logs (
    id BIGSERIAL PRIMARY KEY,
    fecha DATE NOT NULL,
    hora TIME NOT NULL,
    paso VARCHAR(50),
    status VARCHAR(20),              -- OK, ERROR, SKIP
    mensaje TEXT,
    duracion_seg FLOAT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Índices para queries frecuentes
CREATE INDEX IF NOT EXISTS idx_predicciones_fecha ON predicciones(fecha);
CREATE INDEX IF NOT EXISTS idx_sentiment_fecha ON sentiment(fecha);
CREATE INDEX IF NOT EXISTS idx_operaciones_fecha ON operaciones(fecha);
CREATE INDEX IF NOT EXISTS idx_operaciones_ticker ON operaciones(ticker);
CREATE INDEX IF NOT EXISTS idx_cotizaciones_fecha_ticker ON cotizaciones(fecha, ticker);
"""


class PaiolDB:
    """Cliente de Supabase para persistencia de PAIOL."""

    def __init__(self, url: str = None, key: str = None):
        self.url = url or os.environ.get("SUPABASE_URL")
        self.key = key or os.environ.get("SUPABASE_KEY")

        if not self.url or not self.key:
            raise ValueError(
                "Configurá SUPABASE_URL y SUPABASE_KEY en tu .env\n"
                "Los encontrás en Supabase → Settings → API"
            )

        self.headers = {
            "apikey": self.key,
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal",
        }

    def _post(self, table: str, data: dict) -> bool:
        """Inserta una fila en una tabla."""
        url = f"{self.url}/rest/v1/{table}"
        try:
            resp = requests.post(url, headers=self.headers, json=data)
            if resp.status_code in (200, 201, 204):
                return True
            else:
                logger.error(f"DB insert {table}: {resp.status_code} {resp.text}")
                return False
        except Exception as e:
            logger.error(f"DB error {table}: {e}")
            return False

    def _upsert(self, table: str, data: dict, on_conflict: str = "") -> bool:
        """Inserta o actualiza una fila."""
        url = f"{self.url}/rest/v1/{table}"
        headers = {**self.headers, "Prefer": "resolution=merge-duplicates,return=minimal"}
        try:
            resp = requests.post(url, headers=headers, json=data)
            if resp.status_code in (200, 201, 204):
                return True
            else:
                logger.error(f"DB upsert {table}: {resp.status_code} {resp.text}")
                return False
        except Exception as e:
            logger.error(f"DB error {table}: {e}")
            return False

    def _get(self, table: str, params: dict = None) -> list:
        """Consulta filas de una tabla."""
        url = f"{self.url}/rest/v1/{table}"
        headers = {**self.headers, "Prefer": "return=representation"}
        try:
            resp = requests.get(url, headers=headers, params=params)
            if resp.status_code == 200:
                return resp.json()
            else:
                logger.error(f"DB get {table}: {resp.status_code} {resp.text}")
                return []
        except Exception as e:
            logger.error(f"DB error {table}: {e}")
            return []

    def _patch(self, table: str, params: dict, data: dict) -> bool:
        """Actualiza filas que matcheen los params."""
        url = f"{self.url}/rest/v1/{table}"
        headers = {**self.headers, "Prefer": "return=minimal"}
        try:
            resp = requests.patch(url, headers=headers, params=params, json=data)
            if resp.status_code in (200, 204):
                return True
            else:
                logger.error(f"DB patch {table}: {resp.status_code} {resp.text}")
                return False
        except Exception as e:
            logger.error(f"DB error {table}: {e}")
            return False

    # ================================================================
    # PREDICCIONES
    # ================================================================

    def guardar_prediccion(self, ticker: str, senal: int, confianza: float,
                           probabilidades: dict, fecha: str = None):
        """Guarda una predicción del modelo."""
        if not fecha:
            fecha = datetime.now().strftime("%Y-%m-%d")

        labels = {0: "VENDER", 1: "HOLD", 2: "COMPRAR"}

        return self._upsert("predicciones", {
            "fecha": fecha,
            "ticker": ticker,
            "senal": senal,
            "accion": labels.get(senal, "HOLD"),
            "confianza": confianza,
            "prob_baja": probabilidades.get("BAJA", 0),
            "prob_lateral": probabilidades.get("LATERAL", 0),
            "prob_sube": probabilidades.get("SUBE", 0),
        })

    def guardar_predicciones_batch(self, predicciones: dict, fecha: str = None):
        """Guarda todas las predicciones del día."""
        for ticker, pred in predicciones.items():
            self.guardar_prediccion(
                ticker=ticker,
                senal=pred["prediccion"],
                confianza=pred["confianza"],
                probabilidades=pred.get("probabilidades", {}),
                fecha=fecha,
            )
        logger.info(f"DB: {len(predicciones)} predicciones guardadas")

    # ================================================================
    # SENTIMENT
    # ================================================================

    def guardar_sentiment(self, sentiment: dict):
        """Guarda el análisis de sentiment del día."""
        fecha = sentiment.get("fecha", datetime.now().strftime("%Y-%m-%d"))
        macro = sentiment.get("macro", {})

        return self._upsert("sentiment", {
            "fecha": fecha,
            "dolar_sentiment": macro.get("dolar_sentiment", 0),
            "reservas_sentiment": macro.get("reservas_sentiment", 0),
            "riesgo_pais_sentiment": macro.get("riesgo_pais_sentiment", 0),
            "inflacion_sentiment": macro.get("inflacion_sentiment", 0),
            "eventos_clave": json.dumps(sentiment.get("eventos_clave", [])),
            "instrumentos": json.dumps(sentiment.get("instrumentos", {})),
            "raw_response": json.dumps(sentiment),
        })

    def get_sentiment_historico(self, dias: int = 30) -> list:
        """Obtiene el sentiment de los últimos N días."""
        return self._get("sentiment", {
            "order": "fecha.desc",
            "limit": dias,
        })

    # ================================================================
    # OPERACIONES (PAPER TRADING)
    # ================================================================

    def guardar_operacion(self, ticker: str, accion: str, cantidad: int,
                          precio_entrada: float, confianza: float = 0,
                          fecha: str = None):
        """Registra una operación abierta."""
        if not fecha:
            fecha = datetime.now().strftime("%Y-%m-%d")

        return self._post("operaciones", {
            "fecha": fecha,
            "ticker": ticker,
            "accion": accion,
            "cantidad": cantidad,
            "precio_entrada": precio_entrada,
            "confianza": confianza,
            "cerrada": False,
        })

    def cerrar_operacion(self, op_id: int, precio_salida: float,
                         pnl: float, pnl_pct: float, senal_correcta: bool):
        """Cierra una operación con el resultado."""
        return self._patch("operaciones",
            {"id": f"eq.{op_id}"},
            {
                "precio_salida": precio_salida,
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl_pct, 4),
                "senal_correcta": senal_correcta,
                "cerrada": True,
                "fecha_cierre": datetime.now().strftime("%Y-%m-%d"),
            }
        )

    def get_operaciones_abiertas(self) -> list:
        """Obtiene operaciones sin cerrar."""
        return self._get("operaciones", {
            "cerrada": "eq.false",
            "order": "fecha.desc",
        })

    def get_operaciones_cerradas(self, limit: int = 100) -> list:
        """Obtiene operaciones cerradas."""
        return self._get("operaciones", {
            "cerrada": "eq.true",
            "order": "fecha.desc",
            "limit": limit,
        })

    def get_todas_operaciones(self) -> list:
        """Obtiene todas las operaciones."""
        return self._get("operaciones", {"order": "fecha.desc"})

    # ================================================================
    # COTIZACIONES
    # ================================================================

    def guardar_cotizacion(self, ticker: str, open_p: float, high: float,
                           low: float, close: float, volume: int,
                           fecha: str = None):
        """Guarda cotización diaria."""
        if not fecha:
            fecha = datetime.now().strftime("%Y-%m-%d")

        return self._upsert("cotizaciones", {
            "fecha": fecha,
            "ticker": ticker,
            "open_price": open_p,
            "high_price": high,
            "low_price": low,
            "close_price": close,
            "volume": volume,
        })

    def guardar_cotizaciones_batch(self, cotizaciones: dict, fecha: str = None):
        """Guarda cotizaciones de todos los tickers."""
        for ticker, data in cotizaciones.items():
            if data:
                self.guardar_cotizacion(
                    ticker=ticker,
                    open_p=data.get("apertura", 0),
                    high=data.get("maximo", 0),
                    low=data.get("minimo", 0),
                    close=data.get("ultimoPrecio", 0),
                    volume=data.get("volumen", 0),
                    fecha=fecha,
                )
        logger.info(f"DB: {len(cotizaciones)} cotizaciones guardadas")

    # ================================================================
    # MODELOS
    # ================================================================

    def guardar_modelo(self, f1: float, accuracy: float, train_size: int,
                       test_size: int, reemplazo: bool, parametros: dict = None,
                       notas: str = ""):
        """Registra un entrenamiento de modelo."""
        return self._post("modelos", {
            "fecha": datetime.now().strftime("%Y-%m-%d"),
            "f1_weighted": f1,
            "accuracy": accuracy,
            "train_size": train_size,
            "test_size": test_size,
            "reemplazo": reemplazo,
            "parametros": json.dumps(parametros or {}),
            "notas": notas,
        })

    # ================================================================
    # LOGS
    # ================================================================

    def log_pipeline(self, paso: str, status: str, mensaje: str = "",
                     duracion: float = 0):
        """Registra un paso del pipeline."""
        now = datetime.now()
        return self._post("pipeline_logs", {
            "fecha": now.strftime("%Y-%m-%d"),
            "hora": now.strftime("%H:%M:%S"),
            "paso": paso,
            "status": status,
            "mensaje": mensaje[:500],
            "duracion_seg": duracion,
        })

    # ================================================================
    # MÉTRICAS AGREGADAS
    # ================================================================

    def get_metricas_paper_trading(self) -> dict:
        """Calcula métricas de performance desde la DB."""
        ops = self.get_operaciones_cerradas(limit=1000)

        if not ops:
            return {"mensaje": "Sin operaciones cerradas"}

        total = len(ops)
        ganadoras = sum(1 for o in ops if (o.get("pnl") or 0) > 0)
        pnl_total = sum(o.get("pnl", 0) or 0 for o in ops)
        win_rate = ganadoras / total if total > 0 else 0

        pnls = [o.get("pnl", 0) or 0 for o in ops]
        avg_win = sum(p for p in pnls if p > 0) / max(ganadoras, 1)
        perdedoras = sum(1 for p in pnls if p < 0)
        avg_loss = abs(sum(p for p in pnls if p < 0)) / max(perdedoras, 1)
        profit_factor = avg_win / avg_loss if avg_loss > 0 else float("inf")

        return {
            "total_operaciones": total,
            "ganadoras": ganadoras,
            "perdedoras": perdedoras,
            "win_rate": round(win_rate, 4),
            "pnl_total": round(pnl_total, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 2),
        }


# ================================================================
# SETUP — Correr una vez
# ================================================================

def print_setup_sql():
    """Imprime el SQL para crear las tablas en Supabase."""
    print("=" * 60)
    print("  Copiá este SQL en Supabase → SQL Editor → Run")
    print("=" * 60)
    print(CREATE_TABLES_SQL)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PAIOL Database Setup")
    parser.add_argument("--setup", action="store_true",
                       help="Mostrar SQL para crear tablas")
    parser.add_argument("--test", action="store_true",
                       help="Testear conexión a Supabase")
    parser.add_argument("--metricas", action="store_true",
                       help="Mostrar métricas de paper trading")
    args = parser.parse_args()

    if args.setup:
        print_setup_sql()

    elif args.test:
        from dotenv import load_dotenv
        load_dotenv()
        db = PaiolDB()
        db.log_pipeline("test", "OK", "Conexión exitosa")
        logs = db._get("pipeline_logs", {"limit": 1, "order": "created_at.desc"})
        if logs:
            print(f"✅ Conexión exitosa. Último log: {logs[0]}")
        else:
            print("❌ No se pudo verificar la conexión")

    elif args.metricas:
        from dotenv import load_dotenv
        load_dotenv()
        db = PaiolDB()
        metricas = db.get_metricas_paper_trading()
        print(json.dumps(metricas, indent=2))

    else:
        parser.print_help()
