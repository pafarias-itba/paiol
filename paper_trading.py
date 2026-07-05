"""
PAIOL — Paper Trading (Simulación sin plata real)

Registra predicciones diarias y al día siguiente verifica contra
precios reales. Acumula historial de performance para decidir
cuándo pasar a operar en real.

Flujo:
1. pipeline.py corre con --dry-run → genera predicciones
2. paper_trading registra las predicciones con precio de entrada
3. Al correr al día siguiente, cierra las posiciones del día anterior
   con el precio real y calcula ganancia/pérdida
4. Acumula métricas: win rate, P&L, Sharpe, drawdown

Uso:
    from paper_trading import PaperTrader

    pt = PaperTrader(capital_inicial=100000)

    # Registrar predicciones del día
    pt.registrar_dia(predicciones, ordenes, cotizaciones)

    # Al día siguiente, cerrar posiciones con precios reales
    pt.cerrar_dia_anterior()

    # Ver performance
    pt.reporte()

    # Correr todo automático
    python paper_trading.py
"""

import os
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

from features import TICKERS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PAIOL-Paper")

PAPER_DIR = Path("paper_trading")
HISTORIAL_FILE = PAPER_DIR / "historial.csv"
POSICIONES_FILE = PAPER_DIR / "posiciones_abiertas.json"


class PaperTrader:
    """Simulador de trading para evaluar el modelo sin plata real."""

    def __init__(self, capital_inicial: float = 100_000):
        self.capital_inicial = capital_inicial
        PAPER_DIR.mkdir(exist_ok=True)

        # Cargar historial existente
        if HISTORIAL_FILE.exists():
            self.historial = pd.read_csv(HISTORIAL_FILE, parse_dates=["fecha"])
            logger.info(f"Historial cargado: {len(self.historial)} operaciones")
        else:
            self.historial = pd.DataFrame(columns=[
                "fecha", "ticker", "accion", "cantidad", "precio_entrada",
                "precio_salida", "pnl", "pnl_pct", "confianza",
                "senal_correcta", "cerrada",
            ])

        # Cargar posiciones abiertas
        if POSICIONES_FILE.exists():
            with open(POSICIONES_FILE, "r") as f:
                self.posiciones_abiertas = json.load(f)
        else:
            self.posiciones_abiertas = {}

    def _guardar(self):
        """Persiste historial y posiciones."""
        self.historial.to_csv(HISTORIAL_FILE, index=False)
        with open(POSICIONES_FILE, "w") as f:
            json.dump(self.posiciones_abiertas, f, indent=2, default=str)

    # ================================================================
    # REGISTRAR PREDICCIONES DEL DÍA
    # ================================================================

    def registrar_dia(self, predicciones: dict, ordenes: dict, cotizaciones: dict):
        """
        Registra las predicciones y órdenes simuladas del día.

        Args:
            predicciones: {ticker: {"prediccion": 0/1/2, "confianza": float, ...}}
            ordenes: {ticker: Orden} del position_sizing
            cotizaciones: {ticker: {"ultimoPrecio": float}}
        """
        fecha = datetime.now().strftime("%Y-%m-%d")

        # Primero cerrar posiciones del día anterior
        self.cerrar_posiciones_abiertas()

        # Registrar nuevas posiciones
        nuevas = 0
        for ticker, orden in ordenes.items():
            if hasattr(orden, 'accion'):
                accion = orden.accion
                cantidad = orden.cantidad
            else:
                accion = orden.get("accion", "HOLD")
                cantidad = orden.get("cantidad", 0)

            if accion == "HOLD" or cantidad == 0:
                continue

            precio = cotizaciones.get(ticker, {}).get("ultimoPrecio", 0)
            if not precio:
                continue

            pred = predicciones.get(ticker, {})
            confianza = pred.get("confianza", 0)

            # Registrar posición abierta
            self.posiciones_abiertas[ticker] = {
                "fecha": fecha,
                "accion": accion,
                "cantidad": cantidad,
                "precio_entrada": precio,
                "confianza": confianza,
                "senal": pred.get("prediccion", 1),
            }
            nuevas += 1
            logger.info(f"📝 {ticker}: {accion} x{cantidad} @ ${precio:,.0f}")

        self._guardar()
        logger.info(f"Registradas {nuevas} posiciones para {fecha}")

    # ================================================================
    # CERRAR POSICIONES CON PRECIO REAL
    # ================================================================

    def cerrar_posiciones_abiertas(self):
        """
        Cierra posiciones del día anterior descargando el precio
        de cierre real con yfinance.
        """
        if not self.posiciones_abiertas:
            return

        logger.info("Cerrando posiciones del día anterior...")
        cerradas = 0

        for ticker, pos in list(self.posiciones_abiertas.items()):
            config = TICKERS.get(ticker)
            if not config:
                continue

            # Descargar precio de cierre de hoy
            try:
                df = yf.download(
                    config["yfinance"],
                    period="2d",
                    progress=False,
                )
                if df.empty or len(df) < 1:
                    logger.warning(f"{ticker}: sin datos de precio, manteniendo abierta")
                    continue

                precio_salida = float(df["Close"].iloc[-1])
            except Exception as e:
                logger.error(f"{ticker}: error descargando precio: {e}")
                continue

            # Calcular P&L
            precio_entrada = pos["precio_entrada"]
            cantidad = pos["cantidad"]
            accion = pos["accion"]

            if accion == "COMPRAR":
                pnl = (precio_salida - precio_entrada) * cantidad
                pnl_pct = (precio_salida / precio_entrada) - 1
                senal_correcta = precio_salida > precio_entrada
            else:  # VENDER
                pnl = (precio_entrada - precio_salida) * cantidad
                pnl_pct = (precio_entrada / precio_salida) - 1
                senal_correcta = precio_salida < precio_entrada

            # Registrar en historial
            nueva_fila = pd.DataFrame([{
                "fecha": pos["fecha"],
                "ticker": ticker,
                "accion": accion,
                "cantidad": cantidad,
                "precio_entrada": precio_entrada,
                "precio_salida": round(precio_salida, 2),
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl_pct, 4),
                "confianza": pos["confianza"],
                "senal_correcta": senal_correcta,
                "cerrada": True,
            }])
            self.historial = pd.concat([self.historial, nueva_fila], ignore_index=True)

            emoji = "✅" if pnl > 0 else "❌"
            logger.info(
                f"{emoji} {ticker}: {accion} x{cantidad} "
                f"${precio_entrada:,.0f} → ${precio_salida:,.0f} "
                f"P&L: ${pnl:,.0f} ({pnl_pct:+.2%})"
            )
            cerradas += 1

        # Limpiar posiciones cerradas
        self.posiciones_abiertas = {}
        self._guardar()
        logger.info(f"Cerradas {cerradas} posiciones")

    # ================================================================
    # MÉTRICAS DE PERFORMANCE
    # ================================================================

    def calcular_metricas(self) -> dict:
        """Calcula métricas de performance del paper trading."""
        if self.historial.empty or len(self.historial) == 0:
            return {"mensaje": "Sin operaciones todavía"}

        h = self.historial[self.historial["cerrada"] == True].copy()

        if h.empty:
            return {"mensaje": "Sin operaciones cerradas todavía"}

        total_ops = len(h)
        ganadoras = len(h[h["pnl"] > 0])
        perdedoras = len(h[h["pnl"] < 0])
        neutras = len(h[h["pnl"] == 0])

        win_rate = ganadoras / total_ops if total_ops > 0 else 0

        pnl_total = h["pnl"].sum()
        pnl_promedio = h["pnl"].mean()
        pnl_max_ganancia = h["pnl"].max()
        pnl_max_perdida = h["pnl"].min()

        # Ganancia promedio vs pérdida promedio
        avg_win = h[h["pnl"] > 0]["pnl"].mean() if ganadoras > 0 else 0
        avg_loss = abs(h[h["pnl"] < 0]["pnl"].mean()) if perdedoras > 0 else 0
        profit_factor = avg_win / avg_loss if avg_loss > 0 else float("inf")

        # P&L acumulado
        h_sorted = h.sort_values("fecha")
        pnl_acumulado = h_sorted["pnl"].cumsum()
        max_drawdown = (pnl_acumulado - pnl_acumulado.cummax()).min()

        # Retorno sobre capital
        retorno_total = pnl_total / self.capital_inicial

        # Acierto por confianza
        alta_confianza = h[h["confianza"] >= 0.55]
        win_rate_alta = (
            len(alta_confianza[alta_confianza["senal_correcta"]]) / len(alta_confianza)
            if len(alta_confianza) > 0 else 0
        )

        return {
            "total_operaciones": total_ops,
            "ganadoras": ganadoras,
            "perdedoras": perdedoras,
            "neutras": neutras,
            "win_rate": round(win_rate, 4),
            "pnl_total": round(pnl_total, 2),
            "pnl_promedio": round(pnl_promedio, 2),
            "mejor_operacion": round(pnl_max_ganancia, 2),
            "peor_operacion": round(pnl_max_perdida, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 2),
            "max_drawdown": round(max_drawdown, 2),
            "retorno_total": round(retorno_total, 4),
            "win_rate_alta_confianza": round(win_rate_alta, 4),
            "dias_operados": h["fecha"].nunique(),
        }

    # ================================================================
    # REPORTE
    # ================================================================

    def reporte(self) -> str:
        """Genera reporte completo de paper trading."""
        metricas = self.calcular_metricas()

        if "mensaje" in metricas:
            return f"\n📊 PAPER TRADING: {metricas['mensaje']}\n"

        lines = [
            f"\n{'='*60}",
            f"  PAIOL Paper Trading — Reporte de Performance",
            f"  Capital inicial: ${self.capital_inicial:,.0f} ARS",
            f"{'='*60}",
            "",
            f"📊 RESUMEN ({metricas['dias_operados']} días operados)",
            f"  Operaciones totales: {metricas['total_operaciones']}",
            f"  Ganadoras: {metricas['ganadoras']} | "
            f"Perdedoras: {metricas['perdedoras']} | "
            f"Neutras: {metricas['neutras']}",
            f"  Win rate: {metricas['win_rate']:.1%}",
            "",
            f"💰 P&L",
            f"  Total: ${metricas['pnl_total']:,.0f} ARS ({metricas['retorno_total']:+.2%})",
            f"  Promedio por op: ${metricas['pnl_promedio']:,.0f}",
            f"  Mejor operación: ${metricas['mejor_operacion']:+,.0f}",
            f"  Peor operación: ${metricas['peor_operacion']:+,.0f}",
            "",
            f"📈 CALIDAD",
            f"  Ganancia promedio: ${metricas['avg_win']:,.0f}",
            f"  Pérdida promedio: ${metricas['avg_loss']:,.0f}",
            f"  Profit factor: {metricas['profit_factor']:.2f}x",
            f"  Max drawdown: ${metricas['max_drawdown']:,.0f}",
            "",
            f"🎯 CONFIANZA DEL MODELO",
            f"  Win rate general: {metricas['win_rate']:.1%}",
            f"  Win rate alta confianza (>55%): {metricas['win_rate_alta_confianza']:.1%}",
            "",
        ]

        # Criterios para pasar a real
        listo = (
            metricas["total_operaciones"] >= 20
            and metricas["win_rate"] >= 0.50
            and metricas["profit_factor"] >= 1.2
            and metricas["pnl_total"] > 0
        )

        if metricas["total_operaciones"] < 20:
            lines.append(f"⏳ VEREDICTO: Faltan datos — necesitás al menos 20 operaciones "
                        f"({metricas['total_operaciones']}/20)")
        elif listo:
            lines.append(f"✅ VEREDICTO: El modelo parece listo para operar en real")
        else:
            razones = []
            if metricas["win_rate"] < 0.50:
                razones.append(f"win rate {metricas['win_rate']:.1%} < 50%")
            if metricas["profit_factor"] < 1.2:
                razones.append(f"profit factor {metricas['profit_factor']:.2f} < 1.2")
            if metricas["pnl_total"] <= 0:
                razones.append(f"P&L negativo (${metricas['pnl_total']:,.0f})")
            lines.append(f"❌ VEREDICTO: No operar en real todavía — {', '.join(razones)}")

        lines.append(f"\n{'='*60}")
        return "\n".join(lines)

    # ================================================================
    # HISTORIAL POR TICKER
    # ================================================================

    def reporte_por_ticker(self) -> str:
        """Desglose de performance por ticker."""
        h = self.historial[self.historial["cerrada"] == True]
        if h.empty:
            return "Sin operaciones cerradas."

        lines = ["\n📊 PERFORMANCE POR TICKER\n"]

        for ticker in sorted(h["ticker"].unique()):
            t = h[h["ticker"] == ticker]
            ops = len(t)
            wins = len(t[t["pnl"] > 0])
            pnl = t["pnl"].sum()
            wr = wins / ops if ops > 0 else 0

            emoji = "✅" if pnl > 0 else "❌"
            lines.append(
                f"  {emoji} {ticker}: {ops} ops, WR {wr:.0%}, "
                f"P&L ${pnl:+,.0f}"
            )

        return "\n".join(lines)


# ================================================================
# ENTRY POINT — CORRER DIARIAMENTE
# ================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PAIOL Paper Trading")
    parser.add_argument("--reporte", action="store_true", help="Mostrar reporte de performance")
    parser.add_argument("--cerrar", action="store_true", help="Cerrar posiciones del día anterior")
    parser.add_argument("--registrar", type=str, help="Registrar desde archivo resultado JSON")
    args = parser.parse_args()

    pt = PaperTrader(capital_inicial=100_000)

    if args.reporte:
        print(pt.reporte())
        print(pt.reporte_por_ticker())

    elif args.cerrar:
        pt.cerrar_posiciones_abiertas()
        print(pt.reporte())

    elif args.registrar:
        # Cargar resultado del pipeline
        with open(args.registrar, "r") as f:
            resultado = json.load(f)

        predicciones = resultado.get("predicciones", {})

        # Simular cotizaciones desde las predicciones
        # En producción esto viene de IOL
        from features import PaiolFeatures
        pf = PaiolFeatures()
        cotizaciones = {}
        for ticker in predicciones:
            config = TICKERS.get(ticker)
            if config:
                try:
                    df = yf.download(config["yfinance"], period="1d", progress=False)
                    if not df.empty:
                        cotizaciones[ticker] = {"ultimoPrecio": float(df["Close"].iloc[-1])}
                except:
                    pass

        # Simular órdenes con position sizing
        from position_sizing import PositionManager
        pm = PositionManager(capital_total=100_000, riesgo_pct=0.02)
        ordenes = pm.calcular_todas(predicciones, cotizaciones)

        # Convertir Orden objects a dicts
        ordenes_dict = {}
        for ticker, orden in ordenes.items():
            ordenes_dict[ticker] = {
                "accion": orden.accion,
                "cantidad": orden.cantidad,
            }

        pt.registrar_dia(predicciones, ordenes_dict, cotizaciones)
        print(f"\n✅ Posiciones registradas. Mañana corré: python paper_trading.py --cerrar")

    else:
        print("Uso:")
        print("  python paper_trading.py --registrar outputs/resultado_YYYYMMDD.json")
        print("  python paper_trading.py --cerrar")
        print("  python paper_trading.py --reporte")
