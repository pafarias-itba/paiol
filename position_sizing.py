"""
PAIOL — Position Sizing y Gestión de Riesgo

Calcula cuántos títulos comprar/vender en función de:
- Capital disponible en la cuenta IOL
- Riesgo máximo por operación (% del capital)
- Precio actual del instrumento
- Confianza del modelo

Uso:
    from position_sizing import PositionManager

    pm = PositionManager(capital_total=100000, riesgo_pct=0.02)
    orden = pm.calcular_posicion("GGAL", precio=5500, senal=2, confianza=0.57)
    # {"ticker": "GGAL", "accion": "COMPRAR", "cantidad": 3, "monto": 16500, ...}
"""

import logging
from dataclasses import dataclass
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PAIOL-Position")


# Instrumentos con lote mínimo en BYMA
LOTE_MINIMO = {
    "GGAL": 1,
    "YPFD": 1,
    "MELI": 1,
    "PAMP": 1,
    "SPY": 1,
}


@dataclass
class Orden:
    ticker: str
    accion: str           # "COMPRAR", "VENDER", "HOLD"
    cantidad: int
    precio: float
    monto: float          # cantidad * precio
    pct_capital: float    # % del capital que usa esta orden
    riesgo_max: float     # máxima pérdida en $ (stop loss)
    stop_loss: float      # precio de stop loss
    motivo: str           # razón de la decisión


class PositionManager:
    """Gestión de posiciones y riesgo para PAIOL."""

    def __init__(
        self,
        capital_total: float = 100_000,
        riesgo_pct: float = 0.02,
        max_pct_por_instrumento: float = 0.30,
        stop_loss_pct: float = 0.03,
        confianza_minima: float = 0.45,
        confianza_maxima_posicion: float = 0.70,
    ):
        """
        Args:
            capital_total: Capital total en ARS
            riesgo_pct: Riesgo máximo por operación (0.02 = 2%)
            max_pct_por_instrumento: Máximo % del capital en un solo instrumento
            stop_loss_pct: % de caída para activar stop loss (0.03 = 3%)
            confianza_minima: No opera si la confianza del modelo es menor
            confianza_maxima_posicion: Confianza a partir de la cual usa posición máxima
        """
        self.capital_total = capital_total
        self.riesgo_pct = riesgo_pct
        self.max_pct_por_instrumento = max_pct_por_instrumento
        self.stop_loss_pct = stop_loss_pct
        self.confianza_minima = confianza_minima
        self.confianza_maxima_posicion = confianza_maxima_posicion

        # Valores derivados
        self.riesgo_max_por_op = capital_total * riesgo_pct
        self.max_por_instrumento = capital_total * max_pct_por_instrumento

        logger.info(f"PositionManager: capital={capital_total:,.0f} ARS, "
                    f"riesgo={riesgo_pct:.0%}/op, "
                    f"max/instrumento={max_pct_por_instrumento:.0%}, "
                    f"stop_loss={stop_loss_pct:.0%}")

    def actualizar_capital(self, capital: float):
        """Actualiza el capital (por ejemplo, después de consultar IOL)."""
        self.capital_total = capital
        self.riesgo_max_por_op = capital * self.riesgo_pct
        self.max_por_instrumento = capital * self.max_pct_por_instrumento
        logger.info(f"Capital actualizado: {capital:,.0f} ARS")

    def calcular_posicion(
        self,
        ticker: str,
        precio: float,
        senal: int,
        confianza: float,
        posicion_actual: int = 0,
    ) -> Orden:
        """
        Calcula la orden óptima para un ticker.

        Args:
            ticker: Instrumento ("GGAL", "YPFD", etc.)
            precio: Precio actual del título
            senal: 0=Baja, 1=Lateral, 2=Sube
            confianza: Probabilidad de la clase ganadora (0-1)
            posicion_actual: Títulos que ya tenés de este instrumento

        Returns:
            Orden con cantidad, precio, stop loss, etc.
        """
        # HOLD — no hacer nada
        if senal == 1:
            return Orden(
                ticker=ticker, accion="HOLD", cantidad=0,
                precio=precio, monto=0, pct_capital=0,
                riesgo_max=0, stop_loss=0,
                motivo="Señal LATERAL — sin operación",
            )

        # Confianza insuficiente
        if confianza < self.confianza_minima:
            return Orden(
                ticker=ticker, accion="HOLD", cantidad=0,
                precio=precio, monto=0, pct_capital=0,
                riesgo_max=0, stop_loss=0,
                motivo=f"Confianza {confianza:.1%} < mínima {self.confianza_minima:.0%}",
            )

        # Calcular tamaño de posición escalado por confianza
        factor_confianza = self._escalar_confianza(confianza)
        capital_asignable = self.max_por_instrumento * factor_confianza

        # Limitar por riesgo: no perder más del riesgo_pct del capital
        # cantidad_max_riesgo = riesgo_max / (precio * stop_loss_pct)
        cantidad_por_riesgo = int(self.riesgo_max_por_op / (precio * self.stop_loss_pct))

        # Limitar por capital asignable
        cantidad_por_capital = int(capital_asignable / precio)

        # Tomar el mínimo (el más conservador)
        cantidad = min(cantidad_por_riesgo, cantidad_por_capital)

        # Respetar lote mínimo
        lote_min = LOTE_MINIMO.get(ticker, 1)
        if cantidad < lote_min:
            return Orden(
                ticker=ticker, accion="HOLD", cantidad=0,
                precio=precio, monto=0, pct_capital=0,
                riesgo_max=0, stop_loss=0,
                motivo=f"Capital insuficiente: necesita {precio * lote_min:,.0f} ARS "
                       f"por lote mínimo de {lote_min}",
            )

        # Para VENDER, limitar a la posición actual
        if senal == 0:
            if posicion_actual <= 0:
                return Orden(
                    ticker=ticker, accion="HOLD", cantidad=0,
                    precio=precio, monto=0, pct_capital=0,
                    riesgo_max=0, stop_loss=0,
                    motivo="Señal BAJA pero no tenés posición para vender",
                )
            cantidad = min(cantidad, posicion_actual)

        # Calcular valores finales
        monto = cantidad * precio
        pct_capital = monto / self.capital_total
        stop_loss_precio = precio * (1 - self.stop_loss_pct) if senal == 2 else precio * (1 + self.stop_loss_pct)
        riesgo_max = cantidad * precio * self.stop_loss_pct

        accion = "COMPRAR" if senal == 2 else "VENDER"

        orden = Orden(
            ticker=ticker,
            accion=accion,
            cantidad=cantidad,
            precio=precio,
            monto=monto,
            pct_capital=pct_capital,
            riesgo_max=riesgo_max,
            stop_loss=round(stop_loss_precio, 2),
            motivo=f"{accion} x{cantidad} @ ${precio:,.0f} "
                   f"(confianza {confianza:.1%}, factor {factor_confianza:.2f})",
        )

        logger.info(
            f"{ticker}: {accion} x{cantidad} @ ${precio:,.0f} = ${monto:,.0f} "
            f"({pct_capital:.1%} del capital, riesgo ${riesgo_max:,.0f}, "
            f"SL ${stop_loss_precio:,.0f})"
        )

        return orden

    def _escalar_confianza(self, confianza: float) -> float:
        """
        Escala la posición según la confianza del modelo.
        
        - Confianza mínima (0.45) → 30% de la posición máxima
        - Confianza máxima (0.70+) → 100% de la posición máxima
        - Intermedio → escala lineal
        """
        if confianza >= self.confianza_maxima_posicion:
            return 1.0
        if confianza <= self.confianza_minima:
            return 0.0

        rango = self.confianza_maxima_posicion - self.confianza_minima
        factor = 0.3 + 0.7 * ((confianza - self.confianza_minima) / rango)
        return min(factor, 1.0)

    def calcular_todas(
        self,
        predicciones: dict,
        cotizaciones: dict,
        portafolio: dict = None,
    ) -> dict:
        """
        Calcula órdenes para todos los tickers.

        Args:
            predicciones: {ticker: {"prediccion": 0/1/2, "confianza": float}}
            cotizaciones: {ticker: {"ultimoPrecio": float}} (de IOL)
            portafolio: {ticker: cantidad_actual} (posiciones abiertas)

        Returns:
            {ticker: Orden}
        """
        if portafolio is None:
            portafolio = {}

        ordenes = {}
        capital_comprometido = 0

        for ticker, pred in predicciones.items():
            precio = cotizaciones.get(ticker, {}).get("ultimoPrecio")
            if not precio:
                logger.warning(f"{ticker}: sin cotización, saltando")
                continue

            posicion_actual = portafolio.get(ticker, 0)

            orden = self.calcular_posicion(
                ticker=ticker,
                precio=precio,
                senal=pred["prediccion"],
                confianza=pred["confianza"],
                posicion_actual=posicion_actual,
            )

            # Verificar que no nos pasemos del capital total
            if orden.accion == "COMPRAR":
                if capital_comprometido + orden.monto > self.capital_total:
                    orden = Orden(
                        ticker=ticker, accion="HOLD", cantidad=0,
                        precio=precio, monto=0, pct_capital=0,
                        riesgo_max=0, stop_loss=0,
                        motivo="Capital insuficiente (ya comprometido en otros tickers)",
                    )
                else:
                    capital_comprometido += orden.monto

            ordenes[ticker] = orden

        # Resumen
        total_compras = sum(o.monto for o in ordenes.values() if o.accion == "COMPRAR")
        total_riesgo = sum(o.riesgo_max for o in ordenes.values() if o.accion != "HOLD")
        logger.info(f"\nResumen: {total_compras:,.0f} ARS en compras, "
                    f"riesgo total {total_riesgo:,.0f} ARS "
                    f"({total_riesgo/self.capital_total:.1%} del capital)")

        return ordenes

    def resumen(self, ordenes: dict) -> str:
        """Genera resumen legible de las órdenes."""
        lines = [
            f"\n{'='*60}",
            f"  POSITION SIZING — Capital: ${self.capital_total:,.0f} ARS",
            f"{'='*60}",
            f"  {'Ticker':<8} {'Acción':<10} {'Cant':<6} {'Precio':<12} "
            f"{'Monto':<12} {'% Cap':<8} {'Stop Loss':<12} {'Riesgo'}",
            f"  {'-'*75}",
        ]

        for ticker, orden in ordenes.items():
            if orden.accion == "HOLD":
                lines.append(f"  {ticker:<8} {'HOLD':<10} {'-':<6} "
                           f"{'':12} {'':12} {'':8} {'':12} {orden.motivo}")
            else:
                lines.append(
                    f"  {ticker:<8} {orden.accion:<10} {orden.cantidad:<6} "
                    f"${orden.precio:>10,.0f} ${orden.monto:>10,.0f} "
                    f"{orden.pct_capital:>6.1%} ${orden.stop_loss:>10,.0f} "
                    f"${orden.riesgo_max:>,.0f}"
                )

        total_monto = sum(o.monto for o in ordenes.values() if o.accion == "COMPRAR")
        total_riesgo = sum(o.riesgo_max for o in ordenes.values())
        lines.append(f"  {'-'*75}")
        lines.append(f"  {'TOTAL':<8} {'':10} {'':6} {'':12} "
                    f"${total_monto:>10,.0f} "
                    f"{total_monto/self.capital_total:>6.1%} {'':12} "
                    f"${total_riesgo:>,.0f}")
        lines.append(f"{'='*60}")

        return "\n".join(lines)


# ================================================================
# USO RÁPIDO
# ================================================================

if __name__ == "__main__":
    pm = PositionManager(capital_total=100_000, riesgo_pct=0.02)

    # Simular predicciones del modelo (del resultado real)
    predicciones = {
        "GGAL": {"prediccion": 2, "confianza": 0.5771},
        "YPFD": {"prediccion": 2, "confianza": 0.5404},
        "PAMP": {"prediccion": 2, "confianza": 0.5834},
        "MELI": {"prediccion": 2, "confianza": 0.4145},
        "SPY":  {"prediccion": 1, "confianza": 0.7115},
    }

    # Simular cotizaciones
    cotizaciones = {
        "GGAL": {"ultimoPrecio": 5500},
        "YPFD": {"ultimoPrecio": 55000},
        "PAMP": {"ultimoPrecio": 4000},
        "MELI": {"ultimoPrecio": 25000},
        "SPY":  {"ultimoPrecio": 8000},
    }

    ordenes = pm.calcular_todas(predicciones, cotizaciones)
    print(pm.resumen(ordenes))
