"""
PAIOL — Módulo de integración con API de InvertirOnline (IOL)

Endpoints:
- Autenticación (bearer + refresh token)
- Cotizaciones (tiempo real e históricas)
- Estado de cuenta y portafolio
- Órdenes de compra/venta
- Cancelación de órdenes

Uso:
    from iol_client import IOLClient

    client = IOLClient("tu_usuario", "tu_password")
    # Consultar cotización
    cotiz = client.get_cotizacion("GGAL", "bcba")
    # Comprar
    orden = client.comprar("GGAL", "bcba", cantidad=10, precio=1500.0)

IMPORTANTE: Configurar credenciales vía variables de entorno, NUNCA hardcodear.
"""

import os
import time
import json
import logging
import requests
from datetime import datetime, timedelta
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PAIOL-IOL")


class IOLClient:
    """Cliente para la API REST de InvertirOnline."""

    # URLs base
    BASE_URL = "https://api.invertironline.com"
    SANDBOX_URL = "https://api-sandbox.invertironline.com"

    def __init__(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
        sandbox: bool = False,
    ):
        """
        Args:
            username: Usuario de IOL (o env var IOL_USERNAME)
            password: Contraseña de IOL (o env var IOL_PASSWORD)
            sandbox: True para usar entorno de pruebas
        """
        self.username = username or os.environ.get("IOL_USERNAME")
        self.password = password or os.environ.get("IOL_PASSWORD")

        if not self.username or not self.password:
            raise ValueError(
                "Credenciales requeridas. Seteá IOL_USERNAME e IOL_PASSWORD "
                "como variables de entorno o pasalas al constructor."
            )

        self.base_url = self.SANDBOX_URL if sandbox else self.BASE_URL
        self.sandbox = sandbox

        # Token state
        self._access_token = None
        self._refresh_token = None
        self._token_expiry = None

        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

        # Autenticar al inicializar
        self._authenticate()

    # ================================================================
    # AUTENTICACIÓN
    # ================================================================

    def _authenticate(self):
        """Obtiene bearer token + refresh token."""
        url = f"{self.base_url}/token"
        payload = {
            "grant_type": "password",
            "username": self.username,
            "password": self.password,
        }
        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        response = requests.post(url, data=payload, headers=headers)
        response.raise_for_status()
        data = response.json()

        self._access_token = data["access_token"]
        self._refresh_token = data["refresh_token"]
        # Bearer token dura 15 minutos, renovamos a los 14 por seguridad
        self._token_expiry = datetime.now() + timedelta(minutes=14)

        self.session.headers.update(
            {"Authorization": f"Bearer {self._access_token}"}
        )
        logger.info("Autenticación exitosa%s", " (sandbox)" if self.sandbox else "")

    def _refresh_auth(self):
        """Renueva el token usando el refresh token."""
        url = f"{self.base_url}/token"
        payload = {
            "grant_type": "refresh_token",
            "refresh_token": self._refresh_token,
        }
        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        response = requests.post(url, data=payload, headers=headers)
        if response.status_code != 200:
            logger.warning("Refresh falló, re-autenticando desde cero")
            self._authenticate()
            return

        data = response.json()
        self._access_token = data["access_token"]
        self._refresh_token = data["refresh_token"]
        self._token_expiry = datetime.now() + timedelta(minutes=14)

        self.session.headers.update(
            {"Authorization": f"Bearer {self._access_token}"}
        )
        logger.info("Token renovado")

    def _ensure_token(self):
        """Verifica que el token sea válido, renueva si es necesario."""
        if datetime.now() >= self._token_expiry:
            self._refresh_auth()

    def _get(self, endpoint: str, params: dict = None) -> dict:
        """GET request con manejo automático de token."""
        self._ensure_token()
        url = f"{self.base_url}/api/v2{endpoint}"
        response = self.session.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def _post(self, endpoint: str, data: dict = None) -> dict:
        """POST request con manejo automático de token."""
        self._ensure_token()
        url = f"{self.base_url}/api/v2{endpoint}"
        response = self.session.post(url, json=data)
        response.raise_for_status()
        return response.json()

    def _delete(self, endpoint: str) -> dict:
        """DELETE request con manejo automático de token."""
        self._ensure_token()
        url = f"{self.base_url}/api/v2{endpoint}"
        response = self.session.delete(url)
        response.raise_for_status()
        return response.json()

    # ================================================================
    # COTIZACIONES
    # ================================================================

    def get_cotizacion(self, simbolo: str, mercado: str = "bcba") -> dict:
        """
        Obtiene la cotización actual de un instrumento.

        Args:
            simbolo: Ticker del instrumento (ej: "GGAL", "YPFD")
            mercado: "bcba" para BYMA, "nyse", "nasdaq"

        Returns:
            dict con precio, volumen, variación, puntas, etc.
        """
        return self._get(f"/Titulos/{mercado}/{simbolo}/CotizacionDetalle")

    def get_cotizaciones_panel(self, panel: str, pais: str = "argentina") -> list:
        """
        Obtiene cotizaciones de un panel completo.

        Args:
            panel: "merval", "panelGeneral", "cedears", etc.
            pais: "argentina", "estados_unidos"
        """
        return self._get(f"/{pais}/Titulos/Cotizacion/Paneles/{panel}")

    def get_historico(
        self,
        simbolo: str,
        mercado: str = "bcba",
        fecha_desde: str = None,
        fecha_hasta: str = None,
        ajustada: bool = False,
    ) -> list:
        """
        Obtiene cotizaciones históricas.

        Args:
            simbolo: Ticker del instrumento
            mercado: Mercado ("bcba", "nyse", "nasdaq")
            fecha_desde: "YYYY-MM-DD" (default: 1 año atrás)
            fecha_hasta: "YYYY-MM-DD" (default: hoy)
            ajustada: True para precios ajustados por splits/dividendos
        """
        if not fecha_desde:
            fecha_desde = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        if not fecha_hasta:
            fecha_hasta = datetime.now().strftime("%Y-%m-%d")

        endpoint = "/Titulos/{mercado}/{simbolo}/Cotizacion/serieHistorica/{desde}/{hasta}/{ajustada}"
        endpoint = endpoint.format(
            mercado=mercado,
            simbolo=simbolo,
            desde=fecha_desde,
            hasta=fecha_hasta,
            ajustada="ajustada" if ajustada else "sinAjustar",
        )
        return self._get(endpoint)

    # ================================================================
    # CUENTA Y PORTAFOLIO
    # ================================================================

    def get_estado_cuenta(self) -> dict:
        """Obtiene el estado de cuenta (saldos disponibles en pesos y dólares)."""
        return self._get("/estadocuenta")

    def get_portafolio(self, pais: str = "argentina") -> dict:
        """
        Obtiene el portafolio actual.

        Returns:
            dict con activos, cantidades, precios, rendimientos
        """
        return self._get(f"/portafolio/{pais}")

    # ================================================================
    # ÓRDENES (COMPRA / VENTA)
    # ================================================================

    def comprar(
        self,
        simbolo: str,
        mercado: str = "bcba",
        cantidad: int = 1,
        precio: float = None,
        plazo: str = "t0",
        validez: str = None,
    ) -> dict:
        """
        Envía una orden de compra.

        Args:
            simbolo: Ticker ("GGAL", "YPFD", etc.)
            mercado: "bcba" por defecto
            cantidad: Cantidad de títulos a comprar
            precio: Precio límite. None para precio de mercado
            plazo: "t0" (contado inmediato), "t1" (24hs), "t2" (48hs)
            validez: Fecha de validez "YYYY-MM-DD" (default: hoy)

        Returns:
            dict con número de orden y estado
        """
        if not validez:
            validez = datetime.now().strftime("%Y-%m-%d")

        payload = {
            "mercado": mercado,
            "simbolo": simbolo,
            "cantidad": cantidad,
            "precio": precio,
            "plazo": plazo,
            "validez": validez,
        }

        logger.info(
            "COMPRA: %s x%d @ %s (%s)",
            simbolo, cantidad, precio or "mercado", plazo
        )
        return self._post("/operar/Comprar", data=payload)

    def vender(
        self,
        simbolo: str,
        mercado: str = "bcba",
        cantidad: int = 1,
        precio: float = None,
        plazo: str = "t0",
        validez: str = None,
    ) -> dict:
        """
        Envía una orden de venta.

        Args:
            simbolo: Ticker
            mercado: "bcba"
            cantidad: Cantidad de títulos a vender
            precio: Precio límite. None para precio de mercado
            plazo: "t0", "t1", "t2"
            validez: Fecha de validez
        """
        if not validez:
            validez = datetime.now().strftime("%Y-%m-%d")

        payload = {
            "mercado": mercado,
            "simbolo": simbolo,
            "cantidad": cantidad,
            "precio": precio,
            "plazo": plazo,
            "validez": validez,
        }

        logger.info(
            "VENTA: %s x%d @ %s (%s)",
            simbolo, cantidad, precio or "mercado", plazo
        )
        return self._post("/operar/Vender", data=payload)

    def cancelar_orden(self, numero_orden: int) -> dict:
        """Cancela una orden pendiente por su número."""
        logger.info("CANCELAR orden #%d", numero_orden)
        return self._delete(f"/operaciones/{numero_orden}")

    def get_operaciones(self, estado: str = "todas") -> list:
        """
        Obtiene operaciones.

        Args:
            estado: "todas", "pendientes", "canceladas", "terminadas"
        """
        return self._get(f"/operaciones/{estado}")

    # ================================================================
    # HELPERS PARA PAIOL
    # ================================================================

    def get_cotizaciones_paiol(self) -> dict:
        """
        Obtiene cotizaciones actuales de todos los instrumentos PAIOL.

        Returns:
            dict: {ticker: {precio, variacion, volumen, ...}}
        """
        tickers = {
            "GGAL": "bcba",
            "YPFD": "bcba",
            "MELI": "bcba",
            "PAMP": "bcba",
            "SPY": "bcba",  # CEDEARs cotizan en BCBA
        }
        cotizaciones = {}
        for ticker, mercado in tickers.items():
            try:
                cotiz = self.get_cotizacion(ticker, mercado)
                cotizaciones[ticker] = cotiz
                logger.info(
                    "%s: $%.2f (%.2f%%)",
                    ticker,
                    cotiz.get("ultimoPrecio", 0),
                    cotiz.get("variacionPorcentual", 0),
                )
            except Exception as e:
                logger.error("Error obteniendo %s: %s", ticker, e)
                cotizaciones[ticker] = None

        return cotizaciones

    def ejecutar_senal(
        self,
        ticker: str,
        senal: int,
        cantidad: int,
        precio: float = None,
        mercado: str = "bcba",
    ) -> Optional[dict]:
        """
        Ejecuta una señal del modelo PAIOL.

        Args:
            ticker: Instrumento
            senal: 0=Baja (vender), 1=Lateral (hold), 2=Sube (comprar)
            cantidad: Cantidad de títulos
            precio: Precio límite (None = mercado)
            mercado: Mercado

        Returns:
            dict con resultado de la orden, o None si es hold
        """
        if senal == 2:  # SUBE → Comprar
            logger.info("SEÑAL COMPRA para %s", ticker)
            return self.comprar(ticker, mercado, cantidad, precio)
        elif senal == 0:  # BAJA → Vender
            logger.info("SEÑAL VENTA para %s", ticker)
            return self.vender(ticker, mercado, cantidad, precio)
        else:  # LATERAL → No operar
            logger.info("SEÑAL HOLD para %s — sin operación", ticker)
            return None

    def resumen_cuenta(self) -> dict:
        """
        Resumen rápido de la cuenta para el reporte diario de PAIOL.

        Returns:
            dict con saldos, portafolio y operaciones pendientes
        """
        return {
            "estado_cuenta": self.get_estado_cuenta(),
            "portafolio": self.get_portafolio(),
            "operaciones_pendientes": self.get_operaciones("pendientes"),
        }


# ================================================================
# USO RÁPIDO (para testing)
# ================================================================

if __name__ == "__main__":
    # Configurar credenciales como variables de entorno:
    #   export IOL_USERNAME="tu_usuario"
    #   export IOL_PASSWORD="tu_password"

    # Usar sandbox para pruebas
    client = IOLClient(sandbox=True)

    # Ver estado de cuenta
    print("\n=== ESTADO DE CUENTA ===")
    print(json.dumps(client.get_estado_cuenta(), indent=2))

    # Cotizaciones PAIOL
    print("\n=== COTIZACIONES PAIOL ===")
    cotiz = client.get_cotizaciones_paiol()
    for ticker, data in cotiz.items():
        if data:
            print(f"  {ticker}: ${data.get('ultimoPrecio', 'N/A')}")

    # Portafolio
    print("\n=== PORTAFOLIO ===")
    print(json.dumps(client.get_portafolio(), indent=2))
