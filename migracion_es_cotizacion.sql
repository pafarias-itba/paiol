-- ================================================================
-- Migración 2 — columna `es_cotizacion` (marca notas de precio)
-- ================================================================
-- Notas tipo "Dólar hoy: a cuánto cotiza..." o "/cotizacion/..." describen el
-- precio; usarlas para explicar retornos sería circular. No se borran: se marcan.
-- El scraper setea esta columna de acá en adelante (función es_cotizacion()).
-- CORRER ESTA MIGRACIÓN antes/junto con el deploy del código nuevo (si falta la
-- columna, el scraper degrada y guarda sin la marca, pero conviene tenerla).

ALTER TABLE noticias ADD COLUMN IF NOT EXISTS es_cotizacion BOOLEAN DEFAULT FALSE;

-- Backfill de las filas ya existentes, por patrones de título/URL.
-- (Mismos patrones que la función es_cotizacion() de scraper_noticias.py.)
UPDATE noticias SET es_cotizacion = TRUE
WHERE es_cotizacion IS NOT TRUE AND (
       titulo ILIKE '%a cuánto cotiza%'  OR titulo ILIKE '%a cuánto cotizó%'
    OR titulo ILIKE '%a cuánto abre%'    OR titulo ILIKE '%a cuánto abrió%'
    OR titulo ILIKE '%a cuánto cierra%'  OR titulo ILIKE '%a cuánto cerró%'
    OR titulo ILIKE '%a cuánto opera%'   OR titulo ILIKE '%a cuánto operó%'
    OR link   ILIKE '%/cotizacion/%'
);
