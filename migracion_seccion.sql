-- ================================================================
-- Migración 3 — columna `seccion` (para medir qué secciones aportan)
-- ================================================================
-- Hasta ahora `medio` guardaba la clave del feed (ej. "ambito_negocios"), que mezcla
-- medio + sección. Se separa: `medio` = ambito, `seccion` = negocios.
-- El scraper nuevo ya los guarda separados; esto adapta las filas viejas.

ALTER TABLE noticias ADD COLUMN IF NOT EXISTS seccion VARCHAR(40);

-- Backfill: parte el feed_key viejo (guardado en `medio`) en medio + seccion.
-- Casos especiales: "cronista" (feed general) y "iprofesional" (feed de economía).
UPDATE noticias SET
    seccion = CASE
        WHEN medio = 'cronista'     THEN 'general'
        WHEN medio = 'iprofesional' THEN 'economia'
        WHEN position('_' in medio) > 0 THEN substr(medio, position('_' in medio) + 1)
        ELSE 'general'
    END,
    medio = CASE
        WHEN position('_' in medio) > 0 THEN substr(medio, 1, position('_' in medio) - 1)
        ELSE medio
    END
WHERE seccion IS NULL;

CREATE INDEX IF NOT EXISTS idx_noticias_seccion ON noticias(medio, seccion);
