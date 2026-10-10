-- ================================================================
-- Migración 4 — columna `es_patrocinado` (contenido pagado, no periodístico)
-- ================================================================
-- Ej.: cronista "brand-strategy" son notas pagadas por las empresas; usarlas para
-- medir el efecto del tono de la prensa sería un sesgo. No se borran: se marcan.
-- Según GDELT 2024 el único caso real en estos medios es cronista/brand-strategy;
-- el resto de los patrones queda como red defensiva.

ALTER TABLE noticias ADD COLUMN IF NOT EXISTS es_patrocinado BOOLEAN DEFAULT FALSE;

-- Backfill por la sección en la URL (mismos patrones que es_patrocinado() en Python).
UPDATE noticias SET es_patrocinado = TRUE
WHERE es_patrocinado IS NOT TRUE AND (
       link ILIKE '%/brand-strategy/%'
    OR link ILIKE '%/contenido-patrocinado/%'
    OR link ILIKE '%/branded/%'
    OR link ILIKE '%/publinota/%'
    OR link ILIKE '%/espacio-no-editorial/%'
    OR link ILIKE '%/sponsor/%'
    OR link ILIKE '%/publicidad/%'
);
