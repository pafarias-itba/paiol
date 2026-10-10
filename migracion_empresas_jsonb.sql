-- ================================================================
-- Migración 1 — corregir `empresas` doble-codificada (string -> array JSONB)
-- ================================================================
-- El scraper hacía json.dumps(detectar_empresas(...)) antes de insertar en la
-- columna JSONB, así que quedaba el TEXTO '["GGAL"]' en vez del ARRAY ["GGAL"].
-- Consecuencia: empresas <> '[]'::jsonb daba true para todas las filas, y
-- jsonb_array_elements_text fallaba con "cannot extract elements from a scalar".
--
-- NOTA: estas 425 filas YA se migraron vía Python el 2026-10-10. Este archivo
-- queda como registro y es IDEMPOTENTE: si no quedan filas string, no toca nada.
-- Seguro de correr de nuevo.

UPDATE noticias
SET empresas = (empresas #>> '{}')::jsonb
WHERE jsonb_typeof(empresas) = 'string';
