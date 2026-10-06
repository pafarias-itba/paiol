-- ================================================================
-- Tabla de NOTICIAS individuales (corpus para la tesis)
-- Correr una sola vez en el SQL Editor de Supabase.
--
-- Guarda cada noticia scrapeada (no el agregado diario), para:
--   H1 - muestra etiquetada de titulares (titulo, jerga_local, etiqueta_manual)
--   H2/H3 - asociación noticia->empresa->retorno (empresas, fecha_publicacion)
-- ================================================================

CREATE TABLE IF NOT EXISTS noticias (
    id BIGSERIAL PRIMARY KEY,

    -- identificación / dedup
    link TEXT UNIQUE NOT NULL,              -- URL de la noticia (clave de deduplicación)
    titulo TEXT NOT NULL,
    resumen TEXT,
    medio VARCHAR(40) NOT NULL,             -- fuente RSS (ambito_finanzas, cronista, ...)

    -- tiempos
    fecha_publicacion TIMESTAMPTZ,          -- fecha/hora propia de la noticia (control: horario)
    fecha_scrape DATE NOT NULL,             -- día en que el scraper la capturó

    -- clasificación para la tesis
    empresas JSONB DEFAULT '[]',            -- empresas mencionadas, ej. ["GGAL","YPFD"]  (H2/H3)
    jerga_local BOOLEAN DEFAULT FALSE,      -- titular con jerga económica local (cepo/brecha/CCL)  (H1)
    sentimiento_gemini FLOAT,               -- sentimiento del titular según Gemini, -1..1  (H1)
    etiqueta_manual INTEGER,                -- etiqueta a mano: -1 / 0 / 1 (NULL = sin etiquetar)  (H1)

    -- auditoría
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Índices para las queries del análisis
CREATE INDEX IF NOT EXISTS idx_noticias_fecha_pub ON noticias(fecha_publicacion);
CREATE INDEX IF NOT EXISTS idx_noticias_scrape    ON noticias(fecha_scrape);
CREATE INDEX IF NOT EXISTS idx_noticias_empresas  ON noticias USING GIN(empresas);
CREATE INDEX IF NOT EXISTS idx_noticias_sinetiq   ON noticias(etiqueta_manual) WHERE etiqueta_manual IS NULL;
