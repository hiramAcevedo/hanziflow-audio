"""Configuración global del pipeline de audio.

Ajusta voces, pausas y rutas aquí. Los scripts consumen este módulo.
"""

from pathlib import Path

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
HSK_DB = ROOT.parent / "HSK-word-list" / "hanzi-flow-hsk" / "output" / "hsk_vocabulary.db"
TRANSLATIONS_DIR = ROOT / "translations"
SENTENCES_DIR = ROOT / "sentences"
CACHE_DIR = ROOT / "cache"
OUTPUT_DIR = ROOT / "output"

# -----------------------------------------------------------------------------
# Edge-TTS voices (Chinese — masculinas adultas, imitables)
#
# Las claves son etiquetas de producto (cómo se perciben al oído). El valor
# es el voice-id real de edge-tts. Si cambias un valor, el cache de ese alias
# queda invalidado; si solo renombras la clave, basta con renombrar la carpeta
# correspondiente en cache/<scope>/<alias>/.
# -----------------------------------------------------------------------------
CN_VOICES = {
    "corta":   "zh-CN-YunyangNeural",   # seca, profesional tipo noticiero
    "larga":   "zh-CN-YunxiNeural",     # aguda, marca bien los tonos
    "neutral": "zh-CN-YunjianNeural",   # grave, deportivo/narrativo
}

# Default para v2 (inmersivo) — empezamos con "larga" por claridad de tonos
CN_VOICE_DEFAULT = "larga"

# Spanish voice (v1 bilingüe)
ES_VOICE = "es-MX-JorgeNeural"

# -----------------------------------------------------------------------------
# Speech rates (edge-tts format: +0%, -10%, +15%, etc.)
# -----------------------------------------------------------------------------
# Chino más lento al inicio, luego sube a normal cuando ya domines
CN_RATE = "-15%"   # un poco más lento para claridad de tonos
ES_RATE = "+0%"

# -----------------------------------------------------------------------------
# Silencios entre segmentos (en milisegundos)
# -----------------------------------------------------------------------------
# v1: CN → pausa → ES → pausa entre entradas
V1_PAUSE_AFTER_CN = 900
V1_PAUSE_AFTER_ES = 700
V1_PAUSE_BETWEEN_ENTRIES = 1200

# v2: CN → pausa MENTAL → CN repetido → pausa entre entradas
V2_PAUSE_AFTER_FIRST_CN = 2500   # te da tiempo de pensar significado
V2_PAUSE_AFTER_SECOND_CN = 600
V2_PAUSE_BETWEEN_ENTRIES = 1500

# v3: CN → pausa → ES → pausa → oración CN (requiere banco de oraciones)
V3_PAUSE_AFTER_CN = 900
V3_PAUSE_AFTER_ES = 700
V3_PAUSE_AFTER_SENTENCE = 1500

# -----------------------------------------------------------------------------
# Scopes de generación (qué niveles se generan y en qué orden)
# -----------------------------------------------------------------------------
SCOPES = [
    # (nombre_salida, hsk_version, level)
    ("hsk2.0_l1", "2.0", 1),
    ("hsk2.0_l2", "2.0", 2),
    ("hsk3.0_l1", "3.0", 1),
    ("hsk3.0_l2", "3.0", 2),
]
