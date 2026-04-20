# hanziflow-audio

Pipeline de generación de audio para estudio pasivo del corpus HanziFlow.
Convierte el vocabulario HSK (2.0 L1–L2, 3.0 L1–L2) y módulos de estudio propios
en archivos MP3 compilados por nivel, para escuchar offline mientras corres,
conduces o caminas.

Sub-proyecto independiente dentro del workspace `hanziflow/`.

## Arquitectura

```
hanziflow-audio/
├── config.py                 # voces, tasas, silencios, paths, SCOPES
├── sources.py                # lee HSK DB + traducciones ES + oraciones
├── benchmarks/               # evaluador de LLMs locales (ver benchmarks/README.md)
├── generate_edge.py          # TTS batch con edge-tts (Azure, gratis) — Modos 1/2
├── generate_sentences.py     # genera oraciones ZH vía Ministral local (Modo 3)
├── annotate_pinyin.py        # pypinyin + HSK DB + overrides (Modo 3) — TODO
├── compile.py                # compila MP3s finales por scope × voz × modo
├── translations/             # overlay ES por scope — hsk2.0_l1.json etc.
├── sentences/                # banco de oraciones ZH+ES por scope — hsk2.0_l1.md
├── cache/                    # TTS individuales, .gitignored
└── output/edge/              # MP3s finales por modo, .gitignored
    ├── mode1/                # palabra_ZH + palabra_ES
    ├── mode2/                # solo palabra_ZH
    └── mode3/                # palabra + oración, con 3 Levels internos
        ├── L1_principiante/
        ├── L2_medio/
        └── L3_alto/
```

## Matriz de Modos

Los 3 Modos son **paralelos**, no versiones sucesivas. Cada lista HSK
cubierta tendrá los 3. Modo 3 tiene 3 sub-niveles internos para acompañar
al estudiante conforme avanza.

| Modo | Contenido del audio | Texto asociado |
|---|---|---|
| **1** | palabra_ZH + palabra_ES | 1 versión con pinyin |
| **2** | solo palabra_ZH (pausa mental, repetición) | 1 versión con pinyin |
| **3 — L1 Principiante** | palabra_ZH + palabra_ES + oración_ZH + oración_ES | 1 versión con pinyin |
| **3 — L2 Medio** | palabra_ZH + oración_ZH + español | 2 versiones: con y sin pinyin |
| **3 — L3 Alto** | palabra_ZH + oración_ZH | 2 versiones: con y sin pinyin |

Por lista HSK: 5 audios + 7 textos. Para los 4 scopes del examen (HSK 2.0
L1–L2, HSK 3.0 L1–L2) son **20 audios + 28 textos totales**.

## Reglas duras del pipeline

1. **Oraciones → LLM.** Única pieza probabilística. Modelos de producción:
   Ministral 3B local (91% benchmark) para iteración rápida; Gemini 2.5
   Flash vía API para corpus grande (~$0.03 por los 4 scopes). El script
   `generate_sentences.py` soporta ambos endpoints con los mismos flags
   (`--top-k 0 --presence-penalty 0` apaga kwargs que Gemini rechaza).
2. **Pinyin → HSK DB v2 + pypinyin + overrides manuales.** NUNCA del LLM.
   Los 4 modelos locales fallaron fidelidad pinyin incluso en 不客气. El
   pipeline Python es determinista. Pero atención: la **HSK DB v1 (la que
   está en disco hoy) tiene bug sistémico** — `complete/hsk` ganó prioridad
   sobre MandarinBean y metió lecturas arcaicas en 22/85 single-chars de
   HSK 2.0 L1 (26%). En multi-char la DB sí está bien (preserva tonos
   neutros: 妈妈 `mā ma`, 谢谢 `xiè xie`). La DB v2 invierte prioridades
   por campo: MandarinBean gana `pinyin`, CC-CEDICT mantiene
   `radical/pos/frequency_rank`. **Hasta que v2 no exista, no generar más
   headers de oraciones.** Plan operativo en
   [`../HSK-word-list/hanzi-flow-hsk/HANDOFF_DB_V2.md`](../HSK-word-list/hanzi-flow-hsk/HANDOFF_DB_V2.md).
3. **Traducciones → LLM + revisión manual.** Se cachea en
   `translations/<scope>.json` y se versiona. Errores de español
   (ser/estar, concordancia, pérdida de sujeto) se limpian post-hoc con
   Gemini Pro en la GUI si es necesario.
4. **Voces → edge-tts.** Offline gratis tras generar los MP3; es el
   compromiso más eficiente local-first que tenemos sin mlx-audio estable.
   **Nota:** edge-tts sintetiza desde hanzi crudo, no pinyin, así que los
   MP3 ya generados **no** están corruptos por el bug de la DB v1. Solo
   los archivos de texto con pinyin (`sentences/*.md`) requieren
   regeneración.

## Voces (edge-tts, zh-CN)

Todas masculinas adultas (imitación de tonos más accesible que las femeninas
agudas). Los alias son **etiquetas de producto** por cómo se perciben al oído;
el voice-id real de edge-tts queda a nivel de `config.py` para no confundir
durante el estudio.

| Alias     | Voz edge-tts real     | Carácter al oído                                |
|-----------|-----------------------|--------------------------------------------------|
| `corta`   | zh-CN-YunyangNeural   | seca, profesional tipo noticiero — frases breves |
| `larga`   | zh-CN-YunxiNeural     | aguda, marca bien los tonos — **punto de partida** para HSK 2 |
| `neutral` | zh-CN-YunjianNeural   | grave, deportivo/narrativo — descanso de oído    |

Default v2 (inmersivo): `larga`.

Español: `es-MX-JorgeNeural` (masculina MX, clara).

Configurable en `config.py`. Si renombras un alias **sin** cambiar su voice-id
real, basta con renombrar también la carpeta correspondiente en
`cache/<scope>/<alias>/` — los clips ya generados siguen siendo válidos.

## Setup

```bash
cd hanziflow-audio
uv venv .venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements.txt
```

Requiere `ffmpeg` disponible en PATH (para `pydub`):
```bash
brew install ffmpeg
```

## Uso

### 1. Validar fuentes
```bash
python sources.py
```
Imprime cuántas entradas hay por scope y cuántas traducciones ES / oraciones
están disponibles.

### 2. Generar TTS individual (cache)
```bash
# Una muestra rápida: HSK 2.0 L1 con una sola voz, modo bilingüe
python generate_edge.py --scope hsk2.0_l1 --voice larga --mode v1

# Comparar las tres voces CN en el mismo scope
python generate_edge.py --scope hsk2.0_l1 --voice all --mode v1

# Todo el corpus, todas las voces, todos los modos
python generate_edge.py --scope all --voice all --mode all
```

El cache se respeta: una segunda corrida solo genera lo nuevo.

### 3. Compilar MP3s finales
```bash
# Un solo MP3 compilado
python compile.py --scope hsk2.0_l1 --voice larga --mode v1

# Todos los modos para una voz
python compile.py --scope hsk2.0_l1 --voice larga --mode all

# Todo
python compile.py --scope all --voice all --mode all
```

Los MP3s finales van a `output/edge/<modo>/<scope>_<voz>_<modo>.mp3`.

### 4. Generar oraciones para Modo 3
Requiere LM Studio corriendo local con `mistralai/ministral-3b` cargado.
Qwen3.5 9B se descartó tras benchmark: ignora `enable_thinking=false` y
cae en loops de reasoning.

```bash
# Defaults: LM Studio en :1234 con ministral-3b
python generate_sentences.py --scope hsk2.0_l1

# Los 4 scopes de un tirón
python generate_sentences.py --scope all
```

El script es idempotente: si ya hay oraciones en `sentences/<scope>.md`, solo
genera las faltantes. Batch de 20 palabras por request (sweet spot de
Ministral según el benchmark BATCH_CAPACITY).

## Ampliación

Para agregar un nuevo scope (ej. tus módulos propios 03, 04):

1. Extender `SCOPES` en `config.py` con una fuente alternativa (no HSK DB).
2. Adaptar `sources.load_hsk_level` para leer de tu fuente — o crear un
   `sources.load_module(...)` paralelo.
3. Poner traducciones en `translations/<scope>.json`.
4. (Opcional) Poner oraciones en `sentences/<scope>.md`.
5. Correr `generate_edge.py` y `compile.py` con el nuevo scope.

## Distribución al teléfono

Los MP3s en `output/edge/` son autocontenidos. AirDrop al iPhone → guardar en
la app de Archivos o importar a Música. Offline total, sin dependencia del Mac.
