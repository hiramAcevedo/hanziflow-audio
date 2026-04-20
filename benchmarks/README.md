# Benchmarks de LLMs locales

Panel de pruebas para decidir qué modelo local usar como generador de corpus
(oraciones HSK, traducciones ES, relleno de tablas NSM). Mide 7 dimensiones
críticas para el proyecto y emite un scorecard comparativo.

## Las 7 pruebas

| # | Nombre | Qué mide | Por qué importa |
|---|---|---|---|
| 1A | **FIDELITY-JSON** | Pinyin correcto + trampa de alucinación, output en JSON anidado | Formato estándar de la API; el más común en prod |
| 1B | **FIDELITY-CSV** | Mismo contenido en CSV plano con encabezado | Algunos modelos fallan con anidamiento pero no con tabular |
| 1C | **FIDELITY-YAML** | Mismo contenido en YAML indentado | Mide tolerancia a formatos sensibles a whitespace |
| 2  | **RULE_ADHERENCE** | 5 líneas `hanzi\|pinyin\|es` sin markdown ni numeración | Parseo del output debe ser determinista |
| 3  | **SENTENCE_GEN** | 10 oraciones HSK 1, cada una contiene la palabra, longitud adecuada | Lo que necesitamos para `v3` del audio |
| 4  | **BATCH_CAPACITY** | 50 oraciones de golpe — coverage + no invención | Define el tamaño óptimo de batch para producción |
| 5  | **TABLE_FILL** | Completa 2 filas NSM con 13 columnas alineadas | Lo que se usa al expandir módulos 03/04 |

Las tres variantes de FIDELITY (1A/1B/1C) piden exactamente los mismos datos en
formatos distintos. Comparar sus scores revela si un modelo es malo con JSON
anidado pero bueno con CSV, por ejemplo — información útil para elegir el
formato de interchange del pipeline.

Cada prueba da 0–100%. El total es el promedio simple.

## Uso

Por default corre **3 intentos por prueba** y detecta colapsos catastróficos
(output vacío, loops de token, ausencia de CJK, dominancia patológica de un
carácter). Así medimos tanto habilidad media como probabilidad de basura.

### Workflow (LM Studio, un modelo a la vez)

Con 16 GB de RAM LM Studio tiene 1 modelo cargado. Flujo: cargar → bench →
descargar → siguiente.

```bash
cd hanziflow-audio
source .venv/bin/activate

# 1) En LM Studio: cargar qwen3.5-9b → Start Server (puerto 1234 por default)
python benchmarks/benchmark.py --model qwen3.5-9b --runs 3

# 2) Descargar, cargar glm-4.6v-flash, Start Server
python benchmarks/benchmark.py --model glm-4.6v-flash --runs 3

# 3) Descargar, cargar ministral-3-14b-reasoning-2512, Start Server
python benchmarks/benchmark.py --model ministral-3-14b-reasoning-2512 --runs 3

# 4) Descargar, cargar gemma-4-e4b, Start Server
python benchmarks/benchmark.py --model gemma-4-e4b --runs 3
```

Cada corrida escribe `benchmarks/results_YYYYMMDD_HHMM.md`. Al final tienes 4
reportes — compara tablas finales manualmente o pasa los 4 a Claude para
síntesis.

### Opciones

**Core:**
- `--runs N` — corridas por test. Default **3** (sweet spot entre ruido y
  costo). Con 1 no detectas inestabilidad. Con 5+ más robusto pero duplica
  tiempo.
- `--base-url URL` — default `http://localhost:1234/v1` (LM Studio). Ollama:
  `http://localhost:11434/v1`.
- `--api-key KEY` — ignorado por LM Studio, cualquier string sirve.
- `--max-tokens N` — cap de tokens de salida (default **4000**). Con
  `--no-think` 4000 sobra. Sin `--no-think` en modelos con reasoning puede
  requerir 8000+.

**Sampling (defaults oficiales de Qwen3 non-thinking, aplican a todos los tests):**
- `--temperature` — default **0.7**. Qwen documenta que greedy decoding
  (`temp=0`) causa repeticiones infinitas en modo thinking. No bajes a 0.
- `--top-p` — default **0.8**.
- `--top-k` — default **20**. Se pasa via `extra_body` (no es standard OpenAI).
- `--presence-penalty` — default **1.0**. Combate repetición degenerada sobre
  el vocabulario ya emitido. Rango útil 0.5–1.5.
- `--no-think` — flag. Inyecta `chat_template_kwargs.enable_thinking=False`.
  Para Qwen3/3.5 salta el chain-of-thought entero. Ignorado por modelos sin
  thinking (Ministral, Gemma, GLM antiguo). Recomendado para tareas de
  corpus: queremos datos, no razonamiento.

**Ejemplo comparativo (Qwen con y sin thinking):**

```bash
python benchmarks/benchmark.py --model qwen3.5-9b --runs 3
python benchmarks/benchmark.py --model qwen3.5-9b --runs 3 --no-think
```

Al comparar los dos reportes se ve cuánto aporta (o resta) el reasoning para
esta tarea específica.

### Qué reporta

- **Tabla resumen:** media de N corridas por test + conteo global de colapsos.
- **Tabla peor caso:** mínimo score por test. Crítico para batch: un modelo
  con media 90% y peor 0% (colapso) es más peligroso que uno con media 80%
  estable, porque en 1000 llamadas te contamina cientos.
- **Detalle por corrida:** cada intento con score, latencia, flag ⚠ de
  colapso y output crudo truncado para inspección manual.

### Detector de colapso — qué detecta

Un modelo puede funcionar 80% del tiempo y de repente vomitar basura. El
benchmark flaguea la corrida como INUTILIZABLE si:

1. Output vacío o demasiado corto (<15–30 chars).
2. Repetición degenerada: mismo carácter >30 veces seguidas (`aaaaaaaa...`).
3. Loop de token: secuencia corta repetida >10 veces.
4. Ausencia total de CJK en tests que lo requieren.
5. Dominancia patológica: un solo carácter es >50% del output.

Las corridas flagueadas cuentan como score=0 en la media, pero se reportan
aparte como tasa de colapso.

## Recomendación de uso posterior

1. Corre los 4 modelos una vez.
2. Elige el de mayor score en `SENTENCE_GEN` + `BATCH_CAPACITY` para producción.
3. Úsalo como `--model` en `generate_sentences.py`.
4. Si el score en `BATCH_CAPACITY` baja bajo 80% con 50, baja el batch a 20 o 30.

## Tamaños recomendados de batch según score BATCH_CAPACITY

| Score | Batch sugerido | Justificación |
|---|---|---|
| ≥ 95% | 50 | Modelo puede con lotes grandes sin degradar |
| 80–94% | 30 | Sweet spot para modelos 7–14B |
| 60–79% | 20 | Aceptable, más overhead |
| < 60% | 10 | El modelo se atraganta — subdivide más |

## Reproducibilidad

- Ground truth en `benchmark.py` (dict `FIDELITY_FACTS`) — no cambiar sin anotar.
- Las 50 palabras de `BATCH_CAPACITY` están literales en el script (no
  dependen de la HSK DB para evitar confusiones al reproducir).
- Sampling: `temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.0`.
  No usamos greedy decoding (`temp=0`) — Qwen documenta que causa
  repeticiones infinitas en modo thinking. La varianza run-to-run es costo
  aceptable para evitar ese modo patológico; por eso corremos 3 intentos y
  reportamos media + peor caso.
- El ground truth de pinyin acepta múltiples formas válidas cuando aplican
  reglas de sandhi (ej. 不客气 acepta tanto `bù kèqi` ortográfico como
  `bú kèqi` fonético). Ver `FIDELITY_FACTS` para detalle.
