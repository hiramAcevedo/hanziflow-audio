"""Panel de pruebas para modelos LLM locales destinados a generar corpus HSK.

Corre 7 pruebas contra uno o varios endpoints OpenAI-compatibles (LM Studio,
Ollama, API externa) y emite un scorecard legible. Cada prueba mide una
dimensión crítica para nuestro pipeline:

    1A. FIDELITY-JSON    — datos sin alucinar en JSON anidado
    1B. FIDELITY-CSV     — mismos datos en CSV plano (adherencia a formato tabular)
    1C. FIDELITY-YAML    — mismos datos en YAML (adherencia a formato indentado)
    2.  RULE_ADHERENCE   — cumple formato pipe estricto sin preámbulo
    3.  SENTENCE_GEN     — oraciones HSK 1 en formato esperado
    4.  BATCH_CAPACITY   — aguanta 50 items de golpe sin olvidar ni inventar
    5.  TABLE_FILL       — completa fila de tabla NSM manteniendo pipes y columnas

Sampling: usa los defaults oficiales de Qwen3 (non-thinking mode), que NO son
greedy. Qwen documenta que greedy decoding + thinking mode produce repeticiones
infinitas. Con --no-think, el modelo salta el chain-of-thought (deseable para
corpus: queremos datos, no razonamiento).

Uso:
    # Un solo modelo, config recomendada para Qwen3
    python benchmark.py --base-url http://localhost:1234/v1 --model qwen3.5-9b --no-think

    # Varios modelos comparativos (carga secuencial en LM Studio)
    python benchmark.py --base-url http://localhost:1234/v1 \\
        --models qwen3.5-9b,glm-4.6v-flash,ministral-3-14b,gemma-4-e4b \\
        --report benchmarks/results_$(date +%Y%m%d).md

El reporte se guarda en markdown con puntuación por prueba y latencia.
"""

import argparse
import json
import os
import re
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List

from openai import OpenAI


# =============================================================================
# DETECCIÓN DE COLAPSO CATASTRÓFICO
# =============================================================================
# Un modelo puede funcionar 80% del tiempo y de repente vomitar basura
# ("00000aaaa000a00amnnaa", output vacío, loop infinito de un token).
# Estas fallas son peligrosas para un pipeline batch: una sola corrida fallida
# contamina cientos de entradas. Detectamos 4 patrones de colapso.

def detect_collapse(raw: str, test_name: str) -> tuple[bool, str]:
    """Devuelve (colapsó, razón). True = corrida INUTILIZABLE para producción."""
    if raw is None or not raw.strip():
        return True, "output vacío"

    # Longitud mínima razonable por test
    min_len = 30 if test_name.startswith("4-") else 15
    if len(raw.strip()) < min_len:
        return True, f"output demasiado corto ({len(raw.strip())} chars)"

    # Repetición degenerada: un solo carácter >30 veces consecutivas
    if re.search(r"(.)\1{30,}", raw):
        return True, "repetición degenerada (mismo char >30 veces seguidas)"

    # Loop de token: una secuencia corta repetida >10 veces seguidas
    if re.search(r"(.{2,20}?)\1{10,}", raw):
        return True, "loop de token (secuencia corta repetida)"

    # Tests que DEBEN tener caracteres chinos — si no hay ninguno, colapsó
    needs_cjk = test_name.startswith(("1-", "3-", "4-", "5-"))
    if needs_cjk:
        cjk_count = sum(1 for ch in raw if "\u4e00" <= ch <= "\u9fff")
        if cjk_count == 0:
            return True, "sin ningún carácter chino en output"
        if cjk_count < 3 and test_name.startswith(("3-", "4-")):
            return True, f"casi sin chino ({cjk_count} chars CJK)"

    # Dominancia de un solo token: >50% del output es el mismo carácter
    if len(raw) > 50:
        from collections import Counter
        top_char, top_n = Counter(raw).most_common(1)[0]
        if top_n / len(raw) > 0.5 and top_char not in " \n|":
            return True, f"dominancia patológica de '{top_char}' ({top_n}/{len(raw)})"

    return False, ""


# =============================================================================
# GROUND TRUTH — datos verificados (jamás pasar al modelo en el prompt)
# =============================================================================

# De hsk_vocabulary.db, HSK 2.0 L1.
# Nota: cuando una palabra tiene múltiples pronunciaciones aceptadas (tono
# sandhi, variantes dialectales), se listan como set {forma1, forma2}. El
# scorer acepta cualquiera. Ejemplo: 不客气 fonéticamente es "bú kèqi" (tono
# sandhi de 不 antes de 4º tono); ortográficamente se escribe "bù kèqi". Ambas
# son correctas y los modelos buenos las producen indistintamente.
FIDELITY_FACTS = {
    "明天":   {"pinyin": {"míngtiān"},          "meaning_es": "mañana"},
    "学校":   {"pinyin": {"xuéxiào"},           "meaning_es": "escuela"},
    "电脑":   {"pinyin": {"diànnǎo"},           "meaning_es": "computadora"},
    "不客气": {"pinyin": {"bù kèqi", "bú kèqi"}, "meaning_es": "de nada"},
    "飞机":   {"pinyin": {"fēijī"},             "meaning_es": "avión"},
}

# Palabra inventada — el modelo NO debe fingir que la conoce
HALLUCINATION_TRAP = "葚蕤祺"  # no es una palabra real; si dice algo confiado, alucinó

HSK1_SAMPLE_FOR_SENTENCES = [
    ("我", "wǒ"), ("吃", "chī"), ("水", "shuǐ"), ("学校", "xuéxiào"),
    ("朋友", "péngyou"), ("喜欢", "xǐhuan"), ("下雨", "xià yǔ"), ("飞机", "fēijī"),
    ("电脑", "diànnǎo"), ("谢谢", "xièxie"),
]


# =============================================================================
# PRUEBAS
# =============================================================================

@dataclass
class RunResult:
    """Una sola corrida de un test."""
    score: float          # 0.0 - 1.0 (0 si colapsó)
    details: str
    latency_s: float
    collapsed: bool = False
    collapse_reason: str = ""
    raw: str = ""
    completion_tokens: int = 0  # tokens producidos (incluye thinking si lo hay)
    finish_reason: str = ""     # "stop" | "length" | "content_filter" | otro


@dataclass
class TestResult:
    """Agregado de N corridas de un test."""
    name: str
    runs: List[RunResult] = field(default_factory=list)

    @property
    def score_mean(self) -> float:
        return statistics.mean(r.score for r in self.runs) if self.runs else 0.0

    @property
    def score_std(self) -> float:
        if len(self.runs) < 2:
            return 0.0
        return statistics.stdev(r.score for r in self.runs)

    @property
    def score_worst(self) -> float:
        return min((r.score for r in self.runs), default=0.0)

    @property
    def collapse_rate(self) -> float:
        if not self.runs:
            return 0.0
        return sum(1 for r in self.runs if r.collapsed) / len(self.runs)

    @property
    def latency_mean(self) -> float:
        return statistics.mean(r.latency_s for r in self.runs) if self.runs else 0.0


@dataclass
class SamplingConfig:
    """Parámetros de sampling aplicados a TODAS las pruebas.

    Defaults basados en la documentación oficial de Qwen3 (non-thinking mode).
    Cambiar solo si sabes lo que haces — greedy decoding está DESACONSEJADO
    explícitamente por los autores para modelos con thinking.
    """
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    presence_penalty: float = 1.0   # combate repetición degenerada
    no_think: bool = False          # si True, inyecta enable_thinking=False

    def api_kwargs(self) -> dict:
        """Devuelve los kwargs listos para pasar a chat.completions.create."""
        kw = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "presence_penalty": self.presence_penalty,
            # top_k no es estándar OpenAI; va por extra_body
            "extra_body": {"top_k": self.top_k},
        }
        if self.no_think:
            # Qwen3/3.5 — switch oficial. Ignorado por modelos sin thinking (Ministral, Gemma).
            kw["extra_body"]["chat_template_kwargs"] = {"enable_thinking": False}
        return kw


@dataclass
class Test:
    name: str
    system: str
    user: str
    scorer: Callable[[str], tuple[float, str]]


# --------- Test 1A / 1B / 1C: FIDELITY en 3 formatos ---------
# El MISMO contenido se pide en JSON, CSV y YAML. Mide dos cosas:
#  (a) fidelidad del dato (pinyin + traducción correctos, trampa de alucinación)
#  (b) adherencia a formato estructurado
# Compararlos directamente revela si un modelo es bueno con el dato pero
# malo con estructuras anidadas (o viceversa).

FIDELITY_WORDS = ["明天", "学校", "电脑", "不客气", "飞机", "葚蕤祺"]

T1_SYSTEM = "Devuelve solo lo pedido. Si no sabes un dato, escribe null."


# ----- 1A: JSON -----
T1_JSON_USER = """Para cada palabra dame pinyin (con acentos) y traducción al español.
Si no conoces la palabra, usa null.

Palabras: """ + ", ".join(FIDELITY_WORDS) + """

Formato JSON:
{"明天": {"pinyin": "...", "es": "..."}, ...}"""


# ----- 1B: CSV -----
T1_CSV_USER = """Para cada palabra dame pinyin (con acentos) y traducción al español.
Si no conoces la palabra, escribe NULL en los dos campos.

Palabras: """ + ", ".join(FIDELITY_WORDS) + """

Formato CSV con encabezado:
hanzi,pinyin,es
明天,míngtiān,mañana
..."""


# ----- 1C: YAML -----
T1_YAML_USER = """Para cada palabra dame pinyin (con acentos) y traducción al español.
Si no conoces la palabra, usa null.

Palabras: """ + ", ".join(FIDELITY_WORDS) + """

Formato YAML:
明天:
  pinyin: míngtiān
  es: mañana
..."""


def _check_fidelity_data(data: dict) -> tuple[float, str]:
    """Dado un dict ya parseado {hanzi: {pinyin, es}}, puntúa fidelidad.
    Compartido por los 3 scorers de formato."""
    hits = 0
    total = 0
    notes = []
    for word, truth in FIDELITY_FACTS.items():
        total += 1
        entry = data.get(word)
        if not isinstance(entry, dict):
            notes.append(f"{word}: falta")
            continue
        pinyin_got = str(entry.get("pinyin") or "").strip().lower()
        pinyin_got_norm = pinyin_got.replace(" ", "")
        valid_forms = {p.lower().replace(" ", "") for p in truth["pinyin"]}
        if pinyin_got_norm in valid_forms:
            hits += 1
        else:
            expected = "|".join(sorted(truth["pinyin"]))
            notes.append(f"{word}: pinyin='{pinyin_got}' (esperado: {expected})")

    # Trampa de alucinación — aceptamos null, None, "null", "NULL", cadena vacía
    total += 1
    trap = data.get(HALLUCINATION_TRAP)
    is_null_like = (
        trap is None
        or (isinstance(trap, dict) and (trap.get("pinyin") in (None, "", "null", "NULL")))
        or (isinstance(trap, str) and trap.strip().lower() in ("", "null", "none"))
    )
    if is_null_like:
        hits += 1
    else:
        notes.append(f"ALUCINACIÓN: inventó dato para {HALLUCINATION_TRAP} → {trap}")

    details = "; ".join(notes) if notes else "todo correcto"
    return hits / total, details


def score_fidelity_json(raw: str) -> tuple[float, str]:
    # Tolerar ```json ... ``` y texto alrededor
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return 0.0, "no JSON detectado"
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        return 0.0, f"JSON inválido: {e}"
    if not isinstance(data, dict):
        return 0.0, "JSON no es objeto"
    return _check_fidelity_data(data)


def score_fidelity_csv(raw: str) -> tuple[float, str]:
    # Extraer bloque CSV (tolera code fences)
    text = re.sub(r"^```\w*\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return 0.0, "CSV vacío"

    # Aceptamos header presente o ausente
    if lines[0].lower().replace(" ", "").startswith("hanzi,"):
        lines = lines[1:]
    if not lines:
        return 0.0, "CSV sin filas"

    data = {}
    malformed = 0
    for l in lines:
        # Split en comas (simple — no soportamos quoting complejo, no lo necesitamos)
        parts = [p.strip().strip('"').strip("'") for p in l.split(",")]
        if len(parts) < 3:
            malformed += 1
            continue
        hanzi, pinyin, es = parts[0], parts[1], ",".join(parts[2:])  # por si "es" tiene coma
        # NULL → None
        py_val = None if pinyin.upper() in ("NULL", "NONE", "") else pinyin
        es_val = None if es.upper() in ("NULL", "NONE", "") else es
        data[hanzi] = {"pinyin": py_val, "es": es_val}

    score, details = _check_fidelity_data(data)
    if malformed:
        details = f"{malformed} fila(s) mal formadas; " + details
        score = max(0.0, score - 0.1 * malformed / max(len(lines), 1))
    return score, details


def score_fidelity_yaml(raw: str) -> tuple[float, str]:
    # Parser YAML mínimo suficiente para este esquema fijo (2 niveles, sin tipos complejos).
    # Evitamos dependencia de PyYAML para mantener el benchmark portable.
    text = re.sub(r"^```\w*\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
    lines = text.splitlines()
    if not lines:
        return 0.0, "YAML vacío"

    data: dict = {}
    current_key = None
    for raw_line in lines:
        line = raw_line.rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue

        # Línea sin indentación → nueva entrada de primer nivel
        if not line.startswith((" ", "\t")):
            m = re.match(r"^([^:]+):\s*(.*)$", line)
            if not m:
                continue
            key = m.group(1).strip().strip('"').strip("'")
            rest = m.group(2).strip()
            if rest and rest.lower() != "null":
                # "明天: null" o "明天: algo" — asumimos valor escalar (no esperado, pero lo manejamos)
                data[key] = rest if rest.lower() != "null" else None
                current_key = None
            else:
                # "明天:" → viene dict anidado o es null explícito
                data[key] = None if rest.lower() == "null" else {}
                current_key = key
        else:
            # Línea indentada → sub-campo de current_key
            if current_key is None or not isinstance(data.get(current_key), dict):
                continue
            m = re.match(r"^\s+([^:]+):\s*(.*)$", line)
            if not m:
                continue
            sub_key = m.group(1).strip().strip('"').strip("'")
            sub_val = m.group(2).strip().strip('"').strip("'")
            if sub_val.lower() == "null":
                sub_val_parsed = None
            else:
                sub_val_parsed = sub_val
            data[current_key][sub_key] = sub_val_parsed

    if not data:
        return 0.0, "YAML sin entradas parseables"
    return _check_fidelity_data(data)


# --------- Test 2: RULE_ADHERENCE ---------
T2_SYSTEM = "Devuelve solo lo pedido en el formato indicado, sin texto adicional."

T2_USER = """Dame 5 palabras HSK 1 en este formato EXACTO, una por línea:
<hanzi>|<pinyin>|<significado español>

Reglas estrictas:
- SOLO 5 líneas, sin numeración
- SIN texto antes ni después
- SIN markdown, SIN código, SIN bullets
- Pinyin con tonos (acentos), no numeración"""


def score_rule_adherence(raw: str) -> tuple[float, str]:
    stripped = raw.strip()
    lines = [l for l in stripped.splitlines() if l.strip()]
    notes = []
    score = 0.0

    # 1. Exactly 5 lines
    if len(lines) == 5:
        score += 0.25
    else:
        notes.append(f"líneas={len(lines)} (esperado 5)")

    # 2. Every line matches pattern hanzi|pinyin|es
    matches = sum(1 for l in lines if re.match(r"^[^\|]+\|[^\|]+\|[^\|]+$", l.strip()))
    if lines:
        score += 0.25 * (matches / len(lines))

    # 3. No markdown artifacts
    bad_markers = ["```", "**", "##", "- ", "1.", "2.", "json", "<", ">"]
    contamination = sum(1 for m in bad_markers if m in stripped)
    if contamination == 0:
        score += 0.25
    else:
        notes.append(f"markdown/basura detectado: {contamination} markers")

    # 4. No pinyin numeric (tones as digits)
    if not re.search(r"[a-zA-Z]\d", stripped):
        score += 0.25
    else:
        notes.append("pinyin con números en vez de acentos")

    if not notes:
        notes.append("todo ok")
    return score, "; ".join(notes)


# --------- Test 3: SENTENCE_GEN ---------
T3_SYSTEM = "Genera oraciones HSK en el formato pedido. Solo las oraciones, sin explicaciones."

T3_USER_TEMPLATE = """Genera una oración de ejemplo para cada palabra HSK 1.
Reglas:
- Nivel HSK 1 (solo usa palabras HSK 1 o más simples)
- Oración de 5 a 12 caracteres chinos
- Formato exacto, una por línea: <palabra>||<oración>
- Sin pinyin, sin traducción, sin numeración, sin explicaciones

Palabras:
{words}"""


def score_sentence_gen(raw: str) -> tuple[float, str]:
    lines = [l.strip() for l in raw.strip().splitlines() if "||" in l]
    expected_words = {w for w, _ in HSK1_SAMPLE_FOR_SENTENCES}
    got = {}
    for line in lines:
        m = re.match(r"^(\S+)\s*\|\|\s*(.+)$", line)
        if m:
            got[m.group(1)] = m.group(2).strip()

    notes = []
    score = 0.0

    # 1. Coverage (50% del score)
    coverage = len(got.keys() & expected_words) / len(expected_words)
    score += 0.50 * coverage
    missing = expected_words - got.keys()
    if missing:
        notes.append(f"faltan: {','.join(missing)}")

    # 2. Each sentence contains the target word (25%)
    contains = sum(1 for w, s in got.items() if w in expected_words and w in s)
    if got:
        score += 0.25 * (contains / max(len(got), 1))
    bad_contain = [w for w, s in got.items() if w in expected_words and w not in s]
    if bad_contain:
        notes.append(f"oración no contiene palabra: {','.join(bad_contain)}")

    # 3. Length constraint (25%)
    def hanzi_len(s: str) -> int:
        return sum(1 for ch in s if "\u4e00" <= ch <= "\u9fff")
    good_len = sum(
        1 for s in got.values() if 4 <= hanzi_len(s) <= 15
    )
    if got:
        score += 0.25 * (good_len / max(len(got), 1))

    if not notes:
        notes.append(f"{len(got)}/{len(expected_words)} ok")
    return score, "; ".join(notes)


def build_t3_prompt() -> str:
    words_block = "\n".join(
        f"- {w} ({p})" for w, p in HSK1_SAMPLE_FOR_SENTENCES
    )
    return T3_USER_TEMPLATE.format(words=words_block)


# --------- Test 4: BATCH_CAPACITY ---------
# Mide si el modelo responde correctamente con 50 items pedidos de golpe.
T4_SYSTEM = T3_SYSTEM

# Usamos las 50 primeras HSK 2.0 L1 por frecuencia. Lista literal para no depender de DB.
BATCH_WORDS = [
    "的","了","我","是","你","在","不","有","他","这","个","和","人","我们","说",
    "好","一","会","都","上","来","很","去","她","想","能","那","看","做","什么",
    "点","没","大","下","里","现在","年","多","吗","太","呢","小","时候","工作","家",
    "怎么","爱","喜欢","吃","谁",
]

T4_USER = f"""Genera una oración HSK 1 por cada palabra de la lista (50 palabras).
Reglas:
- Oraciones cortas (5–12 caracteres)
- Vocabulario HSK 1 o inferior
- Formato exacto: <palabra>||<oración>  — una por línea
- Sin numeración, sin preámbulo, sin explicaciones

Palabras:
{chr(10).join('- ' + w for w in BATCH_WORDS)}
"""


def score_batch_capacity(raw: str) -> tuple[float, str]:
    lines = [l.strip() for l in raw.strip().splitlines() if "||" in l]
    got = {}
    for line in lines:
        m = re.match(r"^(\S+)\s*\|\|\s*(.+)$", line)
        if m:
            got[m.group(1)] = m.group(2).strip()

    expected = set(BATCH_WORDS)
    hits = len(expected & got.keys())
    coverage = hits / len(expected)

    # Penaliza si inventa palabras que no estaban
    spurious = set(got.keys()) - expected
    penalty = min(0.2, 0.02 * len(spurious))

    containment = sum(1 for w in expected & got.keys() if w in got[w]) / max(len(expected), 1)
    score = max(0.0, 0.7 * coverage + 0.3 * containment - penalty)

    notes = [f"cobertura {hits}/{len(expected)}"]
    if spurious:
        notes.append(f"inventó {len(spurious)} palabras extra")
    missing = expected - got.keys()
    if len(missing) <= 8 and missing:
        notes.append(f"faltan: {','.join(sorted(missing))}")
    elif missing:
        notes.append(f"faltan {len(missing)} palabras")

    return score, "; ".join(notes)


# --------- Test 5: TABLE_FILL ---------
# Simula lo que haríamos con los módulos 03/04: pasar una fila NSM parcial
# y que complete las columnas faltantes manteniendo el esquema de pipes.

T5_SYSTEM = "Completa filas de tabla markdown respetando el esquema de pipes. Solo las filas pedidas."

T5_USER = """Completa las 2 filas faltantes de esta tabla siguiendo EXACTAMENTE el esquema.
Las columnas son: # | dx | 汉字 | pinyin | pinyin_num | es | fn | capa | HSK | freq | 中文 ejemplo | es ejemplo | nota

Reglas estrictas:
- Responde SOLO con las 2 filas pedidas, nada más
- Cada fila debe tener exactamente 13 columnas (14 pipes incluyendo bordes)
- No agregues encabezado, no agregues separadores, no numeres las filas tú
- `dx` puede ser "—"
- `fn` es la etiqueta NSM del primitivo

Filas existentes (contexto):
| 1 | ☑ | 我 | wǒ | wo3 | yo; me | P.I | 1 | 2.0=L1, 3.0=L1 | 3 | 我是学生。 | Yo soy estudiante. | Pronombre sujeto y objeto sin cambio. |
| 2 | ☑ | 你 | nǐ | ni3 | tú; te | P.YOU | 1 | 2.0=L1, 3.0=L1 | 5 | 你好！ | ¡Hola! | Cortés: 您 nín. |

Completa para estas palabras (son primitivos NSM de 3ra persona y de "gente"):
- 他  (ta1, P.HE/SHE, HSK 2.0=L1, freq~10)
- 人  (ren2, P.PEOPLE, HSK 2.0=L1, freq~19)

Numerarlas 3 y 4."""


def score_table_fill(raw: str) -> tuple[float, str]:
    lines = [l.strip() for l in raw.strip().splitlines() if l.strip().startswith("|")]
    notes = []
    score = 0.0

    # 1. Exactly 2 rows (25%)
    if len(lines) == 2:
        score += 0.25
    else:
        notes.append(f"filas={len(lines)} (esperado 2)")

    # 2. Each row has 13 data cells (24 pipes? no — 14 pipes for 13 cols bounded)
    #    Formato "| a | b | ... | m |" tiene 14 pipes.
    cols_ok = 0
    for l in lines:
        # Separar y contar celdas (ignorando pipes externos vacíos)
        cells = [c for c in l.split("|")]
        # Normalmente split produce: '', ' a ', ' b ', ..., ' m ', ''
        cell_count = len([c for c in cells if c.strip() != ""]) if cells[0].strip() == "" else len(cells)
        # Más simple: contar pipes
        pipes = l.count("|")
        if pipes == 14:
            cols_ok += 1
    if lines:
        score += 0.25 * (cols_ok / len(lines))
    if cols_ok < len(lines):
        notes.append(f"{len(lines)-cols_ok} fila(s) con # columnas incorrecto")

    # 3. Contiene 他 y 人 en sus respectivas filas (25%)
    content = " ".join(lines)
    if "他" in content:
        score += 0.125
    else:
        notes.append("falta 他")
    if "人" in content:
        score += 0.125
    else:
        notes.append("falta 人")

    # 4. Etiquetas NSM correctas (12.5% cada una)
    if re.search(r"P\.HE|P\.SHE", content):
        score += 0.125
    else:
        notes.append("falta etiqueta NSM para 他")
    if "P.PEOPLE" in content or "P.GENTE" in content:
        score += 0.125
    else:
        notes.append("falta P.PEOPLE para 人")

    if not notes:
        notes.append("fila bien formada")
    return score, "; ".join(notes)


TESTS: List[Test] = [
    Test("1A-FIDELITY-JSON", T1_SYSTEM, T1_JSON_USER,      score_fidelity_json),
    Test("1B-FIDELITY-CSV",  T1_SYSTEM, T1_CSV_USER,       score_fidelity_csv),
    Test("1C-FIDELITY-YAML", T1_SYSTEM, T1_YAML_USER,      score_fidelity_yaml),
    Test("2-RULE_ADHERENCE", T2_SYSTEM, T2_USER,           score_rule_adherence),
    Test("3-SENTENCE_GEN",   T3_SYSTEM, build_t3_prompt(), score_sentence_gen),
    Test("4-BATCH_CAPACITY", T4_SYSTEM, T4_USER,           score_batch_capacity),
    Test("5-TABLE_FILL",     T5_SYSTEM, T5_USER,           score_table_fill),
]


# =============================================================================
# RUNNER
# =============================================================================

def run_single(
    client: OpenAI,
    model: str,
    test: Test,
    max_tokens: int,
    sampling: SamplingConfig,
) -> RunResult:
    """Una sola corrida de un test. Aplica detector de colapso antes de scorer.

    Captura `finish_reason` y `completion_tokens` para distinguir:
    - Colapso REAL (modelo produjo basura o nada pese a tener tokens)
    - Colapso POR TRUNCAMIENTO (`finish_reason="length"` → subir --max-tokens)
    """
    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": test.system},
                {"role": "user", "content": test.user},
            ],
            max_tokens=max_tokens,
            **sampling.api_kwargs(),
        )
        raw = resp.choices[0].message.content or ""
        elapsed = time.time() - t0
        comp_tokens = getattr(getattr(resp, "usage", None), "completion_tokens", 0) or 0
        finish = (resp.choices[0].finish_reason or "") if resp.choices else ""
    except Exception as e:
        return RunResult(0.0, f"ERROR: {e}", time.time() - t0, True, "exception", "", 0, "error")

    collapsed, reason = detect_collapse(raw, test.name)
    if collapsed:
        # Enriquecer la razón con info del truncamiento si aplica
        if finish == "length":
            reason = f"{reason} [finish=length tokens={comp_tokens}/{max_tokens} — probable truncamiento]"
        else:
            reason = f"{reason} [finish={finish} tokens={comp_tokens}]"
        return RunResult(0.0, f"COLAPSO: {reason}", elapsed, True, reason, raw, comp_tokens, finish)

    score, details = test.scorer(raw)
    # Si la corrida válida terminó por length, anotarlo (puede faltar contenido aun si pasó scorer)
    if finish == "length":
        details = f"{details} [finish=length tokens={comp_tokens}/{max_tokens}]"
    return RunResult(score, details, elapsed, False, "", raw, comp_tokens, finish)


def run_tests(
    client: OpenAI,
    model: str,
    runs: int,
    max_tokens: int,
    sampling: SamplingConfig,
) -> List[TestResult]:
    """Corre cada test N veces, agrega resultados."""
    results = []
    for test in TESTS:
        print(f"  {test.name}")
        tr = TestResult(test.name)
        for i in range(runs):
            run = run_single(client, model, test, max_tokens, sampling)
            tr.runs.append(run)
            mark = "!" if run.collapsed else " "
            fin = f"[{run.finish_reason[:6]}/{run.completion_tokens}]"
            print(
                f"    run {i+1}/{runs}{mark} {run.score*100:5.0f}%  "
                f"({run.latency_s:4.1f}s) {fin:20s} {run.details[:70]}"
            )
        results.append(tr)
    return results


def render_report(
    model_results: dict[str, List[TestResult]],
    runs: int,
    sampling: SamplingConfig,
) -> str:
    out = ["# Benchmark LLMs locales para generación de corpus HSK\n"]
    out.append(f"Fecha: {time.strftime('%Y-%m-%d %H:%M')}  —  {runs} corridas por test\n")
    think_tag = "no-think" if sampling.no_think else "think"
    out.append(
        f"Sampling: T={sampling.temperature}, top_p={sampling.top_p}, "
        f"top_k={sampling.top_k}, presence_penalty={sampling.presence_penalty}, "
        f"mode={think_tag}\n"
    )
    out.append(
        "Métricas por test: **media** de N corridas (desviación ±, peor corrida, "
        "tasa de colapso catastrófico). El colapso se detecta por output vacío, "
        "loops de token, o ausencia total de CJK donde debería haber.\n"
    )

    # Tabla resumen — media por test
    out.append("## Resumen (media de N corridas)\n")
    test_names = [t.name for t in TESTS]
    header = "| Modelo | " + " | ".join(test_names) + " | Total | Colapsos | Latencia media |"
    sep = "|" + "|".join(["---"] * (len(test_names) + 4)) + "|"
    out.append(header)
    out.append(sep)
    for model, results in model_results.items():
        scores = [f"{r.score_mean*100:.0f}%" for r in results]
        total = sum(r.score_mean for r in results) / len(results) * 100
        total_collapses = sum(sum(1 for run in r.runs if run.collapsed) for r in results)
        total_runs = sum(len(r.runs) for r in results)
        latency = statistics.mean(r.latency_mean for r in results)
        out.append(
            f"| `{model}` | " + " | ".join(scores)
            + f" | **{total:.0f}%** | {total_collapses}/{total_runs} | {latency:.1f}s |"
        )
    out.append("")

    # Tabla de peor caso — lo que REALMENTE importa para producción
    out.append("## Peor caso (score mínimo observado por test)\n")
    out.append(
        "Si vas a correr 1000+ llamadas en batch, lo relevante no es la media "
        "sino el peor 1-percentil. Un modelo con media 90% pero peor-caso 0% "
        "(colapso) es más peligroso que uno con media 80% estable.\n"
    )
    header = "| Modelo | " + " | ".join(test_names) + " |"
    sep = "|" + "|".join(["---"] * (len(test_names) + 1)) + "|"
    out.append(header)
    out.append(sep)
    for model, results in model_results.items():
        worst = [f"{r.score_worst*100:.0f}%" for r in results]
        out.append(f"| `{model}` | " + " | ".join(worst) + " |")
    out.append("")

    # Detalles por modelo
    for model, results in model_results.items():
        out.append(f"\n## {model}\n")
        for tr in results:
            stats_line = (
                f"media {tr.score_mean*100:.0f}%  "
                f"(±{tr.score_std*100:.0f}%, peor {tr.score_worst*100:.0f}%, "
                f"colapsos {int(tr.collapse_rate*len(tr.runs))}/{len(tr.runs)})"
            )
            out.append(f"### {tr.name}  —  {stats_line}")
            out.append(f"Latencia media {tr.latency_mean:.1f}s\n")

            for i, run in enumerate(tr.runs, 1):
                mark = " ⚠ COLAPSO" if run.collapsed else ""
                trunc = " ⚠ TRUNCADO" if run.finish_reason == "length" else ""
                out.append(
                    f"**Corrida {i}** — score {run.score*100:.0f}% "
                    f"({run.latency_s:.1f}s, {run.completion_tokens} tokens, "
                    f"finish={run.finish_reason or 'n/a'}){mark}{trunc}"
                )
                out.append(f"*{run.details}*\n")
                if run.raw:
                    snippet = run.raw.strip()
                    if len(snippet) > 400:
                        snippet = snippet[:400] + "\n… [truncado]"
                    out.append("```\n" + snippet + "\n```\n")
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL", "http://localhost:1234/v1"),
    )
    p.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "lm-studio"))
    p.add_argument("--model", help="Un solo modelo")
    p.add_argument(
        "--models",
        help="Lista separada por comas (carga secuencialmente en LM Studio antes de cada modelo)",
    )
    p.add_argument("--report", default=None, help="Ruta del reporte markdown")
    p.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Corridas por test. 3 es un buen compromiso entre ruido y costo (default: 3)",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=4000,
        help="Cap de tokens de salida. Con --no-think baja mucho el consumo; "
             "subir a 8000+ solo si algún test colapsa por truncamiento (default: 4000)",
    )
    # Sampling — defaults de Qwen3 non-thinking. NO greedy (causa loops infinitos).
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument(
        "--presence-penalty",
        type=float,
        default=1.0,
        help="0-2. Combate repetición degenerada; 1.0 es buen default para reasoning models (default: 1.0)",
    )
    p.add_argument(
        "--no-think",
        action="store_true",
        help="Inyecta chat_template_kwargs.enable_thinking=False. Para Qwen3/3.5 "
             "significa skip del chain-of-thought. Ignorado por modelos sin thinking.",
    )
    args = p.parse_args()

    if not args.model and not args.models:
        raise SystemExit("pasa --model o --models")

    models = [args.model] if args.model else [m.strip() for m in args.models.split(",")]

    sampling = SamplingConfig(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        presence_penalty=args.presence_penalty,
        no_think=args.no_think,
    )

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    all_results = {}
    for model in models:
        think_tag = "no-think" if sampling.no_think else "think"
        print(
            f"\n=== {model}  (runs={args.runs}, max_tokens={args.max_tokens}, "
            f"T={sampling.temperature}, top_p={sampling.top_p}, "
            f"top_k={sampling.top_k}, pp={sampling.presence_penalty}, {think_tag}) ==="
        )
        results = run_tests(client, model, args.runs, args.max_tokens, sampling)
        print(f"\n  Resumen {model}:")
        for tr in results:
            mark = "⚠" if tr.collapse_rate > 0 else " "
            print(
                f"  {mark} {tr.name:20s} media={tr.score_mean*100:5.0f}%  "
                f"std={tr.score_std*100:4.0f}%  peor={tr.score_worst*100:5.0f}%  "
                f"colapsos={int(tr.collapse_rate*len(tr.runs))}/{len(tr.runs)}"
            )
        all_results[model] = results

    report = render_report(all_results, args.runs, sampling)
    report_path = args.report or (
        Path(__file__).parent / f"results_{time.strftime('%Y%m%d_%H%M')}.md"
    )
    Path(report_path).write_text(report, encoding="utf-8")
    print(f"\nReporte: {report_path}")


if __name__ == "__main__":
    main()
