"""Carga entradas de vocabulario desde la base HSK + overlays de traducción ES.

Cada entrada normalizada tiene la forma:
    {
        "simplified": str,
        "pinyin": str,
        "meanings_en": list[str],
        "translation_es": str | None,     # de translations/*.json
        "example_sentence": str | None,   # de sentences/*.md (v3)
    }

Las traducciones ES se guardan en translations/<scope>.json como:
    { "我": "yo", "的": "partícula posesiva; de", ... }

Las oraciones se guardan en sentences/<scope>.md con formato:
    ## 我 (wǒ)
    我是中国人。
    Soy chino.

El header admite el pinyin entre paréntesis opcionalmente para legibilidad
humana; el parser solo usa el hanzi como key. La segunda línea (traducción ES)
se ignora aquí porque las traducciones se cargan desde translations/*.json.
"""

import json
import re
import sqlite3
from pathlib import Path
from typing import List, Optional

from config import HSK_DB, TRANSLATIONS_DIR, SENTENCES_DIR


# Separador de formas alternativas en la DB v2 (convención heredada de la
# fuente HSK original). Ej: `爸爸｜爸`, `零｜〇`. Es `｜` U+FF5C (full-width),
# NO el pipe ASCII. El schema v2 no tiene campo `alternative_forms` todavía,
# así que se concatenó con este separador. Aquí lo deshacemos para el audio:
# nos quedamos con la forma izquierda (canónica HSK, la más larga/oficial).
# Parche táctico — borrar cuando el schema v3 introduzca `alternative_forms`.
ALT_FORM_SEPARATOR = "｜"


def _canonical_form(text: str) -> str:
    """Devuelve la parte a la izquierda del separador ｜ y la strippea.

    Idempotente para strings sin el separador. Aplica tanto a `simplified`
    (`"爸爸｜爸"` → `"爸爸"`) como a `pinyin` (`"bàba ｜ bà"` → `"bàba"`).
    """
    if ALT_FORM_SEPARATOR not in text:
        return text
    return text.split(ALT_FORM_SEPARATOR, 1)[0].strip()


def load_hsk_level(hsk_version: str, level: int) -> List[dict]:
    """Carga todas las palabras de un nivel HSK desde la base compartida."""
    if not HSK_DB.exists():
        raise FileNotFoundError(
            f"HSK DB no encontrada en {HSK_DB}. "
            "Corre el pipeline HSK-word-list primero."
        )

    conn = sqlite3.connect(HSK_DB)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        """
        SELECT DISTINCT v.simplified, v.pinyin, v.meanings_en
        FROM vocabulary v
        JOIN vocabulary_levels vl ON vl.simplified = v.simplified
        WHERE vl.hsk_version = ? AND vl.level = ?
        ORDER BY COALESCE(v.frequency_rank, 999999), v.simplified
        """,
        (hsk_version, level),
    )
    rows = cur.fetchall()
    conn.close()

    # Dedup post-normalización: la DB v2 guarda algunas entradas en dos filas
    # paralelas — una limpia (`爸爸`, con frequency_rank) y otra con separador
    # (`爸爸｜爸`, sin frequency_rank y con pinyin peor formateado "bàba ｜ bà").
    # Tras `_canonical_form` ambas colapsan a la misma key. El ORDER BY de
    # arriba coloca primero la fila con frequency_rank real (COALESCE a 999999
    # manda las None al final), así que quedarnos con la primera ocurrencia
    # preserva el pinyin bien espaciado ("bà ba") y los meanings_en canónicos.
    # Parche táctico — borrar cuando v3 tenga alternative_forms como columna.
    entries: list[dict] = []
    seen: set[str] = set()
    for row in rows:
        simp = _canonical_form(row["simplified"])
        if simp in seen:
            continue
        seen.add(simp)
        meanings = []
        if row["meanings_en"]:
            try:
                meanings = json.loads(row["meanings_en"])
            except json.JSONDecodeError:
                meanings = [row["meanings_en"]]
        entries.append(
            {
                "simplified": simp,
                "pinyin": _canonical_form(row["pinyin"] or ""),
                "meanings_en": meanings,
            }
        )
    return entries


def load_translations_es(scope: str) -> dict:
    """Devuelve dict simplified -> traducción española. Vacío si no existe aún."""
    path = TRANSLATIONS_DIR / f"{scope}.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_sentences(scope: str) -> dict:
    """Devuelve dict simplified -> oración CN (para v3). Vacío si no existe."""
    path = SENTENCES_DIR / f"{scope}.md"
    if not path.exists():
        return {}

    sentences = {}
    current_word: Optional[str] = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            # Acepta `## 我` y también `## 我 (wǒ)` — el pinyin entre paréntesis
            # es decorativo, solo usamos el primer token (hanzi) como key.
            m = re.match(r"^##\s+(\S+)", line)
            if m:
                current_word = m.group(1)
                continue
            if not current_word:
                continue
            stripped = line.strip()
            if not stripped:
                continue
            # Tolera dos formatos para la oración CN:
            #   "ZH: 我是中国人。"   ← formato Gemini con prefijo
            #   "我是中国人。"       ← formato legacy sin prefijo
            # Ignora la línea ES: (traducciones vienen de translations/*.json).
            upper = stripped.upper()
            if upper.startswith("ES:") or upper.startswith("ESPAÑOL:"):
                continue
            if upper.startswith("ZH:") or upper.startswith("CN:"):
                stripped = stripped.split(":", 1)[1].strip()
            if stripped:
                sentences.setdefault(current_word, stripped)
    return sentences


def load_sentences_es(scope: str) -> dict:
    """Devuelve dict simplified -> oración ES (para v3sub1/sub2). Vacío si no existe."""
    path = SENTENCES_DIR / f"{scope}.md"
    if not path.exists():
        return {}

    sentences = {}
    current_word: Optional[str] = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            m = re.match(r"^##\s+(\S+)", line)
            if m:
                current_word = m.group(1)
                continue
            if not current_word:
                continue
            stripped = line.strip()
            if not stripped:
                continue
            upper = stripped.upper()
            if upper.startswith("ES:") or upper.startswith("ESPAÑOL:"):
                es_text = stripped.split(":", 1)[1].strip()
                if es_text:
                    sentences.setdefault(current_word, es_text)
    return sentences


def build_scope(scope: str, hsk_version: str, level: int) -> List[dict]:
    """Combina datos HSK + traducciones ES + oraciones (CN+ES) para un scope concreto."""
    words = load_hsk_level(hsk_version, level)
    translations = load_translations_es(scope)
    sentences = load_sentences(scope)
    sentences_es = load_sentences_es(scope)

    enriched = []
    for w in words:
        simp = w["simplified"]
        enriched.append(
            {
                **w,
                "translation_es": translations.get(simp),
                "example_sentence": sentences.get(simp),
                "example_sentence_es": sentences_es.get(simp),
            }
        )
    return enriched


if __name__ == "__main__":
    # Diagnóstico rápido: cuántas traducciones faltan en cada scope
    from config import SCOPES

    for scope, ver, lvl in SCOPES:
        entries = build_scope(scope, ver, lvl)
        total = len(entries)
        with_es = sum(1 for e in entries if e["translation_es"])
        with_sent = sum(1 for e in entries if e["example_sentence"])
        print(
            f"{scope:15s}  total={total:4d}  "
            f"ES={with_es:4d} ({100*with_es//max(total,1):3d}%)  "
            f"oraciones={with_sent:4d}"
        )
