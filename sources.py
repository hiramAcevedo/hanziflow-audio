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
    ## 我
    我是中国人。
"""

import json
import re
import sqlite3
from pathlib import Path
from typing import List, Optional

from config import HSK_DB, TRANSLATIONS_DIR, SENTENCES_DIR


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

    entries = []
    for row in rows:
        meanings = []
        if row["meanings_en"]:
            try:
                meanings = json.loads(row["meanings_en"])
            except json.JSONDecodeError:
                meanings = [row["meanings_en"]]
        entries.append(
            {
                "simplified": row["simplified"],
                "pinyin": row["pinyin"] or "",
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
            m = re.match(r"^##\s+(\S+)\s*$", line)
            if m:
                current_word = m.group(1)
                continue
            if current_word and line.strip():
                # Primera línea no vacía tras el header = oración
                sentences.setdefault(current_word, line.strip())
    return sentences


def build_scope(scope: str, hsk_version: str, level: int) -> List[dict]:
    """Combina datos HSK + traducciones ES + oraciones para un scope concreto."""
    words = load_hsk_level(hsk_version, level)
    translations = load_translations_es(scope)
    sentences = load_sentences(scope)

    enriched = []
    for w in words:
        simp = w["simplified"]
        enriched.append(
            {
                **w,
                "translation_es": translations.get(simp),
                "example_sentence": sentences.get(simp),
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
