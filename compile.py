"""Compila MP3s individuales (cache/) en audios finales por scope × voz × modo.

Produce archivos agrupados por scope (una carpeta por lista HSK):
    output/edge/hsk2.0_l1/v1/hsk2.0_l1_corta_v1.mp3
    output/edge/hsk2.0_l1/v2/hsk2.0_l1_corta_v2.mp3
    output/edge/hsk2.0_l1/v3/hsk2.0_l1_corta_v3sub1.mp3
    output/edge/hsk2.0_l1/v3/hsk2.0_l1_corta_v3sub2.mp3
    output/edge/hsk2.0_l1/v3/hsk2.0_l1_corta_v3sub3.mp3
    output/edge/hsk2.0_l1/v3/hsk2.0_l1_corta_v3sub4.mp3

Formato de cada modo:
    v1     : [CN_word] [pausa] [ES_word] [pausa entre entradas]
    v2     : [CN_word] [pausa larga] [CN_word repetido] [pausa entre entradas]
    v3sub1 : [CN_word] [pausa] [ES_word] [pausa] [CN_sent] [pausa] [ES_sent] [pausa entre entradas]
    v3sub2 : [CN_word] [pausa] [CN_sent] [pausa] [ES_sent] [pausa entre entradas]
    v3sub3 : [CN_word] [pausa] [ES_word] [pausa] [CN_sent] [pausa entre entradas]    (legacy v3)
    v3sub4 : [CN_word] [pausa] [CN_sent] [pausa entre entradas]                      (inmersivo total)

Uso:
    python compile.py --scope hsk2.0_l1 --voice corta --mode v1
    python compile.py --scope all --voice all --mode all
    python compile.py --scope hsk3.0_l1 --voice all --mode v3sub4  # sin ES, usa caché actual
"""

import argparse
import hashlib
from pathlib import Path
from typing import List

from pydub import AudioSegment

from config import (
    CACHE_DIR,
    CN_VOICES,
    OUTPUT_DIR,
    SCOPES,
    V1_PAUSE_AFTER_CN,
    V1_PAUSE_AFTER_ES,
    V1_PAUSE_BETWEEN_ENTRIES,
    V2_PAUSE_AFTER_FIRST_CN,
    V2_PAUSE_AFTER_SECOND_CN,
    V2_PAUSE_BETWEEN_ENTRIES,
    V3_PAUSE_AFTER_CN,
    V3_PAUSE_AFTER_ES,
    V3_PAUSE_AFTER_SENTENCE,
)
from sources import build_scope


def entry_id(simplified: str) -> str:
    return hashlib.md5(simplified.encode("utf-8")).hexdigest()[:10]


def silence(ms: int) -> AudioSegment:
    return AudioSegment.silent(duration=ms)


def load_clip(path: Path) -> AudioSegment | None:
    if not path.exists():
        return None
    return AudioSegment.from_file(path)


def compile_v1(scope: str, voice_key: str, entries: List[dict]) -> AudioSegment:
    track = AudioSegment.silent(duration=500)
    missing_es = 0
    for e in entries:
        eid = entry_id(e["simplified"])
        cn = load_clip(CACHE_DIR / scope / voice_key / f"{eid}_cn.mp3")
        es = load_clip(CACHE_DIR / scope / "_es" / f"{eid}_es.mp3")
        if cn is None:
            continue
        track += cn + silence(V1_PAUSE_AFTER_CN)
        if es is not None:
            track += es + silence(V1_PAUSE_AFTER_ES)
        else:
            missing_es += 1
            track += silence(V1_PAUSE_AFTER_ES)
        track += silence(V1_PAUSE_BETWEEN_ENTRIES)
    if missing_es:
        print(f"  ⚠ v1: {missing_es} entradas sin traducción ES (pausa vacía)")
    return track


def compile_v2(scope: str, voice_key: str, entries: List[dict]) -> AudioSegment:
    track = AudioSegment.silent(duration=500)
    for e in entries:
        eid = entry_id(e["simplified"])
        cn = load_clip(CACHE_DIR / scope / voice_key / f"{eid}_cn.mp3")
        if cn is None:
            continue
        track += cn + silence(V2_PAUSE_AFTER_FIRST_CN)
        track += cn + silence(V2_PAUSE_AFTER_SECOND_CN)
        track += silence(V2_PAUSE_BETWEEN_ENTRIES)
    return track


def _load_v3_clips(scope: str, voice_key: str, e: dict) -> dict:
    """Devuelve los 4 posibles clips de una entrada V3."""
    eid = entry_id(e["simplified"])
    return {
        "cn_word": load_clip(CACHE_DIR / scope / voice_key / f"{eid}_cn.mp3"),
        "cn_sent": load_clip(CACHE_DIR / scope / voice_key / f"{eid}_sent.mp3"),
        "es_word": load_clip(CACHE_DIR / scope / "_es" / f"{eid}_es.mp3"),
        "es_sent": load_clip(CACHE_DIR / scope / "_es" / f"{eid}_sent_es.mp3"),
    }


def compile_v3sub1(scope: str, voice_key: str, entries: List[dict]) -> AudioSegment:
    """sub1: CN_word + ES_word + CN_sent + ES_sent (completo bilingüe)."""
    track = AudioSegment.silent(duration=500)
    missing = {"es_word": 0, "cn_sent": 0, "es_sent": 0}
    for e in entries:
        c = _load_v3_clips(scope, voice_key, e)
        if c["cn_word"] is None:
            continue
        track += c["cn_word"] + silence(V3_PAUSE_AFTER_CN)
        track += (c["es_word"] + silence(V3_PAUSE_AFTER_ES)) if c["es_word"] else silence(V3_PAUSE_AFTER_ES)
        if not c["es_word"]:
            missing["es_word"] += 1
        track += (c["cn_sent"] + silence(V3_PAUSE_AFTER_CN)) if c["cn_sent"] else silence(V3_PAUSE_AFTER_CN)
        if not c["cn_sent"]:
            missing["cn_sent"] += 1
        track += (c["es_sent"] + silence(V3_PAUSE_AFTER_SENTENCE)) if c["es_sent"] else silence(V3_PAUSE_AFTER_SENTENCE)
        if not c["es_sent"]:
            missing["es_sent"] += 1
    for k, v in missing.items():
        if v:
            print(f"  ⚠ v3sub1: {v} entradas sin {k} (pausa vacía)")
    return track


def compile_v3sub2(scope: str, voice_key: str, entries: List[dict]) -> AudioSegment:
    """sub2: CN_word + CN_sent + ES_sent (sin palabra ES, contexto bilingüe)."""
    track = AudioSegment.silent(duration=500)
    missing = {"cn_sent": 0, "es_sent": 0}
    for e in entries:
        c = _load_v3_clips(scope, voice_key, e)
        if c["cn_word"] is None:
            continue
        track += c["cn_word"] + silence(V3_PAUSE_AFTER_CN)
        track += (c["cn_sent"] + silence(V3_PAUSE_AFTER_CN)) if c["cn_sent"] else silence(V3_PAUSE_AFTER_CN)
        if not c["cn_sent"]:
            missing["cn_sent"] += 1
        track += (c["es_sent"] + silence(V3_PAUSE_AFTER_SENTENCE)) if c["es_sent"] else silence(V3_PAUSE_AFTER_SENTENCE)
        if not c["es_sent"]:
            missing["es_sent"] += 1
    for k, v in missing.items():
        if v:
            print(f"  ⚠ v3sub2: {v} entradas sin {k} (pausa vacía)")
    return track


def compile_v3sub3(scope: str, voice_key: str, entries: List[dict]) -> AudioSegment:
    """sub3: CN_word + ES_word + CN_sent (legacy v3, ES puntual sobre oración)."""
    track = AudioSegment.silent(duration=500)
    used = 0
    for e in entries:
        c = _load_v3_clips(scope, voice_key, e)
        if c["cn_word"] is None:
            continue
        track += c["cn_word"] + silence(V3_PAUSE_AFTER_CN)
        if c["es_word"] is not None:
            track += c["es_word"] + silence(V3_PAUSE_AFTER_ES)
        if c["cn_sent"] is not None:
            track += c["cn_sent"] + silence(V3_PAUSE_AFTER_SENTENCE)
            used += 1
        else:
            track += silence(V3_PAUSE_AFTER_SENTENCE)
    print(f"  ↳ v3sub3: {used}/{len(entries)} entradas con oración")
    return track


def compile_v3sub4(scope: str, voice_key: str, entries: List[dict]) -> AudioSegment:
    """sub4: CN_word + CN_sent (inmersivo total, sin muletas ES)."""
    track = AudioSegment.silent(duration=500)
    used = 0
    for e in entries:
        c = _load_v3_clips(scope, voice_key, e)
        if c["cn_word"] is None:
            continue
        track += c["cn_word"] + silence(V3_PAUSE_AFTER_CN)
        if c["cn_sent"] is not None:
            track += c["cn_sent"] + silence(V3_PAUSE_AFTER_SENTENCE)
            used += 1
        else:
            track += silence(V3_PAUSE_AFTER_SENTENCE)
    print(f"  ↳ v3sub4: {used}/{len(entries)} entradas con oración")
    return track


def compile_scope(
    scope: str, hsk_version: str, level: int, voice_keys: List[str], modes: List[str]
) -> None:
    entries = build_scope(scope, hsk_version, level)
    print(f"\n[{scope}] {len(entries)} entradas")

    for voice_key in voice_keys:
        for mode in modes:
            print(f"  compilando {voice_key} / {mode}...")
            if mode == "v1":
                track = compile_v1(scope, voice_key, entries)
            elif mode == "v2":
                track = compile_v2(scope, voice_key, entries)
            elif mode == "v3sub1":
                track = compile_v3sub1(scope, voice_key, entries)
            elif mode == "v3sub2":
                track = compile_v3sub2(scope, voice_key, entries)
            elif mode == "v3sub3":
                track = compile_v3sub3(scope, voice_key, entries)
            elif mode == "v3sub4":
                track = compile_v3sub4(scope, voice_key, entries)
            else:
                continue

            # Output dir: v3sub* todos van a v3/ (mismo scope folder)
            out_parent = "v3" if mode.startswith("v3sub") else mode
            out_dir = OUTPUT_DIR / "edge" / scope / out_parent
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{scope}_{voice_key}_{mode}.mp3"
            track.export(
                out_path,
                format="mp3",
                bitrate="96k",
                tags={
                    "title": f"HanziFlow {scope} [{voice_key} {mode}]",
                    "artist": "HanziFlow",
                    "album": f"HanziFlow {scope}",
                },
            )
            dur_min = len(track) / 1000 / 60
            size_kb = out_path.stat().st_size / 1024
            print(
                f"    ✓ {out_path.name}  "
                f"({dur_min:.1f} min, {size_kb:.0f} KB)"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compilar MP3s finales")
    parser.add_argument("--scope", default="all")
    parser.add_argument("--voice", default="all")
    parser.add_argument(
        "--mode",
        default="all",
        help="v1 | v2 | v3sub1 | v3sub2 | v3sub3 | v3sub4 | v3 (=all v3sub*) | all",
    )
    args = parser.parse_args()

    if args.scope == "all":
        scopes = SCOPES
    else:
        scopes = [s for s in SCOPES if s[0] == args.scope]
        if not scopes:
            raise SystemExit(f"scope desconocido: {args.scope}")

    if args.voice == "all":
        voice_keys = list(CN_VOICES.keys())
    else:
        voice_keys = [args.voice]

    if args.mode == "all":
        modes = ["v1", "v2", "v3sub1", "v3sub2", "v3sub3", "v3sub4"]
    elif args.mode == "v3":
        modes = ["v3sub1", "v3sub2", "v3sub3", "v3sub4"]
    else:
        modes = [args.mode]

    for scope_name, ver, lvl in scopes:
        compile_scope(scope_name, ver, lvl, voice_keys, modes)

    print("\n✓ Compilación completa. MP3s en output/edge/")


if __name__ == "__main__":
    main()
