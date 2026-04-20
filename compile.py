"""Compila MP3s individuales (cache/) en audios finales por scope × voz × modo.

Produce archivos como:
    output/edge/v1/hsk2.0_l1_yunjian_v1.mp3
    output/edge/v2/hsk2.0_l1_yunjian_v2.mp3
    output/edge/v3/hsk2.0_l1_yunjian_v3.mp3

Formato de cada modo:
    v1: [CN] [pausa] [ES] [pausa entre entradas]
    v2: [CN] [pausa larga] [CN repetido] [pausa entre entradas]
    v3: [CN] [pausa] [ES] [pausa] [oración CN] [pausa entre entradas]

Uso:
    python compile.py --scope hsk2.0_l1 --voice yunjian --mode v1
    python compile.py --scope all --voice all --mode all
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


def compile_v3(scope: str, voice_key: str, entries: List[dict]) -> AudioSegment:
    track = AudioSegment.silent(duration=500)
    used = 0
    for e in entries:
        eid = entry_id(e["simplified"])
        cn = load_clip(CACHE_DIR / scope / voice_key / f"{eid}_cn.mp3")
        es = load_clip(CACHE_DIR / scope / "_es" / f"{eid}_es.mp3")
        sent = load_clip(CACHE_DIR / scope / voice_key / f"{eid}_sent.mp3")
        if cn is None:
            continue
        track += cn + silence(V3_PAUSE_AFTER_CN)
        if es is not None:
            track += es + silence(V3_PAUSE_AFTER_ES)
        if sent is not None:
            track += sent + silence(V3_PAUSE_AFTER_SENTENCE)
            used += 1
        else:
            track += silence(V3_PAUSE_AFTER_SENTENCE)
    print(f"  ↳ v3: {used}/{len(entries)} entradas con oración")
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
            elif mode == "v3":
                track = compile_v3(scope, voice_key, entries)
            else:
                continue

            out_dir = OUTPUT_DIR / "edge" / mode
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
        "--mode", default="all", help="v1 | v2 | v3 | all"
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
        modes = ["v1", "v2", "v3"]
    else:
        modes = [args.mode]

    for scope_name, ver, lvl in scopes:
        compile_scope(scope_name, ver, lvl, voice_keys, modes)

    print("\n✓ Compilación completa. MP3s en output/edge/")


if __name__ == "__main__":
    main()
