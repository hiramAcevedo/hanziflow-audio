"""Genera archivos TTS individuales (por palabra y traducción) usando edge-tts.

Cada entrada HSK produce hasta 3 archivos en cache/<scope>/<voice>/:
    - <id>_cn.mp3    (voz china leyendo hanzi)
    - <id>_es.mp3    (voz española leyendo traducción)
    - <id>_sent.mp3  (voz china leyendo oración de ejemplo) — si existe

Los archivos individuales son intermedios. compile.py los concatena.

Uso:
    python generate_edge.py --scope hsk2.0_l1 --voice yunjian --mode v1
    python generate_edge.py --scope hsk2.0_l1 --voice all --mode both

Modos:
    v1   — genera CN + ES (bilingüe)
    v2   — genera solo CN (inmersivo, pero es el mismo archivo CN que v1)
    v3   — genera CN + ES + oración
    all  — genera todo lo que haya
    both — alias de all
"""

import argparse
import asyncio
import hashlib
from pathlib import Path
from typing import List

import edge_tts

from config import (
    CACHE_DIR,
    CN_RATE,
    CN_VOICES,
    ES_RATE,
    ES_VOICE,
    SCOPES,
)
from sources import build_scope


CONCURRENCY = 5  # edge-tts tolera hasta ~8 sin rate-limit; 5 es conservador


def entry_id(simplified: str) -> str:
    """ID estable y corto para el archivo (hash del hanzi, evita problemas FS)."""
    return hashlib.md5(simplified.encode("utf-8")).hexdigest()[:10]


# Umbral mínimo para considerar un mp3 como cache-hit válido. edge-tts a veces
# crea el archivo antes de stream-earlo; si el WebSocket muere a mitad (503,
# network blip), queda un archivo de 0 bytes que en la próxima corrida se
# detectaba como "ya existe" y jamás se regeneraba. Un mp3 CN/ES típico pesa
# >5KB incluso para una sola palabra; 200 bytes es un umbral seguro.
MIN_VALID_MP3_BYTES = 200


async def synth(text: str, voice: str, rate: str, out_path: Path) -> None:
    """Sintetiza un segmento con edge-tts y lo escribe a disco.

    Cache-hit sólo si el archivo existe Y tiene tamaño plausible. Archivos
    truncados (típicamente por 503 transitorio de Bing) se sobrescriben.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size >= MIN_VALID_MP3_BYTES:
        return  # cache hit válido
    if out_path.exists():
        out_path.unlink()  # archivo corrupto previo — limpiar antes de resynth
    communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate)
    await communicate.save(str(out_path))


async def _worker(queue: asyncio.Queue) -> None:
    while True:
        task = await queue.get()
        if task is None:
            queue.task_done()
            return
        text, voice, rate, out_path = task
        try:
            await synth(text, voice, rate, out_path)
        except Exception as exc:
            print(f"  ! error sintetizando {out_path.name}: {exc}")
        queue.task_done()


async def generate_for_scope(
    scope: str,
    hsk_version: str,
    level: int,
    voice_keys: List[str],
    include_es: bool,
    include_sentence: bool,
    include_sentence_es: bool,
    include_cn: bool = True,
) -> None:
    entries = build_scope(scope, hsk_version, level)
    print(f"\n[{scope}] {len(entries)} entradas — voces CN: {', '.join(voice_keys)}")

    queue: asyncio.Queue = asyncio.Queue()
    workers = [asyncio.create_task(_worker(queue)) for _ in range(CONCURRENCY)]

    total = 0
    for e in entries:
        eid = entry_id(e["simplified"])

        # Chino — una versión por voz CN elegida
        if include_cn:
            for vk in voice_keys:
                voice = CN_VOICES[vk]
                out = CACHE_DIR / scope / vk / f"{eid}_cn.mp3"
                await queue.put((e["simplified"], voice, CN_RATE, out))
                total += 1

        # Español palabra (una sola versión, compartida entre todas las voces CN)
        if include_es and e.get("translation_es"):
            out = CACHE_DIR / scope / "_es" / f"{eid}_es.mp3"
            await queue.put((e["translation_es"], ES_VOICE, ES_RATE, out))
            total += 1

        # Oración CN (una por voz CN)
        if include_sentence and e.get("example_sentence"):
            for vk in voice_keys:
                voice = CN_VOICES[vk]
                out = CACHE_DIR / scope / vk / f"{eid}_sent.mp3"
                await queue.put((e["example_sentence"], voice, CN_RATE, out))
                total += 1

        # Oración ES (una sola, compartida entre voces CN) — necesaria para v3sub1 y v3sub2
        if include_sentence_es and e.get("example_sentence_es"):
            out = CACHE_DIR / scope / "_es" / f"{eid}_sent_es.mp3"
            await queue.put((e["example_sentence_es"], ES_VOICE, ES_RATE, out))
            total += 1

    print(f"  ↳ {total} archivos TTS en cola (cache se respeta)")
    # Sentinel por worker
    for _ in workers:
        await queue.put(None)
    await queue.join()
    for w in workers:
        await w


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch TTS con edge-tts")
    parser.add_argument(
        "--scope",
        default="all",
        help="hsk2.0_l1 | hsk2.0_l2 | hsk3.0_l1 | hsk3.0_l2 | all",
    )
    parser.add_argument(
        "--voice",
        default="all",
        help="yunjian | yunyang | yunxi | all",
    )
    parser.add_argument(
        "--mode",
        default="v1",
        choices=["v1", "v2", "v3", "v3sub1", "v3sub2", "v3sub3", "v3sub4", "sent_es", "all"],
        help=(
            "v1=CN+ES_word, v2=solo CN, "
            "v3=CN+ES_word+CN_sent (legacy), "
            "v3sub1=+ES_sent, v3sub2=+CN_sent+ES_sent, "
            "v3sub3=v3 legacy, v3sub4=CN+CN_sent, "
            "sent_es=solo oraciones_ES, all=todo"
        ),
    )
    args = parser.parse_args()

    # Scopes a procesar
    if args.scope == "all":
        scopes = SCOPES
    else:
        scopes = [s for s in SCOPES if s[0] == args.scope]
        if not scopes:
            raise SystemExit(f"scope desconocido: {args.scope}")

    # Voces CN
    if args.voice == "all":
        voice_keys = list(CN_VOICES.keys())
    else:
        if args.voice not in CN_VOICES:
            raise SystemExit(f"voz desconocida: {args.voice}")
        voice_keys = [args.voice]

    # Qué generar por modo
    # sent_es: modo aislado para generar solo oraciones ES (ningún clip CN/ES_word)
    if args.mode == "sent_es":
        include_cn = False
        include_es = False
        include_sentence = False
        include_sentence_es = True
    else:
        include_cn = True
        include_es = args.mode in ("v1", "v3", "v3sub1", "v3sub3", "all")
        include_sentence = args.mode in (
            "v3", "v3sub1", "v3sub2", "v3sub3", "v3sub4", "all",
        )
        include_sentence_es = args.mode in ("v3sub1", "v3sub2", "all")

    async def runall():
        for scope_name, ver, lvl in scopes:
            await generate_for_scope(
                scope_name, ver, lvl, voice_keys,
                include_es, include_sentence, include_sentence_es,
                include_cn=include_cn,
            )

    asyncio.run(runall())
    print("\n✓ Generación TTS completa. Usa compile.py para armar los MP3 finales.")


if __name__ == "__main__":
    main()
