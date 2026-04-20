"""Genera oraciones de ejemplo HSK (ZH + ES) usando un LLM local.

Destino: alimentar el Modo 3 del pipeline de audio (ver hanziflow-audio/README.md).
Por cada palabra HSK del scope, pedimos al modelo:
  - 1 oración china nivel-apropiada (5–15 caracteres, vocabulario HSK del nivel o inferior).
  - Su traducción al español, corta.

El pinyin NUNCA se pide al LLM. Se genera después con pypinyin + HSK DB +
overrides manuales (ver annotate_pinyin.py).

Default: LM Studio en localhost:1234 con mistralai/ministral-3b cargado.
Ministral es el único modelo local que pasó el benchmark interno (91% sin
colapsos). Qwen3.5-9B quedó descartado: ignora enable_thinking=false y cae en
loops de reasoning con salida vacía.

Uso:
    # Modelo default (Ministral en LM Studio)
    python generate_sentences.py --scope hsk2.0_l1

    # Prueba rápida con 5 palabras antes de escalar
    python generate_sentences.py --scope hsk2.0_l1 --limit 5

    # Los 4 scopes de un tirón
    python generate_sentences.py --scope all

Formato de salida: sentences/<scope>.md
    ## 我 (wǒ)
    ZH: 我是学生。
    ES: Soy estudiante.

Idempotente: si ya hay una entrada para la palabra, no se regenera.
"""

import argparse
import os
import re
import time
from pathlib import Path
from typing import List, Optional, Tuple

from openai import OpenAI

from config import SCOPES, SENTENCES_DIR
from sources import build_scope


# System prompt corto y neutral. Versiones largas con múltiples reglas causaban
# loops de indecisión en los modelos locales (observado en Qwen, GLM, Gemma).
SYSTEM_PROMPT = (
    "Genera oraciones HSK cortas y sus traducciones al español. "
    "Devuelve solo lo pedido, una línea por palabra, sin explicaciones."
)

USER_TEMPLATE = """Para cada palabra HSK de abajo, escribe una oración china corta
(5–15 caracteres) que la use de forma natural, y su traducción al español.

Usa SOLO vocabulario HSK {version} nivel {level} o inferior.
Genera UNA fila por palabra, en el mismo orden de la lista. No repitas palabras.

Formato exacto, tabla markdown de 3 columnas sin header:
| 我 | 我是学生。 | Soy estudiante. |
| 吃 | 我喜欢吃米饭。 | Me gusta comer arroz. |

Palabras objetivo:
{words_block}"""


def parse_response(text: str, wanted: Optional[set] = None) -> dict:
    """Extrae dict palabra -> (oración_zh, trad_es) de la respuesta del LLM.

    Acepta filas tipo tabla markdown:
        | palabra | zh | es |
        palabra | zh | es
    Se queda con la PRIMERA aparición de cada palabra (dedup contra duplicados).
    Si se pasa `wanted`, solo acepta palabras que estén en ese set.
    """
    result: dict = {}
    for line in text.splitlines():
        line = line.strip()
        if "|" not in line:
            continue
        # Despojar pipes de borde si vienen ("| a | b | c |" → "a | b | c")
        line = line.strip("|").strip()
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 3:
            continue
        # Separadores de header markdown ("---|---|---") se saltan solos por el strip
        word = re.sub(r"^\d+[\.\)]\s*", "", parts[0]).strip()
        zh = parts[1].strip()
        es = parts[2].strip() if len(parts) >= 3 else ""
        if not word or not zh:
            continue
        if set(word) <= set("-:"):  # header divisor "---"
            continue
        if wanted is not None and word not in wanted:
            continue
        # Primer hit gana → duplicados posteriores se ignoran
        result.setdefault(word, (zh, es))
    return result


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def _load_existing(path: Path) -> dict:
    """Lee sentences/<scope>.md y devuelve {palabra: (zh, es)}. Vacío si no existe."""
    if not path.exists():
        return {}
    existing = {}
    current_word: Optional[str] = None
    current_zh: Optional[str] = None
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^##\s+(\S+)", line)
        if m:
            # Cierra la entrada anterior si quedó a medias
            if current_word and current_zh:
                existing[current_word] = (current_zh, existing.get(current_word, ("", ""))[1])
            current_word = m.group(1)
            current_zh = None
            continue
        if not current_word:
            continue
        if line.startswith("ZH:"):
            val = line[3:].strip()
            current_zh = val if val and val != "<pendiente>" else None
        elif line.startswith("ES:"):
            es = line[3:].strip()
            if current_zh and es and es != "<pendiente>":
                existing[current_word] = (current_zh, es)
            current_word = None
            current_zh = None
    return existing


def _write(out_path: Path, entries: List[dict], sentences: dict, scope: str) -> None:
    lines = [
        f"# Oraciones {scope}",
        "",
        "_Generado por generate_sentences.py._ ",
        "_Cada entrada tiene header `## palabra (pinyin)` y dos líneas: `ZH:` y `ES:`._",
        "",
    ]
    for e in entries:
        simp = e["simplified"]
        pair = sentences.get(simp)
        lines.append(f"## {simp} ({e['pinyin']})")
        if pair:
            zh, es = pair
            lines.append(f"ZH: {zh}")
            lines.append(f"ES: {es}")
        else:
            lines.append("ZH: <pendiente>")
            lines.append("ES: <pendiente>")
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def generate_for_scope(
    scope: str,
    version: str,
    level: int,
    client: OpenAI,
    model: str,
    sampling: dict,
    batch_size: int = 20,
    limit: Optional[int] = None,
) -> Tuple[int, int]:
    """Devuelve (nuevas_generadas, total_en_scope)."""
    out_path = SENTENCES_DIR / f"{scope}.md"
    SENTENCES_DIR.mkdir(parents=True, exist_ok=True)

    existing = _load_existing(out_path)
    entries = build_scope(scope, version, level)
    pending = [e for e in entries if e["simplified"] not in existing]
    if limit is not None:
        pending = pending[:limit]

    print(
        f"[{scope}] total={len(entries)}  "
        f"ya={len(existing)}  pendientes={len(pending)}"
        + (f"  (limit={limit})" if limit is not None else "")
    )
    if not pending:
        print("  nada pendiente.")
        return 0, len(entries)

    all_sentences = dict(existing)
    nuevas = 0
    for batch in chunks(pending, batch_size):
        words_block = "\n".join(
            f"- {e['simplified']} ({e['pinyin']})" for e in batch
        )
        user_msg = USER_TEMPLATE.format(
            level=level, version=version, words_block=words_block
        )
        print(
            f"  batch de {len(batch)} "
            f"[{batch[0]['simplified']}...{batch[-1]['simplified']}]"
        )

        t0 = time.time()
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                **sampling,
            )
        except Exception as exc:
            print(f"    ⚠ fallo API: {exc}")
            continue

        elapsed = time.time() - t0
        text = response.choices[0].message.content or ""
        wanted = {e["simplified"] for e in batch}
        parsed = parse_response(text, wanted=wanted)
        hits = sum(1 for e in batch if e["simplified"] in parsed)
        print(f"    ↳ {hits}/{len(batch)} en {elapsed:.1f}s")

        # Si no parseó nada, vuelca el raw para debug y muestra un preview.
        if hits == 0:
            debug_path = SENTENCES_DIR / f"{scope}.debug.txt"
            debug_path.write_text(
                f"# batch [{batch[0]['simplified']}...{batch[-1]['simplified']}]\n"
                f"# finish_reason={response.choices[0].finish_reason}\n"
                f"# usage={getattr(response, 'usage', None)}\n"
                f"--- raw content ---\n{text}\n",
                encoding="utf-8",
            )
            preview = text[:500].replace("\n", " | ")
            print(f"    ⚠ 0 parseos. raw guardado en {debug_path}")
            print(f"    preview: {preview}")

        for e in batch:
            if e["simplified"] in parsed:
                all_sentences[e["simplified"]] = parsed[e["simplified"]]
                nuevas += 1

        # Write-through para no perder progreso ante un crash
        _write(out_path, entries, all_sentences, scope)

    _write(out_path, entries, all_sentences, scope)
    print(f"  ✓ guardado en {out_path} (+{nuevas} nuevas, {len(all_sentences)} total)")
    return nuevas, len(entries)


def build_sampling(args) -> dict:
    """Construye kwargs de sampling. Defaults Qwen-safe (sirven también para Ministral).

    Convención de "apagar un parámetro": pasar 0 (o None cuando aplique).
      top_k=0            → no se envía extra_body (Gemini OpenAI-compat no lo acepta).
      presence_penalty=0 → no se envía el campo (gemini-3-*-preview lo rechaza).
    no_think solo aplica para LM Studio / Ollama (Qwen, GLM, Gemma).
    reasoning_effort es para Gemini 3.x y o-series — va como parámetro de primer
    nivel, NO dentro de extra_body.
    """
    kwargs = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
    }
    if args.presence_penalty and args.presence_penalty != 0:
        kwargs["presence_penalty"] = args.presence_penalty
    if getattr(args, "reasoning_effort", None):
        kwargs["reasoning_effort"] = args.reasoning_effort
    extra: dict = {}
    if args.top_k and args.top_k > 0:
        extra["top_k"] = args.top_k
    if args.no_think:
        extra["chat_template_kwargs"] = {"enable_thinking": False}
    if extra:
        kwargs["extra_body"] = extra
    return kwargs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generar oraciones HSK (ZH + ES) vía LLM local para Modo 3"
    )
    parser.add_argument("--scope", default="all", help="scope específico o 'all'")
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL", "http://localhost:1234/v1"),
        help="Endpoint OpenAI-compatible (LM Studio: 1234, Ollama: 11434)",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENAI_API_KEY", "lm-studio"),
        help="API key (cualquier valor para LM Studio/Ollama)",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("SENTENCES_MODEL", "mistralai/ministral-3b"),
        help="Modelo a usar (default Ministral 3B — único que pasó el benchmark interno)",
    )
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Generar solo las primeras N palabras pendientes (prueba rápida)",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--presence-penalty", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=4000)
    parser.add_argument(
        "--no-think", action="store_true",
        help="Inyecta chat_template_kwargs.enable_thinking=False (Qwen3/3.5)",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high"],
        default=None,
        help="Gemini 3.x / o-series: nivel de thinking (solo si el modelo lo soporta)",
    )
    args = parser.parse_args()

    if args.scope == "all":
        scopes = SCOPES
    else:
        scopes = [s for s in SCOPES if s[0] == args.scope]
        if not scopes:
            print(f"scope '{args.scope}' no existe. Ver SCOPES en config.py.")
            return

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    sampling = build_sampling(args)

    print(f"Modelo: {args.model} @ {args.base_url}")
    print(f"Sampling: temp={args.temperature} top_p={args.top_p} "
          f"top_k={args.top_k} presence={args.presence_penalty} "
          f"no_think={args.no_think} "
          f"reasoning_effort={args.reasoning_effort or 'off'}")
    print()

    total_nuevas = 0
    for scope_name, ver, lvl in scopes:
        nuevas, _ = generate_for_scope(
            scope_name, ver, lvl, client, args.model, sampling,
            args.batch_size, args.limit,
        )
        total_nuevas += nuevas
        print()

    print(f"Total nuevas oraciones generadas: {total_nuevas}")


if __name__ == "__main__":
    main()
