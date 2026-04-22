#!/usr/bin/env python3
"""
Top-tier 4-model council runner for high-stakes questions.

Models (all via OpenRouter, high reasoning effort):
  - openai/gpt-5.4
  - anthropic/claude-opus-4.7
  - google/gemini-3.1-pro-preview
  - x-ai/grok-4.20

Chairman: anthropic/claude-opus-4.7 (high effort).

Usage:
  python council_topfour.py \
      --query-file question.md \
      --out-dir C:\\Users\\bhara\\dev\\specs\\codex-workflow-system \
      --name playbook-credential-hardening \
      --type playbook \
      --project codex-workflow-system \
      --title "Credential Hardening Playbook"

Output:
  {out-dir}/{name}.md    synthesis with frontmatter + backlinks
  {out-dir}/{name}.json  raw perspectives + usage
"""

import os
import sys
import json
import asyncio
import argparse
from datetime import datetime
from pathlib import Path

import aiohttp
from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).parent
load_dotenv(SCRIPT_DIR / ".env")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

THEORISTS = [
    {"name": "gpt",    "slug": "openai/gpt-5.4",                "effort": "high"},
    {"name": "claude", "slug": "anthropic/claude-opus-4.7",     "effort": "high"},
    {"name": "gemini", "slug": "google/gemini-3.1-pro-preview", "effort": "high"},
    {"name": "grok",   "slug": "x-ai/grok-4.20",                "effort": "high"},
]
CHAIRMAN_SLUG = "anthropic/claude-opus-4.7"
CHAIRMAN_EFFORT = "high"

MAX_RETRIES = 3
RETRY_DELAY = 4
MAX_TOKENS_THEORIST = 8000
MAX_TOKENS_CHAIRMAN = 16000
REQUEST_TIMEOUT = 600

THEORIST_SYSTEM = (
    "You are a senior specialist on a Council of frontier AI models. "
    "Give concrete, actionable, correct guidance on the user's question. "
    "Be thorough, opinionated, and call out weaknesses in any premise the user states. "
    "Prefer specific commands and code where they help."
)

CHAIRMAN_SYSTEM = (
    "You are the Chairman of a Council of frontier AI advisors. "
    "Synthesize the theorist responses into a single authoritative playbook. "
    "Preserve the strongest concrete recommendations; resolve contradictions; "
    "where theorists disagree, state which position is safer or stronger and why. "
    "Structure the output with clear numbered sections. End with a short "
    "'Council Disagreements and Verdict' section. Be dense, specific, command-level."
)


async def call_openrouter(session, slug, system, prompt, max_tokens, effort=None, tag=""):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/research-council",
        "X-Title": "Top-Tier Council",
    }
    payload = {
        "model": slug,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    }
    if effort:
        payload["reasoning"] = {"effort": effort}

    for attempt in range(MAX_RETRIES):
        try:
            async with session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT),
            ) as r:
                if r.status == 200:
                    data = await r.json()
                    choice = data["choices"][0]["message"]
                    content = choice.get("content") or choice.get("reasoning") or ""
                    return {
                        "success": True,
                        "content": content,
                        "usage": data.get("usage", {}),
                        "native_finish": data["choices"][0].get("finish_reason"),
                    }
                err = await r.text()
                print(f"  [{tag or slug}] HTTP {r.status}: {err[:350]}", flush=True)
        except Exception as e:
            print(f"  [{tag or slug}] attempt {attempt+1} exception: {e}", flush=True)
        await asyncio.sleep(RETRY_DELAY * (attempt + 1))
    return {"success": False, "error": f"{slug} max retries"}


async def run_theorist(session, t, query):
    print(f"[theorist start] {t['name']:8} {t['slug']}  effort={t['effort']}", flush=True)
    res = await call_openrouter(
        session, t["slug"], THEORIST_SYSTEM, query,
        MAX_TOKENS_THEORIST, effort=t["effort"], tag=t["name"],
    )
    res["name"] = t["name"]
    res["model"] = t["slug"]
    res["effort"] = t["effort"]
    status = "OK" if res.get("success") else "FAIL"
    ln = len(res.get("content") or "")
    print(f"[theorist done]  {t['name']:8} {status} ({ln} chars, finish={res.get('native_finish')})", flush=True)
    return res


def _titleize(slug):
    return " ".join(w.capitalize() for w in slug.replace("_", "-").split("-"))


def _frontmatter(artifact_type, project, created_by):
    date = datetime.now().strftime("%Y-%m-%d")
    lines = ["---", f"type: {artifact_type}"]
    if project:
        lines.append(f"project: {project}")
    lines.append(f"date: {date}")
    if created_by:
        lines.append(f"created_by: {created_by}")
    lines.append("---")
    return "\n".join(lines) + "\n"


def _nav_header(project, title):
    if project:
        return f"# [[{project}-home|{_titleize(project)}]] — {title}\n*[[dev-hub|Hub]]*\n"
    return f"# {title}\n*[[dev-hub|Hub]]*\n"


async def main():
    ap = argparse.ArgumentParser()
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--query", help="Literal query text.")
    grp.add_argument("--query-file", help="Path to file containing the query.")
    ap.add_argument("--out-dir", required=True, help="Where to write the artifact.")
    ap.add_argument("--name", required=True, help="Artifact filename stem (e.g. playbook-credential-hardening).")
    ap.add_argument("--type", dest="artifact_type", default="playbook",
                    help="Frontmatter type (playbook/research/decision/reference/...).")
    ap.add_argument("--project", default="", help="Frontmatter project slug + project-home backlink target.")
    ap.add_argument("--title", default="", help="H1 title. Defaults to titleized name.")
    args = ap.parse_args()

    if not OPENROUTER_API_KEY:
        print("FATAL: OPENROUTER_API_KEY missing from environment / .env", flush=True)
        sys.exit(2)

    query = args.query or Path(args.query_file).read_text(encoding="utf-8")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"{args.name}.md"
    json_path = out_dir / f"{args.name}.json"
    title = args.title or _titleize(args.name)

    t0 = datetime.now()
    print("=" * 78, flush=True)
    print(f"TOP-TIER COUNCIL: {title}", flush=True)
    for t in THEORISTS:
        print(f"  - {t['name']:8} : {t['slug']:42} effort={t['effort']}", flush=True)
    print(f"  Chairman: {CHAIRMAN_SLUG}  effort={CHAIRMAN_EFFORT}", flush=True)
    print(f"  Out: {md_path}", flush=True)
    print("=" * 78, flush=True)

    async with aiohttp.ClientSession() as session:
        perspectives = await asyncio.gather(
            *[run_theorist(session, t, query) for t in THEORISTS]
        )

    successful = [p for p in perspectives if p.get("success") and p.get("content")]
    print(f"\n[phase 1] {len(successful)}/{len(THEORISTS)} succeeded", flush=True)
    if len(successful) < 2:
        print("FATAL: fewer than 2 theorist responses, aborting", flush=True)
        sys.exit(3)

    print(f"\n[phase 2] chairman synthesizing from {len(successful)} perspectives...\n", flush=True)
    blocks = []
    for p in successful:
        blocks.append(f"\n\n=== {p['name'].upper()} ({p['model']}) ===\n{p['content']}")
    chairman_prompt = (
        f"ORIGINAL QUERY:\n{query}\n\n"
        f"{len(successful)} COUNCIL PERSPECTIVES:\n{''.join(blocks)}\n\n"
        f"---\nProduce the synthesized playbook now. Use clear numbered sections "
        f"and plain-text headings (no parentheses, em-dashes, or colons in ## headings). "
        f"End with a 'Council Disagreements and Verdict' section."
    )

    async with aiohttp.ClientSession() as session:
        chairman = await call_openrouter(
            session, CHAIRMAN_SLUG, CHAIRMAN_SYSTEM, chairman_prompt,
            MAX_TOKENS_CHAIRMAN, effort=CHAIRMAN_EFFORT, tag="chairman",
        )

    synthesis = chairman.get("content") or f"(synthesis failed: {chairman.get('error')})"

    frontmatter = _frontmatter(args.artifact_type, args.project, "llm-council-topfour")
    nav = _nav_header(args.project, title)
    context_block = (
        f"> Synthesis from a 4-model frontier council "
        f"(GPT-5.4 high, Claude Opus 4.7 high, Gemini 3.1 Pro, Grok 4.20 high). "
        f"Chairman: Claude Opus 4.7 (high effort)."
    )
    doc = f"{frontmatter}{nav}\n{context_block}\n\n---\n\n{synthesis}\n"
    md_path.write_text(doc, encoding="utf-8")
    print(f"[saved md]   {md_path}", flush=True)

    raw = {
        "timestamp": t0.isoformat(),
        "elapsed_seconds": (datetime.now() - t0).total_seconds(),
        "query": query,
        "theorists_config": THEORISTS,
        "chairman_model": CHAIRMAN_SLUG,
        "chairman_effort": CHAIRMAN_EFFORT,
        "perspectives": perspectives,
        "synthesis": synthesis,
        "synthesis_success": chairman.get("success", False),
        "synthesis_error": chairman.get("error"),
        "usage": {
            "theorists": [
                {"name": p.get("name"), "model": p.get("model"),
                 "success": p.get("success"), "usage": p.get("usage"),
                 "content_len": len(p.get("content") or "")}
                for p in perspectives
            ],
            "chairman": chairman.get("usage"),
        },
        "artifact": {"path": str(md_path), "type": args.artifact_type,
                     "project": args.project, "name": args.name, "title": title},
    }
    json_path.write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[saved json] {json_path}", flush=True)

    total_cost = 0.0
    for p in perspectives:
        u = p.get("usage") or {}
        total_cost += u.get("cost", 0) or 0
    cu = chairman.get("usage") or {}
    total_cost += cu.get("cost", 0) or 0
    print(f"\n[done] elapsed={raw['elapsed_seconds']:.1f}s  est_cost=${total_cost:.4f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
