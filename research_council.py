#!/usr/bin/env python3
"""
Research Council - Multi-Model LLM Deliberation System
-------------------------------------------------------
Based on Karpathy's llm-council concept, adapted for theoretical synthesis.
Multiple LLMs deliberate independently, then a chairman synthesizes perspectives.

Key difference: No peer-ranking stage - we want additive theoretical perspectives.
"""

import os
import asyncio
import argparse
import json
from pathlib import Path
from typing import Optional
from datetime import datetime

import aiohttp
from dotenv import load_dotenv

# Load environment from .env in same directory as script
SCRIPT_DIR = Path(__file__).parent
load_dotenv(SCRIPT_DIR / '.env')

# === COUNCIL CONFIGURATION ===
# Models can be overridden via .env file
COUNCIL_CONFIG = {
    "anthropic": {
        "model": os.getenv("ANTHROPIC_MODEL", "claude-opus-4-5-20251101"),
        "role": "primary",
        "chairman": True,
        "api_url": "https://api.anthropic.com/v1/messages"
    },
    "openai": {
        "model": os.getenv("OPENAI_MODEL", "gpt-5.2"),
        "role": "secondary",
        "api_url": "https://api.openai.com/v1/chat/completions"
    },
    "openrouter": {
        "model": os.getenv("OPENROUTER_MODEL", "google/gemini-3-pro-preview"),
        "role": "tertiary",
        "api_url": "https://openrouter.ai/api/v1/chat/completions"
    }
}

# API Keys
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
OPENAI_COUNCIL_KEY = os.getenv('OPENAI_COUNCIL_KEY')
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

# === PRICING (per 1M tokens, USD) ===
PRICING = {
    "anthropic": {
        "input": 15.00,   # Claude Opus 4.5
        "output": 75.00
    },
    "openai": {
        "input": 10.00,   # GPT-5.2 estimate
        "output": 30.00
    },
    "openrouter": {
        "input": 2.50,    # Gemini 3 Pro via OpenRouter
        "output": 7.50
    }
}


class ResearchCouncil:
    """
    Multi-model LLM council for theoretical synthesis.

    Three theorists deliberate independently, then the chairman
    synthesizes their perspectives into a unified output.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.config = COUNCIL_CONFIG
        self._validate_api_keys()

    def _validate_api_keys(self):
        """Ensure all required API keys are present."""
        missing = []
        if not ANTHROPIC_API_KEY:
            missing.append("ANTHROPIC_API_KEY")
        if not OPENAI_COUNCIL_KEY:
            missing.append("OPENAI_COUNCIL_KEY")
        if not OPENROUTER_API_KEY:
            missing.append("OPENROUTER_API_KEY")

        if missing:
            raise ValueError(f"Missing API keys: {', '.join(missing)}")

    def _log(self, message: str):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(f"[Council] {message}")

    def _normalize_usage(self, provider: str, usage: dict) -> dict:
        """Normalize usage data to common format."""
        if provider == "anthropic":
            return {
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0)
            }
        else:  # OpenAI and OpenRouter use same format
            return {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0)
            }

    def _calculate_cost(self, provider: str, usage: dict) -> float:
        """Calculate cost in USD for a single API call."""
        normalized = self._normalize_usage(provider, usage)
        pricing = PRICING.get(provider, {"input": 0, "output": 0})

        input_cost = (normalized["input_tokens"] / 1_000_000) * pricing["input"]
        output_cost = (normalized["output_tokens"] / 1_000_000) * pricing["output"]

        return input_cost + output_cost

    def _aggregate_usage(self, perspectives: list[dict], chairman_result: dict = None) -> dict:
        """Aggregate usage across all API calls and calculate total cost."""
        usage_breakdown = []
        total_input = 0
        total_output = 0
        total_cost = 0.0

        # Process theorist perspectives
        for p in perspectives:
            if p.get("success") and p.get("usage"):
                provider = p["provider"]
                usage = p["usage"]
                normalized = self._normalize_usage(provider, usage)
                cost = self._calculate_cost(provider, usage)

                usage_breakdown.append({
                    "provider": provider,
                    "role": p.get("role", "theorist"),
                    "input_tokens": normalized["input_tokens"],
                    "output_tokens": normalized["output_tokens"],
                    "cost_usd": round(cost, 6)
                })

                total_input += normalized["input_tokens"]
                total_output += normalized["output_tokens"]
                total_cost += cost

        # Process chairman synthesis
        if chairman_result and chairman_result.get("success") and chairman_result.get("usage"):
            usage = chairman_result["usage"]
            normalized = self._normalize_usage("anthropic", usage)
            cost = self._calculate_cost("anthropic", usage)

            usage_breakdown.append({
                "provider": "anthropic",
                "role": "chairman",
                "input_tokens": normalized["input_tokens"],
                "output_tokens": normalized["output_tokens"],
                "cost_usd": round(cost, 6)
            })

            total_input += normalized["input_tokens"]
            total_output += normalized["output_tokens"]
            total_cost += cost

        return {
            "breakdown": usage_breakdown,
            "totals": {
                "input_tokens": total_input,
                "output_tokens": total_output,
                "total_tokens": total_input + total_output,
                "cost_usd": round(total_cost, 6)
            }
        }

    async def _call_anthropic(
        self,
        session: aiohttp.ClientSession,
        prompt: str,
        system: str = None,
        max_tokens: int = 4096
    ) -> dict:
        """Call Anthropic Claude API with retry logic."""
        config = self.config["anthropic"]
        headers = {
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }

        payload = {
            "model": config["model"],
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}]
        }
        if system:
            payload["system"] = system

        for attempt in range(MAX_RETRIES):
            try:
                self._log(f"Anthropic attempt {attempt + 1}/{MAX_RETRIES}")
                async with session.post(
                    config["api_url"],
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "provider": "anthropic",
                            "model": config["model"],
                            "role": config["role"],
                            "success": True,
                            "content": data["content"][0]["text"],
                            "usage": data.get("usage", {})
                        }
                    else:
                        error_text = await response.text()
                        self._log(f"Anthropic error {response.status}: {error_text[:200]}")
                        if attempt < MAX_RETRIES - 1:
                            await asyncio.sleep(RETRY_DELAY * (attempt + 1))
            except asyncio.TimeoutError:
                self._log(f"Anthropic timeout on attempt {attempt + 1}")
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY * (attempt + 1))
            except Exception as e:
                self._log(f"Anthropic exception: {str(e)}")
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY * (attempt + 1))

        return {
            "provider": "anthropic",
            "model": config["model"],
            "role": config["role"],
            "success": False,
            "error": "Max retries exceeded",
            "content": None
        }

    async def _call_openai(
        self,
        session: aiohttp.ClientSession,
        prompt: str,
        system: str = None,
        max_tokens: int = 4096
    ) -> dict:
        """Call OpenAI API with retry logic."""
        config = self.config["openai"]
        headers = {
            "Authorization": f"Bearer {OPENAI_COUNCIL_KEY}",
            "Content-Type": "application/json"
        }

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": config["model"],
            "max_completion_tokens": max_tokens,
            "messages": messages
        }

        for attempt in range(MAX_RETRIES):
            try:
                self._log(f"OpenAI attempt {attempt + 1}/{MAX_RETRIES}")
                async with session.post(
                    config["api_url"],
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "provider": "openai",
                            "model": config["model"],
                            "role": config["role"],
                            "success": True,
                            "content": data["choices"][0]["message"]["content"],
                            "usage": data.get("usage", {})
                        }
                    else:
                        error_text = await response.text()
                        self._log(f"OpenAI error {response.status}: {error_text[:200]}")
                        if attempt < MAX_RETRIES - 1:
                            await asyncio.sleep(RETRY_DELAY * (attempt + 1))
            except asyncio.TimeoutError:
                self._log(f"OpenAI timeout on attempt {attempt + 1}")
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY * (attempt + 1))
            except Exception as e:
                self._log(f"OpenAI exception: {str(e)}")
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY * (attempt + 1))

        return {
            "provider": "openai",
            "model": config["model"],
            "role": config["role"],
            "success": False,
            "error": "Max retries exceeded",
            "content": None
        }

    async def _call_openrouter(
        self,
        session: aiohttp.ClientSession,
        prompt: str,
        system: str = None,
        max_tokens: int = 4096
    ) -> dict:
        """Call OpenRouter API (Gemini) with retry logic."""
        config = self.config["openrouter"]
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/research-council",
            "X-Title": "Research Council"
        }

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": config["model"],
            "max_tokens": max_tokens,
            "messages": messages
        }

        for attempt in range(MAX_RETRIES):
            try:
                self._log(f"OpenRouter attempt {attempt + 1}/{MAX_RETRIES}")
                async with session.post(
                    config["api_url"],
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "provider": "openrouter",
                            "model": config["model"],
                            "role": config["role"],
                            "success": True,
                            "content": data["choices"][0]["message"]["content"],
                            "usage": data.get("usage", {})
                        }
                    else:
                        error_text = await response.text()
                        self._log(f"OpenRouter error {response.status}: {error_text[:200]}")
                        if attempt < MAX_RETRIES - 1:
                            await asyncio.sleep(RETRY_DELAY * (attempt + 1))
            except asyncio.TimeoutError:
                self._log(f"OpenRouter timeout on attempt {attempt + 1}")
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY * (attempt + 1))
            except Exception as e:
                self._log(f"OpenRouter exception: {str(e)}")
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY * (attempt + 1))

        return {
            "provider": "openrouter",
            "model": config["model"],
            "role": config["role"],
            "success": False,
            "error": "Max retries exceeded",
            "content": None
        }

    async def _gather_perspectives(
        self,
        prompt: str,
        system: str = None
    ) -> list[dict]:
        """Gather perspectives from all three theorists in parallel."""
        async with aiohttp.ClientSession() as session:
            tasks = [
                self._call_anthropic(session, prompt, system),
                self._call_openai(session, prompt, system),
                self._call_openrouter(session, prompt, system)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Handle any exceptions that weren't caught
            processed = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    provider = ["anthropic", "openai", "openrouter"][i]
                    processed.append({
                        "provider": provider,
                        "success": False,
                        "error": str(result),
                        "content": None
                    })
                else:
                    processed.append(result)

            return processed

    async def _chairman_synthesize(
        self,
        query: str,
        perspectives: list[dict],
        context: str = None
    ) -> dict:
        """Chairman (Claude) synthesizes all perspectives into unified output."""
        # Filter successful responses
        valid_perspectives = [p for p in perspectives if p["success"]]
        failed_providers = [p["provider"] for p in perspectives if not p["success"]]

        if not valid_perspectives:
            return {
                "success": False,
                "error": "All theorists failed to respond",
                "synthesis": None,
                "failed_providers": failed_providers
            }

        # Build synthesis prompt
        perspectives_text = ""
        for p in valid_perspectives:
            perspectives_text += f"\n\n--- {p['provider'].upper()} ({p['role']}) ---\n"
            perspectives_text += p["content"]

        synthesis_prompt = f"""As the Chairman of this Council, synthesize the following
perspectives into a unified, coherent response.

ORIGINAL QUERY:
{query}

{f'CONTEXT PROVIDED:{chr(10)}{context}{chr(10)}' if context else ''}

COUNCIL PERSPECTIVES:
{perspectives_text}

---

Provide a synthesis that:
1. Identifies common threads across perspectives
2. Notes unique insights from each council member
3. Resolves any tensions or contradictions
4. Presents a unified conclusion that incorporates the strongest elements

Structure your synthesis with clear sections."""

        system_prompt = """You are the Chairman of a Council - an expert synthesizer
of diverse perspectives. Your role is to find coherence across viewpoints while
preserving the unique value each perspective brings. Be thorough but concise."""

        async with aiohttp.ClientSession() as session:
            result = await self._call_anthropic(
                session,
                synthesis_prompt,
                system_prompt,
                max_tokens=8192
            )

        return {
            "success": result["success"],
            "synthesis": result.get("content"),
            "error": result.get("error"),
            "failed_providers": failed_providers if failed_providers else None,
            "usage": result.get("usage")
        }

    async def deliberate(
        self,
        query: str,
        context: str = None
    ) -> dict:
        """
        Main deliberation method: all 3 models respond, chairman synthesizes.

        Args:
            query: The question or topic for deliberation
            context: Optional additional context

        Returns:
            dict with perspectives, synthesis, and metadata
        """
        self._log(f"Starting deliberation on: {query[:100]}...")

        # Build theorist prompt
        theorist_prompt = f"""Provide your perspective on the following query.
Draw upon your knowledge to offer unique insights.

QUERY:
{query}

{f'CONTEXT:{chr(10)}{context}' if context else ''}

Provide a substantive, well-reasoned response."""

        system_prompt = """You are a member of a Council of diverse AI perspectives.
Provide your unique, substantive perspective on the query. Be thorough and
don't shy away from nuanced or contrarian positions if warranted."""

        # Gather perspectives in parallel
        self._log("Gathering perspectives from all theorists...")
        perspectives = await self._gather_perspectives(theorist_prompt, system_prompt)

        successful = sum(1 for p in perspectives if p["success"])
        self._log(f"Received {successful}/3 successful responses")

        # Chairman synthesizes
        self._log("Chairman synthesizing perspectives...")
        synthesis_result = await self._chairman_synthesize(query, perspectives, context)

        # Aggregate usage data
        usage = self._aggregate_usage(perspectives, synthesis_result)
        self._log(f"Total tokens: {usage['totals']['total_tokens']}, Cost: ${usage['totals']['cost_usd']:.4f}")

        return {
            "query": query,
            "context": context,
            "perspectives": perspectives,
            "synthesis": synthesis_result.get("synthesis"),
            "success": synthesis_result["success"],
            "failed_providers": synthesis_result.get("failed_providers"),
            "timestamp": datetime.now().isoformat(),
            "usage": usage,
            "config": {k: {"model": v["model"], "role": v["role"]}
                      for k, v in self.config.items()}
        }

    async def query_single(
        self,
        query: str,
        context: str = None,
        model: str = "anthropic"
    ) -> dict:
        """
        Query a single model (no synthesis).

        Args:
            query: The question or topic
            context: Optional additional context
            model: Provider name ('anthropic', 'openai', 'openrouter')

        Returns:
            dict with content, usage, and metadata
        """
        self._log(f"Querying single model ({model}): {query[:100]}...")

        # Build prompt
        prompt = f"""Answer the following query thoughtfully and thoroughly.

QUERY:
{query}

{f'CONTEXT:{chr(10)}{context}' if context else ''}

Provide a substantive, well-reasoned response."""

        system_prompt = """You are a helpful AI assistant. Provide thorough,
accurate, and insightful responses to queries."""

        # Call the appropriate model
        async with aiohttp.ClientSession() as session:
            if model == "anthropic":
                result = await self._call_anthropic(session, prompt, system_prompt)
            elif model == "openai":
                result = await self._call_openai(session, prompt, system_prompt)
            elif model == "openrouter":
                result = await self._call_openrouter(session, prompt, system_prompt)
            else:
                return {
                    "query": query,
                    "success": False,
                    "error": f"Unknown model: {model}"
                }

        # Calculate usage/cost
        usage = None
        if result.get("success") and result.get("usage"):
            normalized = self._normalize_usage(model, result["usage"])
            cost = self._calculate_cost(model, result["usage"])
            usage = {
                "input_tokens": normalized["input_tokens"],
                "output_tokens": normalized["output_tokens"],
                "cost_usd": round(cost, 6)
            }
            self._log(f"Tokens: {normalized['input_tokens']} in, {normalized['output_tokens']} out, Cost: ${cost:.4f}")

        return {
            "query": query,
            "context": context,
            "model": model,
            "content": result.get("content"),
            "success": result.get("success", False),
            "error": result.get("error"),
            "timestamp": datetime.now().isoformat(),
            "usage": usage
        }

    async def deliberate_streaming(
        self,
        query: str,
        context: str = None
    ):
        """
        Streaming version of deliberate - yields progress events.

        Yields dicts with 'event' and 'data' keys for SSE.
        """
        self._log(f"Starting streaming deliberation on: {query[:100]}...")

        # Build theorist prompt
        theorist_prompt = f"""Provide your perspective on the following query.
Draw upon your knowledge to offer unique insights.

QUERY:
{query}

{f'CONTEXT:{chr(10)}{context}' if context else ''}

Provide a substantive, well-reasoned response."""

        system_prompt = """You are a member of a Council of diverse AI perspectives.
Provide your unique, substantive perspective on the query. Be thorough and
don't shy away from nuanced or contrarian positions if warranted."""

        # Yield start event
        yield {"event": "start", "data": {"query": query, "timestamp": datetime.now().isoformat()}}

        # Call each model and yield progress as they complete
        perspectives = []
        providers = ["anthropic", "openai", "openrouter"]

        async with aiohttp.ClientSession() as session:
            # Create tasks for all three
            tasks = {
                "anthropic": asyncio.create_task(self._call_anthropic(session, theorist_prompt, system_prompt)),
                "openai": asyncio.create_task(self._call_openai(session, theorist_prompt, system_prompt)),
                "openrouter": asyncio.create_task(self._call_openrouter(session, theorist_prompt, system_prompt))
            }

            # Wait for each to complete and yield progress
            for provider in providers:
                yield {"event": "model_start", "data": {"provider": provider}}

            # Process as they complete
            pending = set(tasks.values())
            task_to_provider = {v: k for k, v in tasks.items()}

            while pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    provider = task_to_provider[task]
                    try:
                        result = task.result()
                        perspectives.append(result)
                        yield {
                            "event": "model_complete",
                            "data": {
                                "provider": provider,
                                "success": result.get("success", False),
                                "content": result.get("content"),
                                "usage": result.get("usage")
                            }
                        }
                    except Exception as e:
                        perspectives.append({
                            "provider": provider,
                            "success": False,
                            "error": str(e),
                            "content": None
                        })
                        yield {
                            "event": "model_complete",
                            "data": {"provider": provider, "success": False, "error": str(e)}
                        }

        # Chairman synthesis
        yield {"event": "synthesis_start", "data": {}}

        synthesis_result = await self._chairman_synthesize(query, perspectives, context)

        yield {
            "event": "synthesis_complete",
            "data": {
                "synthesis": synthesis_result.get("synthesis"),
                "success": synthesis_result.get("success"),
                "usage": synthesis_result.get("usage")
            }
        }

        # Aggregate usage
        usage = self._aggregate_usage(perspectives, synthesis_result)

        # Final result
        yield {
            "event": "complete",
            "data": {
                "query": query,
                "context": context,
                "perspectives": perspectives,
                "synthesis": synthesis_result.get("synthesis"),
                "success": synthesis_result["success"],
                "failed_providers": synthesis_result.get("failed_providers"),
                "timestamp": datetime.now().isoformat(),
                "usage": usage
            }
        }

    async def theorize_council(
        self,
        concept: str,
        sources: list[dict]
    ) -> dict:
        """
        Academic theory mode with quote extraction and synthesis.

        Args:
            concept: The theoretical concept to explore
            sources: List of dicts with 'title', 'author', 'content' keys

        Returns:
            dict with theoretical analysis sections
        """
        self._log(f"Theorizing on concept: {concept}")

        # Format sources
        sources_text = ""
        for i, source in enumerate(sources, 1):
            sources_text += f"\n\n[Source {i}] {source.get('author', 'Unknown')}: "
            sources_text += f"\"{source.get('title', 'Untitled')}\"\n"
            sources_text += source.get('content', '')[:3000]  # Truncate long sources

        theorist_prompt = f"""Analyze the following concept in light of the provided source material.

CONCEPT: {concept}

SOURCES:
{sources_text}

---

Provide your analysis with these sections:

1. KEY QUOTES: Extract 2-3 directly relevant quotes from the sources (with attribution)

2. CONNECTIONS: Identify connections to ideas, frameworks, or patterns in your knowledge

3. ANALYSIS: Your perspective on how these sources inform the concept

4. GAPS & QUESTIONS: What questions or gaps remain?

Be precise and cite specific passages from the sources."""

        system_prompt = """You are a member of a Council analyzing source material.
Provide substantive analysis that connects the source material with relevant ideas
and frameworks. Be rigorous in your citations."""

        # Gather theorist perspectives
        self._log("Gathering theoretical analyses...")
        perspectives = await self._gather_perspectives(theorist_prompt, system_prompt)

        # Chairman synthesis for source analysis mode
        synthesis_prompt = f"""As Chairman, synthesize these analyses into a unified output.

CONCEPT UNDER ANALYSIS: {concept}

COUNCIL ANALYSES:
"""
        for p in perspectives:
            if p["success"]:
                synthesis_prompt += f"\n\n--- {p['provider'].upper()} ---\n{p['content']}"

        synthesis_prompt += """

---

Create a unified synthesis with:

1. CONSOLIDATED QUOTES: The most significant quotes across all analyses (deduplicated)

2. FRAMEWORK: A coherent framework that integrates the best insights

3. SYNTHESIS: Your unified position on the concept based on all perspectives

4. OPEN QUESTIONS: Key questions or directions for further exploration

Be comprehensive but avoid redundancy."""

        async with aiohttp.ClientSession() as session:
            synthesis_result = await self._call_anthropic(
                session,
                synthesis_prompt,
                "You are the Chairman synthesizer - create a coherent, unified document.",
                max_tokens=8192
            )

        # Aggregate usage data (synthesis_result is raw API result, wrap for aggregation)
        chairman_for_usage = {"success": synthesis_result["success"], "usage": synthesis_result.get("usage")}
        usage = self._aggregate_usage(perspectives, chairman_for_usage)
        self._log(f"Total tokens: {usage['totals']['total_tokens']}, Cost: ${usage['totals']['cost_usd']:.4f}")

        return {
            "concept": concept,
            "sources_count": len(sources),
            "perspectives": perspectives,
            "synthesis": synthesis_result.get("content"),
            "success": synthesis_result["success"],
            "timestamp": datetime.now().isoformat(),
            "usage": usage
        }


async def test_connections(verbose: bool = True) -> dict:
    """Test all API connections with minimal requests."""
    results = {
        "anthropic": {"status": "pending"},
        "openai": {"status": "pending"},
        "openrouter": {"status": "pending"}
    }

    test_prompt = "Respond with exactly: 'Council member active.'"

    async with aiohttp.ClientSession() as session:
        # Test Anthropic
        if verbose:
            print("Testing Anthropic (Claude)...")
        try:
            headers = {
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
            payload = {
                "model": COUNCIL_CONFIG["anthropic"]["model"],
                "max_tokens": 50,
                "messages": [{"role": "user", "content": test_prompt}]
            }
            async with session.post(
                COUNCIL_CONFIG["anthropic"]["api_url"],
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    results["anthropic"] = {
                        "status": "success",
                        "model": COUNCIL_CONFIG["anthropic"]["model"],
                        "response": data["content"][0]["text"][:100]
                    }
                else:
                    error = await resp.text()
                    results["anthropic"] = {"status": "failed", "error": error[:200]}
        except Exception as e:
            results["anthropic"] = {"status": "failed", "error": str(e)}

        # Test OpenAI
        if verbose:
            print("Testing OpenAI (GPT)...")
        try:
            headers = {
                "Authorization": f"Bearer {OPENAI_COUNCIL_KEY}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": COUNCIL_CONFIG["openai"]["model"],
                "max_completion_tokens": 50,
                "messages": [{"role": "user", "content": test_prompt}]
            }
            async with session.post(
                COUNCIL_CONFIG["openai"]["api_url"],
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    results["openai"] = {
                        "status": "success",
                        "model": COUNCIL_CONFIG["openai"]["model"],
                        "response": data["choices"][0]["message"]["content"][:100]
                    }
                else:
                    error = await resp.text()
                    results["openai"] = {"status": "failed", "error": error[:200]}
        except Exception as e:
            results["openai"] = {"status": "failed", "error": str(e)}

        # Test OpenRouter
        if verbose:
            print("Testing OpenRouter (Gemini)...")
        try:
            headers = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/research-council",
                "X-Title": "Research Council"
            }
            payload = {
                "model": COUNCIL_CONFIG["openrouter"]["model"],
                "max_tokens": 50,
                "messages": [{"role": "user", "content": test_prompt}]
            }
            async with session.post(
                COUNCIL_CONFIG["openrouter"]["api_url"],
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    results["openrouter"] = {
                        "status": "success",
                        "model": COUNCIL_CONFIG["openrouter"]["model"],
                        "response": data["choices"][0]["message"]["content"][:100]
                    }
                else:
                    error = await resp.text()
                    results["openrouter"] = {"status": "failed", "error": error[:200]}
        except Exception as e:
            results["openrouter"] = {"status": "failed", "error": str(e)}

    return results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Research Council - Multi-Model LLM Deliberation System"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test all API connections"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Run a deliberation with the given query"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Save results to JSON file"
    )

    args = parser.parse_args()

    if args.test:
        print("=" * 60)
        print("RESEARCH COUNCIL - API Connection Test")
        print("=" * 60)

        # Check API keys first
        print("\nAPI Key Status:")
        keys = [
            ("ANTHROPIC_API_KEY", ANTHROPIC_API_KEY),
            ("OPENAI_COUNCIL_KEY", OPENAI_COUNCIL_KEY),
            ("OPENROUTER_API_KEY", OPENROUTER_API_KEY)
        ]
        all_keys_present = True
        for name, key in keys:
            if key:
                print(f"  [OK] {name}: {key[:12]}...")
            else:
                print(f"  [MISSING] {name}")
                all_keys_present = False

        if not all_keys_present:
            print("\nCannot proceed: Missing API keys in .env")
            return

        print("\nTesting connections...")
        results = asyncio.run(test_connections(verbose=True))

        print("\n" + "=" * 60)
        print("Results:")
        print("=" * 60)

        for provider, result in results.items():
            status = result["status"]
            if status == "success":
                print(f"\n[SUCCESS] {provider.upper()}")
                print(f"  Model: {result.get('model', 'unknown')}")
                print(f"  Response: {result.get('response', 'N/A')}")
            else:
                print(f"\n[FAILED] {provider.upper()}")
                print(f"  Error: {result.get('error', 'Unknown error')}")

        # Summary
        successful = sum(1 for r in results.values() if r["status"] == "success")
        print(f"\n{'=' * 60}")
        print(f"Summary: {successful}/3 providers operational")
        print("=" * 60)

    elif args.query:
        print("Running deliberation...")
        council = ResearchCouncil(verbose=args.verbose)
        result = asyncio.run(council.deliberate(args.query))

        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"Results saved to: {args.output}")
        else:
            print("\n" + "=" * 60)
            print("SYNTHESIS")
            print("=" * 60)
            print(result.get("synthesis", "No synthesis generated"))

            if result.get("failed_providers"):
                print(f"\nNote: Failed providers: {result['failed_providers']}")

            # Show usage stats
            if result.get("usage"):
                usage = result["usage"]
                print("\n" + "-" * 60)
                print("USAGE & COST")
                print("-" * 60)
                for item in usage.get("breakdown", []):
                    print(f"  {item['provider']:12} ({item['role']:10}): "
                          f"{item['input_tokens']:>6} in / {item['output_tokens']:>6} out "
                          f"= ${item['cost_usd']:.4f}")
                totals = usage.get("totals", {})
                print("-" * 60)
                print(f"  {'TOTAL':12} {' ':12}  "
                      f"{totals.get('input_tokens', 0):>6} in / {totals.get('output_tokens', 0):>6} out "
                      f"= ${totals.get('cost_usd', 0):.4f}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
