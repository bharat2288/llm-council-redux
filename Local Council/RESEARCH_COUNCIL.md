# Council

A multi-model LLM deliberation system. Three AI models independently respond to a query, then a Chairman synthesizes their perspectives into a unified output.

## Why Use It

- **Reduce single-model bias** - Different models have different training, perspectives, and blind spots
- **Get diverse viewpoints** - Each model may surface insights the others miss
- **Synthesis > averaging** - The Chairman doesn't just merge; it resolves tensions and identifies the strongest elements
- **Works for any query** - From theoretical research to practical decisions to creative brainstorming

## Models (Default)

| Role | Model | Provider |
|------|-------|----------|
| Primary + Chairman | Claude Opus 4.5 | Anthropic |
| Secondary | GPT-5.2 | OpenAI |
| Tertiary | Gemini 3 Pro | OpenRouter |

Models are configurable via `.env` file.

## Two Modes

### Deliberate
General-purpose multi-perspective analysis.

**Input:** Query + optional context
**Output:** 3 perspectives + unified synthesis

**Use for:** Any question where diverse viewpoints add value - research questions, decisions, analysis, brainstorming, complex problems.

### Theorize (with sources)
Multi-model analysis of source material.

**Input:** Concept + source texts
**Output:** Quote extraction, connections, analysis, open questions + synthesis

**Use for:** When you have specific text/sources you want multiple models to analyze and synthesize perspectives on.

---

## Running the Council

### Web UI
```bash
python council_server.py
# Open http://localhost:5000
```

### CLI
```bash
# Test connections
python research_council.py --test

# Run deliberation
python research_council.py --query "Your question here" -v

# Save to file
python research_council.py --query "Your question" -o output.json
```

### API
```
POST /api/deliberate  {"query": "...", "context": "..."}
POST /api/theorize    {"concept": "...", "sources": "..."}
GET  /api/test        Test all connections
GET  /api/config      Current model config
```

---

## Configuration

Edit `.env` to change models:

```bash
# Models (uncomment to override defaults)
ANTHROPIC_MODEL=claude-opus-4-5-20251101
OPENAI_MODEL=gpt-5.2
OPENROUTER_MODEL=google/gemini-3-pro-preview
```

---

## COUNCIL Macro

Use this format when chatting with another LLM to generate Council-ready queries.

### Invocation

```
COUNCIL: [your topic or question]
```

### What the LLM Should Produce

When you invoke `COUNCIL: [topic]`, the LLM generates a structured query optimized for multi-model deliberation:

```
COUNCIL QUERY
=============

QUERY:
[Clear, well-formed question or topic statement]

CONTEXT (optional):
[Any relevant background, constraints, or framing]

---
Paste into Council UI or use with API.
```

### System Instruction for COUNCIL Macro

Add this to your custom instructions or system prompt in other LLM tools:

```
When the user invokes "COUNCIL: [topic]", generate a structured query for
a multi-model deliberation system. The system sends the query to 3 different
LLMs (Claude, GPT, Gemini) who respond independently, then synthesizes their
perspectives.

Format your response as:

COUNCIL QUERY
=============

QUERY:
[Transform the user's topic into a clear, specific question or analysis prompt.
 If the topic is vague, make it concrete. If it's a decision, frame it as
 "What are the key considerations for..." If it's exploratory, frame it as
 "Analyze [X] considering [relevant dimensions]..."]

CONTEXT (optional):
[Add any context that would help the models give better responses - domain,
 constraints, what kind of output is most useful, etc. Leave blank if none needed.]

---
Ready for Council deliberation.

Guidelines:
- Make the query specific enough to get substantive responses
- Frame it to invite diverse perspectives (not yes/no questions)
- Include context that shapes relevance
- The query should work for any domain - technical, creative, strategic, etc.
```

### Examples

**Simple:**
```
User: COUNCIL: should I use React or Vue for a small project

Response:
COUNCIL QUERY
=============

QUERY:
What are the key factors to consider when choosing between React and Vue.js
for a small web project? Compare their learning curves, ecosystem maturity,
performance characteristics, and suitability for solo developers or small teams.

CONTEXT:
Small project scope - likely a single developer, needs to ship relatively quickly.
Interested in practical trade-offs, not just feature comparisons.

---
Ready for Council deliberation.
```

**Research:**
```
User: COUNCIL: relationship between attention and consciousness

Response:
COUNCIL QUERY
=============

QUERY:
Analyze the theoretical relationship between attention and consciousness.
Consider: (1) Are attention and consciousness dissociable? (2) What do
theories like Global Workspace, Integrated Information, and Higher-Order
theories say about their relationship? (3) What empirical evidence bears
on whether attention is necessary/sufficient for consciousness?

CONTEXT:
Looking for a synthesis across cognitive science, philosophy of mind, and
neuroscience perspectives. Interested in both theoretical arguments and
empirical findings.

---
Ready for Council deliberation.
```

**Practical:**
```
User: COUNCIL: how to structure a startup equity split among 3 co-founders

Response:
COUNCIL QUERY
=============

QUERY:
What frameworks and considerations should guide equity allocation among
three co-founders of an early-stage startup? Address: role-based allocation,
vesting structures, handling future contributions vs. past contributions,
and common pitfalls that lead to co-founder disputes.

CONTEXT:
Pre-funding stage, no revenue yet. Founders have different roles (technical,
business, domain expert). Looking for practical guidance, not just theory.

---
Ready for Council deliberation.
```

---

## Cost Tracking

Each query shows usage breakdown:
- Token counts per model (input/output)
- Estimated cost per model
- Total cost for the deliberation

Pricing estimates (per 1M tokens):
| Provider | Input | Output |
|----------|-------|--------|
| Anthropic (Opus 4.5) | $15 | $75 |
| OpenAI (GPT-5.2) | $10 | $30 |
| OpenRouter (Gemini 3 Pro) | $2.50 | $7.50 |

---

## Files

| File | Purpose |
|------|---------|
| `research_council.py` | Core logic - API calls, synthesis |
| `council_server.py` | Flask server for web UI |
| `council_frontend.html` | Web interface |
| `.env` | API keys + model config |
