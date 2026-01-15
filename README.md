# LLM Council Redux

A multi-model LLM deliberation system. Three AI models independently respond to a query, then a Chairman synthesizes their perspectives into a unified output.

Built on [Andrej Karpathy's llm-council](https://github.com/karpathy/llm-council) concept, with significant extensions.

## What's Different from Karpathy's Original

| Feature | Karpathy's | This Project |
|---------|------------|--------------|
| **Architecture** | React + FastAPI | Flask + embedded HTML (single file frontend) |
| **Peer Review Stage** | Models rank each other's responses | Removed - we want additive perspectives, not competition |
| **Theorize Mode** | Not available | Analyze source texts with quote extraction |
| **History** | Conversation-based | Persistent query history with reload |
| **Cost Tracking** | Not available | Per-query token counts and cost estimates |
| **Streaming** | Not available | SSE streaming shows progress as models respond |
| **File Upload** | Not available | PDF/TXT/MD upload for source analysis |

## Why Use It

- **Reduce single-model bias** - Different models have different training, perspectives, and blind spots
- **Get diverse viewpoints** - Each model may surface insights the others miss
- **Synthesis > averaging** - The Chairman doesn't just merge; it resolves tensions and identifies the strongest elements

## Two Modes

### Deliberate
General-purpose multi-perspective analysis.

**Input:** Query + optional context
**Output:** 3 perspectives + unified synthesis

### Theorize
Multi-model analysis of source material.

**Input:** Concept + source texts (paste or upload PDF/TXT)
**Output:** Quote extraction, connections, analysis, open questions + synthesis

## Quick Start

```bash
# Setup
cp .env.example .env  # Add your API keys
pip install -r requirements.txt

# Run
python council_server.py
```

Open http://localhost:5000

## Configuration

Edit `.env`:

```bash
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_COUNCIL_KEY=sk-proj-...
OPENROUTER_API_KEY=sk-or-v1-...

# Optional: override default models
# ANTHROPIC_MODEL=claude-opus-4-5-20251101
# OPENAI_MODEL=gpt-5.2
# OPENROUTER_MODEL=google/gemini-3-pro-preview
```

## Default Models

| Role | Model | Provider |
|------|-------|----------|
| Primary + Chairman | Claude Opus 4.5 | Anthropic |
| Secondary | GPT-5.2 | OpenAI |
| Tertiary | Gemini 3 Pro | OpenRouter |

## API

```
POST /api/deliberate       {"query": "...", "context": "..."}
POST /api/deliberate/stream  (SSE streaming version)
POST /api/theorize         {"concept": "...", "sources": "..."}
POST /api/upload           multipart/form-data file upload
GET  /api/test             Test all API connections
GET  /api/history          List past queries
GET  /api/history/<id>     Reload a past result
```

## License

MIT
