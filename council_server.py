#!/usr/bin/env python3
"""
Council Server - Flask backend for Research Council frontend
------------------------------------------------------------
Simple REST API to run deliberations and theorize operations.
"""

import asyncio
import json
import tempfile
import os
import uuid
import re
import requests
from datetime import datetime
from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS
from pathlib import Path
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

from research_council import ResearchCouncil, test_connections

# Load environment variables
SCRIPT_DIR = Path(__file__).parent
load_dotenv(SCRIPT_DIR / '.env')

# API key for naming (uses OpenAI for speed/cost)
OPENAI_API_KEY = os.getenv('OPENAI_COUNCIL_KEY')

# History file path
HISTORY_FILE = Path(__file__).parent / 'council_history.json'


def load_history():
    """Load history from JSON file."""
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                print(f"[History] Loaded {len(data)} entries from {HISTORY_FILE}")
                return data
        except json.JSONDecodeError as e:
            print(f"[History] JSON parse error: {e}")
            return []
        except Exception as e:
            print(f"[History] Error loading history: {e}")
            return []
    else:
        print(f"[History] File not found: {HISTORY_FILE}")
    return []


def save_history(history):
    """Save history to JSON file."""
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def generate_entry_name(query: str, mode: str = 'deliberate') -> str:
    """
    Generate a short, descriptive name for a history entry using GPT-4o-mini.
    Falls back to extracting first few words if LLM call fails.
    """
    # Fallback: extract meaningful first words
    def fallback_name(text: str) -> str:
        # Remove common prefixes and clean up
        text = re.sub(r'^(given|based on|considering|analyze|explain|what|how|why|describe)\s+', '', text.lower())
        # Get first ~6 words
        words = text.split()[:6]
        name = ' '.join(words)
        if len(name) > 50:
            name = name[:47] + '...'
        return name.title() if name else 'Untitled Query'

    if not OPENAI_API_KEY:
        return fallback_name(query)

    try:
        response = requests.post(
            'https://api.openai.com/v1/chat/completions',
            headers={
                'Authorization': f'Bearer {OPENAI_API_KEY}',
                'Content-Type': 'application/json'
            },
            json={
                'model': 'gpt-4o-mini',
                'messages': [
                    {
                        'role': 'system',
                        'content': 'Generate a concise 3-6 word title that captures the essence of this query. No quotes, no punctuation at the end. Just the title.'
                    },
                    {
                        'role': 'user',
                        'content': query[:500]  # Limit input length
                    }
                ],
                'max_tokens': 20,
                'temperature': 0.3
            },
            timeout=5  # Quick timeout for naming
        )

        if response.status_code == 200:
            name = response.json()['choices'][0]['message']['content'].strip()
            # Clean up any quotes or extra punctuation
            name = name.strip('"\'').rstrip('.')
            if name:
                print(f"[History] Generated name: {name}")
                return name
    except Exception as e:
        print(f"[History] Name generation failed: {e}")

    return fallback_name(query)


def add_to_history(result, mode='deliberate'):
    """Add a run result to history."""
    history = load_history()

    # Get the query text
    query_text = result.get('query') or result.get('concept', '')

    # Generate a descriptive name for this entry
    entry_name = generate_entry_name(query_text, mode)

    # Create history entry
    entry = {
        'id': str(uuid.uuid4())[:8],
        'name': entry_name,  # Auto-generated descriptive name
        'mode': mode,
        'timestamp': result.get('timestamp', datetime.now().isoformat()),
        'query': query_text,
        'context': result.get('context'),
        'sources_count': result.get('sources_count'),
        'success': result.get('success', False),
        'usage': result.get('usage'),
        'result': result  # Full result for reload
    }

    # Add to beginning (newest first)
    history.insert(0, entry)

    # Keep last 100 entries
    history = history[:100]

    save_history(history)
    return entry['id']

# PDF parsing
try:
    import fitz  # PyMuPDF
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("Warning: PyMuPDF not installed. PDF upload disabled. Run: pip install PyMuPDF")

app = Flask(__name__)
CORS(app)

# File upload config
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'md', 'markdown'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text_from_file(filepath, filename):
    """Extract text content from uploaded file."""
    ext = filename.rsplit('.', 1)[1].lower()

    if ext in ('txt', 'md', 'markdown'):
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()

    elif ext == 'pdf':
        if not PDF_SUPPORT:
            raise ValueError("PDF support not available. Install PyMuPDF: pip install PyMuPDF")

        text_parts = []
        with fitz.open(filepath) as doc:
            for page_num, page in enumerate(doc, 1):
                text = page.get_text()
                if text.strip():
                    text_parts.append(f"[Page {page_num}]\n{text}")

        return "\n\n".join(text_parts)

    else:
        raise ValueError(f"Unsupported file type: {ext}")


# Initialize council
council = ResearchCouncil(verbose=True)


def run_async(coro):
    """Helper to run async functions in sync Flask context."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@app.route('/')
def index():
    """Serve the frontend HTML."""
    frontend_path = Path(__file__).parent / 'council_frontend.html'
    return send_file(frontend_path)


@app.route('/api/test', methods=['GET'])
def api_test():
    """Test all API connections."""
    results = run_async(test_connections(verbose=False))
    return jsonify(results)


@app.route('/api/deliberate', methods=['POST'])
def api_deliberate():
    """Run a deliberation with the council."""
    data = request.get_json()

    if not data or 'query' not in data:
        return jsonify({"error": "Missing 'query' field"}), 400

    query = data['query']
    context = data.get('context')

    try:
        result = run_async(council.deliberate(query, context))
        # Auto-save to history
        history_id = add_to_history(result, mode='deliberate')
        result['history_id'] = history_id
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/query', methods=['POST'])
def api_single_query():
    """Query a single model (no synthesis)."""
    data = request.get_json()

    if not data or 'query' not in data:
        return jsonify({"error": "Missing 'query' field"}), 400

    if 'model' not in data:
        return jsonify({"error": "Missing 'model' field"}), 400

    query = data['query']
    context = data.get('context')
    model = data['model']

    if model not in ['anthropic', 'openai', 'openrouter']:
        return jsonify({"error": f"Invalid model: {model}"}), 400

    try:
        result = run_async(council.query_single(query, context, model))
        # Auto-save to history with model name
        history_id = add_to_history(result, mode=f'single-{model}')
        result['history_id'] = history_id
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/deliberate/stream', methods=['POST'])
def api_deliberate_stream():
    """Run a deliberation with SSE streaming progress."""
    data = request.get_json()

    if not data or 'query' not in data:
        return jsonify({"error": "Missing 'query' field"}), 400

    query = data['query']
    context = data.get('context')

    def generate():
        """Generator that yields SSE events."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        final_result = None

        try:
            async def stream_events():
                async for event in council.deliberate_streaming(query, context):
                    yield event

            # Create the async generator
            async_gen = stream_events()

            # Iterate over events
            while True:
                try:
                    event = loop.run_until_complete(async_gen.__anext__())
                    event_type = event.get("event", "message")
                    event_data = event.get("data", {})

                    # Capture final result for history
                    if event_type == "complete":
                        final_result = event_data

                    yield f"event: {event_type}\ndata: {json.dumps(event_data)}\n\n"
                except StopAsyncIteration:
                    break

            # Save to history after stream completes
            if final_result:
                history_id = add_to_history(final_result, mode='deliberate')
                yield f"event: history_saved\ndata: {json.dumps({'history_id': history_id})}\n\n"
        finally:
            loop.close()

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'
        }
    )


@app.route('/api/theorize', methods=['POST'])
def api_theorize():
    """Run theorize_council with sources."""
    data = request.get_json()

    if not data or 'concept' not in data:
        return jsonify({"error": "Missing 'concept' field"}), 400

    concept = data['concept']
    sources = data.get('sources', [])

    # Parse sources if provided as text blocks
    if isinstance(sources, str):
        # Split by double newlines and create source objects
        blocks = [b.strip() for b in sources.split('\n\n') if b.strip()]
        sources = [{"content": block, "title": f"Source {i+1}", "author": "Unknown"}
                   for i, block in enumerate(blocks)]

    try:
        result = run_async(council.theorize_council(concept, sources))
        # Auto-save to history
        history_id = add_to_history(result, mode='theorize')
        result['history_id'] = history_id
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/config', methods=['GET'])
def api_config():
    """Get current council configuration."""
    from research_council import COUNCIL_CONFIG
    return jsonify(COUNCIL_CONFIG)


@app.route('/api/history', methods=['GET'])
def api_history_list():
    """Get list of history entries (metadata only)."""
    history = load_history()
    # Return just metadata, not full results
    entries = []
    for h in history:
        # Backward compatibility: generate name for older entries without one
        name = h.get('name')
        if not name:
            query = h.get('query', '')
            # Simple fallback for older entries
            words = query.split()[:5]
            name = ' '.join(words)
            if len(name) > 40:
                name = name[:37] + '...'
            name = name.title() if name else 'Untitled'

        entries.append({
            'id': h['id'],
            'name': name,  # Auto-generated descriptive name
            'mode': h['mode'],
            'timestamp': h['timestamp'],
            'query': h['query'][:100] + ('...' if len(h['query']) > 100 else ''),
            'success': h['success'],
            'cost': h.get('usage', {}).get('totals', {}).get('cost_usd')
        })
    return jsonify(entries)


@app.route('/api/history/<history_id>', methods=['GET'])
def api_history_get(history_id):
    """Get a specific history entry with full result."""
    history = load_history()
    for h in history:
        if h['id'] == history_id:
            return jsonify(h['result'])
    return jsonify({"error": "History entry not found"}), 404


@app.route('/api/history/<history_id>', methods=['DELETE'])
def api_history_delete(history_id):
    """Delete a history entry."""
    history = load_history()
    history = [h for h in history if h['id'] != history_id]
    save_history(history)
    return jsonify({"success": True})


@app.route('/api/upload', methods=['POST'])
def api_upload():
    """Upload and parse a document file."""
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "error": f"File type not supported. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        }), 400

    # Check file size
    file.seek(0, 2)  # Seek to end
    size = file.tell()
    file.seek(0)  # Reset to beginning

    if size > MAX_FILE_SIZE:
        return jsonify({
            "error": f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB"
        }), 400

    try:
        # Save to temp file for processing
        filename = secure_filename(file.filename)
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        # Extract text
        text = extract_text_from_file(tmp_path, filename)

        # Clean up temp file
        os.unlink(tmp_path)

        return jsonify({
            "success": True,
            "filename": filename,
            "text": text,
            "length": len(text)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    print("=" * 60)
    print("RESEARCH COUNCIL SERVER")
    print("=" * 60)
    print("Starting server at http://localhost:5000")
    print("Frontend at http://localhost:5000/")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=True)
