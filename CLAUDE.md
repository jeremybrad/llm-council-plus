# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LLM Council Plus is a 3-stage deliberation system where multiple LLMs collaboratively answer user questions through:
1. **Stage 1**: Individual model responses (with optional web search context)
2. **Stage 2**: Anonymous peer review/ranking to prevent bias
3. **Stage 3**: Chairman synthesis of collective wisdom

**Key Innovation**: Hybrid architecture supporting OpenRouter (cloud), Ollama (local), Groq (fast inference), direct provider connections, and custom OpenAI-compatible endpoints.

## Running the Application

**Quick Start:**
```bash
./start.sh
```

**Manual Start:**
```bash
# Backend (from project root)
uv run python -m backend.main

# Frontend (in new terminal)
cd frontend
npm run dev
```

**Ports:**
- Backend: `http://localhost:8002` (registered with Mission Control)
- Frontend: `http://localhost:5173`

**Port Selection Process:**
When changing ports, ALWAYS check Mission Control first:
```bash
curl -s http://localhost:3333/api/services | python3 -c "
import json, sys
data = json.load(sys.stdin)
used = sorted(set(p for s in data.get('services',[]) for p in s.get('ports',[])))
print('Ports in use:', used)
"
```
This prevents conflicts with other ecosystem services.

**Network Access:**
```bash
# Backend already listens on 0.0.0.0:8002
# Frontend with network access:
cd frontend && npm run dev -- --host
```

**Installing Dependencies:**
```bash
# Backend
uv sync

# Frontend
cd frontend
npm install
```

**Important**: If switching between Intel/Apple Silicon Macs with iCloud sync:
```bash
rm -rf frontend/node_modules && cd frontend && npm install
```
This fixes binary incompatibilities (e.g., `@rollup/rollup-darwin-*` variants).

## Architecture Overview

### Backend (`backend/`)

**Provider System** (`backend/providers/`)
- **Base**: `base.py` - Abstract interface for all LLM providers
- **Implementations**: `openrouter.py`, `ollama.py`, `groq.py`, `openai.py`, `anthropic.py`, `google.py`, `mistral.py`, `deepseek.py`, `custom_openai.py`
- **Auto-routing**: Model IDs with prefix (e.g., `openai:gpt-4.1`, `ollama:llama3`, `custom:model-name`) route to correct provider
- **Routing logic**: `council.py:get_provider_for_model()` handles prefix parsing

**Core Modules**

| Module | Purpose |
|--------|---------|
| `council.py` | Orchestration: stage1/2/3 collection, rankings, title generation |
| `search.py` | Web search: DuckDuckGo, Tavily, Brave with Jina Reader content fetch |
| `settings.py` | Config management, persisted to `data/settings.json` |
| `prompts.py` | Default system prompts for all stages |
| `main.py` | FastAPI app with streaming SSE endpoint |
| `storage.py` | Conversation persistence in `data/conversations/{id}.json` |

### Frontend (`frontend/src/`)

| Component | Purpose |
|-----------|---------|
| `App.jsx` | Main orchestration, SSE streaming, conversation state |
| `ChatInterface.jsx` | User input, web search toggle, execution mode |
| `Stage1.jsx` | Tab view of individual model responses |
| `Stage2.jsx` | Peer rankings with de-anonymization, aggregate scores |
| `Stage3.jsx` | Chairman synthesis (final answer) |
| `CouncilGrid.jsx` | Visual grid of council members with provider icons |
| `Settings.jsx` | 5-section settings: LLM API Keys, Council Config, System Prompts, Search Providers, Backup & Reset |
| `Sidebar.jsx` | Conversation list with inline delete confirmation |
| `SearchableModelSelect.jsx` | Searchable dropdown for model selection |

**Styling**: "Midnight Glass" dark theme with glassmorphic effects. Primary colors: blue (#3b82f6) and cyan (#06b6d4) gradients. Font: Merriweather 15px/1.7 for content, JetBrains Mono for errors.

## Critical Implementation Details

### Python Module Imports
**ALWAYS** use relative imports in backend modules:
```python
from .config import ...
from .council import ...
```
**NEVER** use absolute imports like `from backend.config import ...`

**Run backend as module** from project root:
```bash
uv run python -m backend.main  # Correct
cd backend && python main.py  # WRONG - breaks imports
```

### Model ID Prefix Format
```
openrouter:anthropic/claude-sonnet-4  → Cloud via OpenRouter
ollama:llama3.1:latest                → Local via Ollama
groq:llama3-70b-8192                  → Fast inference via Groq
openai:gpt-4.1                        → Direct OpenAI connection
anthropic:claude-sonnet-4             → Direct Anthropic connection
custom:model-name                     → Custom OpenAI-compatible endpoint
```

### Model Name Display Helper
Use this pattern in Stage components to handle both `/` and `:` delimiters:
```jsx
const getShortModelName = (modelId) => {
  if (!modelId) return 'Unknown';
  if (modelId.includes('/')) return modelId.split('/').pop();
  if (modelId.includes(':')) return modelId.split(':').pop();
  return modelId;
};
```

### Provider Icon Detection (CouncilGrid.jsx)
Check prefixes FIRST before name-based detection to avoid mismatches:
```jsx
const getProviderInfo = (modelId) => {
    const id = modelId.toLowerCase();
    // Check prefixes FIRST (order matters!)
    if (id.startsWith('custom:')) return PROVIDER_CONFIG.custom;
    if (id.startsWith('ollama:')) return PROVIDER_CONFIG.ollama;
    if (id.startsWith('groq:')) return PROVIDER_CONFIG.groq;
    // Then check name-based patterns...
};
```

### Stage 2 Ranking Format
The prompt enforces strict format for parsing:
```
1. Individual evaluations
2. Blank line
3. "FINAL RANKING:" header (all caps, with colon)
4. Numbered list: "1. Response C", "2. Response A", etc.
```
Fallback regex extracts "Response X" patterns if format not followed.

### Streaming & Abort Logic
- Backend checks `request.is_disconnected()` inside loops
- Frontend aborts via AbortController signal
- **Critical**: Always inject raw `Request` object into streaming endpoints (Pydantic models lack `is_disconnected()`)

### ReactMarkdown Safety
```jsx
<div className="markdown-content">
  <ReactMarkdown>
    {typeof content === 'string' ? content : String(content || '')}
  </ReactMarkdown>
</div>
```
Always wrap in `.markdown-content` div and ensure string type (some providers return arrays/objects).

### Tab Bounds Safety
In Stage1/Stage2, auto-adjust activeTab when out of bounds during streaming:
```jsx
useEffect(() => {
  if (activeTab >= responses.length && responses.length > 0) {
    setActiveTab(responses.length - 1);
  }
}, [responses.length]);
```

## Common Gotchas

1. **Port Conflicts**: Backend uses 8002. Check Mission Control (`curl http://localhost:3333/api/services`) before changing ports. Update `backend/main.py` and `frontend/src/api.js` together.

2. **CORS Errors**: Frontend origins must match `main.py` CORS middleware (localhost:5173 and :3000).

3. **Missing Metadata**: `label_to_model` and `aggregate_rankings` are ephemeral - only in API responses, not stored.

4. **Duplicate Tabs**: Use immutable state updates (spread operator), not mutations. StrictMode runs effects twice.

5. **Search Rate Limits**: DuckDuckGo can rate-limit. Retry logic in `search.py` handles this.

6. **Jina Reader 451 Errors**: Many news sites block AI scrapers. Use Tavily/Brave or set `full_content_results` to 0.

7. **Model Deduplication**: When multiple sources provide same model, use Map-based deduplication preferring direct connections.

8. **Binary Dependencies**: `node_modules` in iCloud can break between Mac architectures. Delete and reinstall.

9. **Custom Endpoint Icons**: Models from custom endpoints may match name patterns (e.g., "claude"). Check `custom:` prefix first.

## Data Flow

```
User Query (+ optional web search)
    ↓
[Web Search: DuckDuckGo/Tavily/Brave + Jina Reader]
    ↓
Stage 1: Parallel queries → Stream individual responses
    ↓
Stage 2: Anonymize → Parallel peer rankings → Parse rankings
    ↓
Calculate aggregate rankings
    ↓
Stage 3: Chairman synthesis → Stream final answer
    ↓
Save conversation (stage1, stage2, stage3 only)
```

## Execution Modes

Three modes control deliberation depth:
- **Chat Only**: Stage 1 only (quick responses)
- **Chat + Ranking**: Stages 1 & 2 (peer review without synthesis)
- **Full Deliberation**: All 3 stages (default)

## Testing & Debugging

```bash
# Check Ollama models
curl http://localhost:11434/api/tags

# Test custom endpoint
curl https://your-endpoint.com/v1/models -H "Authorization: Bearer $API_KEY"

# View logs
# Watch terminal running backend/main.py
```

## Web Search

**Providers**: DuckDuckGo (free), Tavily (API), Brave (API)

**Full Content Fetching**: Jina Reader (`https://r.jina.ai/{url}`) extracts article text for top N results (configurable 0-10, default 3). Falls back to summary if fetch fails or yields <500 chars. 25-second timeout per article, 60-second total search budget.

**Search Query Processing**:
- **Direct** (default): Send exact query to search engine
- **YAKE**: Extract keywords first (useful for long prompts)

## Settings

**UI Sections** (sidebar navigation):
1. **LLM API Keys**: OpenRouter, Groq, Ollama, Direct providers, Custom endpoint
2. **Council Config**: Model selection with Remote/Local toggles, temperature controls, "I'm Feeling Lucky" randomizer
3. **System Prompts**: Stage 1/2/3 prompts with reset-to-default
4. **Search Providers**: DuckDuckGo, Tavily, Brave + Jina full content settings
5. **Backup & Reset**: Import/Export config, reset to defaults

**Auto-Save Behavior**:
- **Credentials auto-save**: API keys and URLs save immediately on successful test
- **Configs require manual save**: Model selections, prompts, temperatures
- UX flow: Test → Success → Auto-save → Clear input → "Settings saved!"

**Temperature Controls**:
- Council Heat: Stage 1 creativity (default: 0.5)
- Chairman Heat: Stage 3 synthesis (default: 0.4)
- Stage 2 Heat: Peer ranking consistency (default: 0.3)

**Rate Limit Warnings**:
- Formula: `(council_members × 2) + 2` requests per council run
- OpenRouter free tier: 20 RPM, 50 requests/day
- Groq: 30 RPM, 14,400 requests/day

**Storage**: `data/settings.json`

## Design Principles

- **Graceful Degradation**: Single model failure doesn't block entire council
- **Transparency**: All raw outputs inspectable via tabs
- **De-anonymization**: Models receive "Response A/B/C", frontend displays real names
- **Progress Indicators**: "X/Y completed" during streaming
- **Provider Flexibility**: Mix cloud, local, and custom endpoints freely

## Code Safety Guidelines

**Communication:**
- NEVER make assumptions when requirements are vague - ask for clarification
- Provide options with pros/cons for different approaches
- Confirm understanding before significant changes

**Code Safety:**
- NEVER use placeholders like `// ...` in edits - this deletes code
- Always provide full content when writing/editing files
- FastAPI: Inject raw `Request` object to access `is_disconnected()`
- React: Use spread operators for immutable state updates (StrictMode runs effects twice)

## OpenAI-Compatible Endpoint (OpenWebUI Integration)

The backend provides an OpenAI-compatible API at `/v1/chat/completions` that allows Roundtable to be used from OpenWebUI and other OpenAI-compatible clients.

### Configuration in OpenWebUI

1. **Add as Connection**:
   - Go to Admin Settings → Connections
   - Add new OpenAI-compatible endpoint:
     - URL: `http://localhost:8002/v1`
     - API Key: Any non-empty string (not validated)

2. **Select Model**:
   - After adding the connection, these models appear:
     - `roundtable` - Default (3 rounds)
     - `roundtable:fast` - Quick mode (1 round)
     - `roundtable:thorough` - Standard (3 rounds)

### Model Variants

| Model String | Rounds | Use Case |
|-------------|--------|----------|
| `roundtable` | 3 | Default deliberation |
| `roundtable:fast` | 1 | Quick responses |
| `roundtable:thorough` | 3 | Standard depth |
| `roundtable:deep` | 5 | Extended deliberation |
| `roundtable:N` | N | Custom round count |

### Request Parameters (via extra_body)

```python
# In OpenAI Python client:
response = client.chat.completions.create(
    model="roundtable:fast",
    messages=[{"role": "user", "content": "Your question"}],
    extra_body={
        "council_models": ["ollama:llama3.2", "ollama:mistral"],
        "num_rounds": 2,
        "chair_model": "ollama:llama3.2",
        "max_parallel": 2,
        "timeout_seconds": 120
    }
)
```

### Streaming Progress

During streaming, progress updates show:
```
[Roundtable Starting: 3 agents, 3 rounds]

**Round 1: Opening**
  - Builder responded
  - Skeptic responded
  - Contrarian responded

**Round 2: Critique**
...
**Chair Final Synthesis...**

---
[Final synthesized answer appears here]
```

### Testing the Endpoint

```bash
# List available models
curl http://localhost:8002/v1/models

# Test chat completion (non-streaming)
curl -X POST http://localhost:8002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "roundtable:fast",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "stream": false
  }'

# Streaming response
curl -X POST http://localhost:8002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "roundtable",
    "messages": [{"role": "user", "content": "Explain async/await"}],
    "stream": true
  }'
```

### Requirements

- Council must have at least 2 models configured in settings
- Models must be accessible (Ollama running, API keys configured)

## Night Shift Runner (Scheduled Jobs)

The Night Shift runner provides a CLI for running overnight batch jobs with safety gates.

### Safety Gates

1. **Repo Root Verification**: `--expected-repo-root` is required and must match the actual git root
2. **Preflight Mode**: Default behavior shows what would run; add `--go` to actually execute
3. **Budget Enforcement**: Configurable limits on tasks, tokens, spend, and timeout

### Usage

```bash
# List available jobs
python -m backend.nightshift list

# Preflight check (default - shows what would happen)
python -m backend.nightshift run \
  --job=repo_docs_refresh \
  --expected-repo-root=/path/to/llm-council-plus

# Actually execute
python -m backend.nightshift run \
  --job=repo_docs_refresh \
  --expected-repo-root=/path/to/llm-council-plus \
  --go

# With budget limits
python -m backend.nightshift run \
  --job=repo_docs_refresh \
  --expected-repo-root=/path/to/llm-council-plus \
  --go \
  --max-tasks=50 \
  --timeout=1800
```

### Available Jobs

| Job | Description |
|-----|-------------|
| `repo_docs_refresh` | Scan docs for broken links, TODOs, and generate health report |

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Error during execution |
| 2 | Preflight only (no --go) or safety gate failed |

### Cron Scheduling

```bash
# Edit crontab
crontab -e

# Run docs refresh every night at 2 AM
0 2 * * * cd /path/to/llm-council-plus && .venv/bin/python -m backend.nightshift run --job=repo_docs_refresh --expected-repo-root=/path/to/llm-council-plus --go >> /var/log/nightshift.log 2>&1

# Run weekly on Sundays at 3 AM with email notification
0 3 * * 0 cd /path/to/llm-council-plus && .venv/bin/python -m backend.nightshift run --job=repo_docs_refresh --expected-repo-root=/path/to/llm-council-plus --go && mail -s "Night Shift Complete" you@example.com < reports/$(date +\%Y-\%m-\%d)_docs_health.md
```

### Output

Reports are generated in the `reports/` directory:
- `YYYY-MM-DD_docs_health.md` - Human-readable markdown report
- `YYYY-MM-DD_docs_health.json` - Machine-readable JSON report

## Future Enhancements

- Model performance analytics over time
- Export conversations to markdown/PDF
- Custom ranking criteria (beyond accuracy/insight)
- Backend caching for repeated queries
- Multiple custom endpoints support
