# 🏡 The Front Porch Exchange

Rural isolation is real. In towns like Greenville, Texas, many neighbors live close together but rarely meet. The Front Porch Exchange helps **any adult with a craft or domestic skill** (crochet, mending, baking…) host a free, **90-minute, intergenerational porch gathering**.

A host has a gentle chat with a guide, one question at a time (a volunteer coordinator can type for them). The app then **matches the host with neighbors in a small community database** who have said they want to learn or join. A 3-agent pipeline builds a printable **Porch Gathering Kit** with:

- an exact 90-minute schedule
- a $0-to-host supply checklist
- a non-alcoholic appetizer-and-drink pairing chosen from Ina Garten's Barefoot Contessa recipes (ingredients, a short method, and a link to the full recipe)
- 4 Porch Story Cards (conversation prompts)
- 4 community safety rules
- a personal invitation for each matched neighbor

The host never has to post flyers around town.

---

## Setup (about 5 minutes)

You need **Python 3.10 or newer** ([python.org](https://www.python.org/downloads/)) and an API key.

> **First, cd into the project folder.** Every command below must be run from the folder that contains `app.py`.

### Windows (PowerShell)

```powershell
# 1. cd into the project folder first (adjust the path to where you cloned/saved it)
cd "C:\path\to\front-porch-exchange"

# 2. Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
#    If PowerShell says "running scripts is disabled", run this once, then activate again:
#    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# 3. Install the three dependencies
pip install -r requirements.txt

# 4. Add your key (skip if a .env already exists here or one folder above)
Copy-Item .env.example .env
notepad .env        # paste your key after PORTKEY_API_KEY= and save

# 5. Run the app (it opens in your browser at http://localhost:8501)
streamlit run app.py
```

### Mac / Linux (Terminal)

```bash
# 1. cd into the project folder first
cd /path/to/front-porch-exchange

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install the three dependencies
pip install -r requirements.txt

# 4. Add your key (skip if a .env already exists here or one folder above)
cp .env.example .env
nano .env           # paste your key after PORTKEY_API_KEY= and save

# 5. Run the app
streamlit run app.py
```

### Run the safety tests (no API key or internet needed)

```bash
python tests/test_safety.py
```

Each check prints PASS or FAIL. The tests cover PII redaction, injection detection, political-content flags, the 90-minute validator, `cost_usd` math, the budget cutoff, neighbor matching, a malicious database row, and more.

---

## Where the app looks for your `.env`

The app checks two places, relative to `app.py`, and loads the **first one it finds**:

1. `.env` in the **same folder** as `app.py`
2. `.env` **one folder above** (`../.env`)

This means graders can keep a single `.env` one level above the repo and it just works. Key names:

| Key | Used for |
|---|---|
| `PORTKEY_API_KEY` | Used first: the course Portkey gateway with model **gpt-6-luna** |
| `OPENAI_API_KEY` | Fallback: OpenAI directly with **gpt-4o-mini** (cost shows "n/a" since course prices apply to gpt-6-luna only) |

If neither key is found, the app shows a clear message instead of crashing. Key values are never printed, logged, or committed (`.env` is in `.gitignore`).

---

## Using the app

The app opens on three tabs:

- **About (landing page):** a short essay on loneliness, why it matters, and the mission of The Front Porch Exchange, followed by an illustrated map of how a gathering comes together (`assets/porch_infographic.svg`).
- **Hosts:** the host interview and the kit.
  - **Load sample host (Miss Dorothy, crochet, Greenville):** skips the interview with a realistic, pre-filled host profile and runs Agents 2 and 3 right away. This is the fastest, cheapest way to see a full kit.
  - **Chat (left):** answer the guide's questions. It asks one question at a time and wraps up within 12 answers.
  - **Kit (right):** Schedule · Supplies (with the refreshment recipes) · Story Cards · Matches, Invitations & Safety. Tick **Mark as invited** after the coordinator sends an invitation. There's also a **Download printable kit (HTML)** button; open the file and print it from your browser.
  - **Start over:** clears the conversation and kit, so you can serve the next host.
- **Neighbors:** the **"Neighbors who want to join"** sign-up form: first name, age band, skills to learn and to share (each with a skill level), availability, whether they're comfortable meeting inside a host's home, and a contact *preference*. It accepts adults 18+ with explicit consent only and rejects addresses, phone numbers and emails. The community database itself isn't shown on the site; the coordinator works from `porch.db`.
**Telemetry** isn't shown on the website. Every API call is still recorded in `runs_log.jsonl`: agent, input, cached, cache-write and output tokens, estimated USD, seconds and a timestamp, with no user text. That file is the source for the tokenomics figures in the report and slides. To bring back the on-screen panel (per-call table, cost at scale, Safety log), call `render_telemetry(ss, sess)` in `main()`.

The **sidebar** shows the active provider and model, a **Prompt caching** on/off toggle (to compare costs), and the **Session budget (USD)** (default $0.10).

> **The seeded neighbors are fictional sample data.** On first run the app creates `porch.db` with 12 made-up neighbors (first names only, no locations) so matching works out of the box. Real sign-ups are marked separately. An older `porch.db` is upgraded automatically (real sign-ups are kept). Delete `porch.db` to reset.

---

## How it works

| Step | What happens | Tool |
|---|---|---|
| **Agent 1 · Host Intake & Story** | A warm interview, one question per turn, 12-turn cap. It emits a `HostProfile` JSON between `===HOST_PROFILE_START===` / `===HOST_PROFILE_END===`, which is hidden from the chat. | n/a |
| **Agent 2 · Gathering Blueprint** | `HostProfile` → `Blueprint` (schedule, supplies, and a refreshment pairing picked from a fixed recipe catalog) in JSON mode, with one repair retry. | `validate_schedule()`, an explicit Python step; its errors go back for one repair pass |
| **Agent 3 · Community Connector & Safety** | Produces invitations, 4 story cards, 4 safety rules and a flyer → the final `PorchGatheringKit`. | `match_neighbors()` via **real API tool-calling**, plus `content_guardrail_check()` (one regenerate if flagged) |

`match_neighbors` is deterministic Python with no LLM. A match on the skills a neighbor wants to **learn** has the highest weight (a matching skill they'd like to **share** also counts), then skill-level fit with the host's beginner-friendliness, then availability overlap. If the host is hosting inside their home, only neighbors who said they're comfortable meeting indoors are considered. An optional "prefer a mix of ages" setting nudges ranking toward different generations. Only opted-in neighbors are considered, and up to `max_guests + 2` are returned to allow for declines.

### Safety defenses (each one recorded in the session's Safety log)

1. **PII redaction:** street addresses, phone numbers and emails become `[redacted]` before reaching any model, the database or the log.
2. **Prompt-injection defense:** user and database text is passed as delimited *data*. Common attacks ("ignore previous instructions", "reveal your system prompt", role changes, key requests) are blocked before any API call, and the interview continues politely.
3. **Content guardrails:** no political, partisan or religious-debate content and no contact details in cards, invitations or the flyer.
4. **Malfunction handling:** schema validation, at most one repair retry, the 90-minute validator, the 12-turn cap, and friendly errors instead of stack traces.
5. **Overspending:** `BudgetExceeded` is raised *before* any call once spend reaches the budget. There's also `max_output_tokens` on every call, a 1,000-character input cap, and a 60-call session cap. Partial kits are kept.
6. **Privacy:** only first name and matched crafts from database rows ever appear in prompts, the kit or `runs_log.jsonl`. `runs_log.jsonl` stores token counts, cost and timing only, with no user text.

### Prompt caching

The stable instructions and schema come first in a developer message whose `input_text` carries `prompt_cache_breakpoint: {"mode": "explicit"}`. The request also sends `prompt_cache_options: {"mode": "explicit", "ttl": "30m"}`. Per-call data (host profile, chat turns) goes after the breakpoint. If the gateway rejects these fields, the app retries once without them (default/implicit caching) and records a `cache_fallback` entry in the Safety log. Prompts are not padded; the telemetry shows whatever `cached_tokens` the API returns, including 0. Note that providers may still apply automatic caching when the toggle is off.

### Pricing used for estimates

gpt-6-luna, USD per 1M tokens (OpenAI Standard short-context, [source](https://developers.openai.com/api/docs/pricing), course Lecture 11): input $0.10 · cached $0.01 · cache write $0.125 · output $0.50.

Cost = (input − cached − cache_write) × input rate + cached × cached rate + cache_write × write rate + output × output rate.

---

## Files

```
app.py               single-file Streamlit app (all logic + UI in main())
requirements.txt     streamlit, openai, python-dotenv
.streamlit/config.toml  site theme (colors + serif base font)
.env.example         key names only, no values
tests/test_safety.py offline PASS/FAIL safety checks
assets/              mockingbird logo, About-page infographic (SVG) + the script that draws it
porch.db             created on first run (git-ignored)
runs_log.jsonl       per-call telemetry, created on first call (git-ignored)
```
