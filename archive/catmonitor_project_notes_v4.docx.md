**🐱 Cat Monitor**

*Autonomous AI Agent — Project Notes*

Version 4 · March 2026 · Walter P · spicysnickers

# **Project Overview**

Cat Monitor is a fully local, autonomous AI agent that processes motion-triggered security camera footage from a backyard stray cat feeding station. It identifies individual cats across visits, tracks their health over time, and reasons about what it observes — without any cloud services or human intervention.

| Hardware | AMD Ryzen 9 9900 · 64GB RAM · 1TB NVMe · 8TB SATA · 24TB External |
| :---- | :---- |
| GPU | AMD Radeon AI Pro R9700 · 32GB VRAM |
| OS | Ubuntu 24.04 LTS |
| Vision Model | Qwen3-VL-8B-Instruct served via vLLM on ROCm |
| Embedding Model | Qwen3-VL-Embedding-2B via vLLM (pooling mode) — second container, \~4GB VRAM |
| Coding Model | Qwen2.5-Coder-7B via Ollama on CPU — localhost:11434 |
| Cameras | Reolink · Platform Front \+ Platform Right · Motion triggered FTP upload |
| Cloud Services | None — fully local inference, no API costs |

# **System Architecture**

The system is structured in layers. Each layer has a single responsibility and can be developed or replaced independently.

## **Layer Stack**

| Layer | Status |
| :---- | :---- |
| ✓ Reolink Cameras | Hardware |
| ✓ FTP Drop Folder | Local network |
| ✓ Folder Watcher | watcher.py |
| ✓ Frame Extractor | tools/frames.py |
| ✓ Vision Model API | vLLM \+ Qwen3-VL-8B-Instruct (GPU, localhost:8000) |
| ✓ Embedding Model API | vLLM \+ Qwen3-VL-Embedding-2B (GPU, localhost:8001) |
| ✓ Cat Identity Engine | db/store.py \+ ChromaDB |
| ✓ Visit Logger | db/store.py \+ SQLite |
| ✓ Agent Loop | agent/loop.py |
| ○ Reflective Agent | In design — see section below |
| ○ Coding Model | Qwen2.5-Coder-7B via Ollama on CPU — planned |
| ○ Human Override Interface | Planned — dashboard correction channel |
| ○ Dashboard | Planned — Flask or FastAPI |

## **Data Flow**

* Reolink camera detects motion → records clip → uploads via FTP

* watcher.py detects new file → waits for upload to complete → triggers agent

* Agent calls extract\_best\_frames() → FFmpeg extracts 1fps frames → Qwen3-VL scores each frame → top 4 selected

* Agent calls analyze\_cat\_image() on each best frame → structured JSON description returned

* Agent calls process\_visit() → ChromaDB identity match → SQLite visit logged

* Agent produces natural language summary → logged

# **Components**

## **1\. vLLM Vision Model Server**

Qwen3-VL-8B-Instruct served via vLLM in a ROCm Docker container. Exposes an OpenAI-compatible REST API on localhost:8000. All vision analysis and agent reasoning calls go through this endpoint.

docker run --restart unless-stopped \
  --group-add=video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --device /dev/kfd \
  --device /dev/dri \
  -p 8000:8000 \
  --ipc=host \
  -e "HF_TOKEN=$HF_TOKEN" \
  -e VLLM_ROCM_USE_AITER=1 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai-rocm:v0.14.0 \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.70 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes

Key flags: \--max-model-len 8192 prevents KV cache overflow. \--enable-auto-tool-choice and \--tool-call-parser hermes enable the agent tool use loop.

## **2\. Embedding Model Server**

Qwen3-VL-Embedding-2B served via vLLM in a second ROCm container on localhost:8001. Uses \--runner pooling mode to expose /v1/embeddings. Built on the same Qwen3-VL foundation as the reasoning model, so visual representations are coherent between the two.

docker run --restart unless-stopped \
  --group-add=video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --device /dev/kfd \
  --device /dev/dri \
  -p 8001:8001 \
  --ipc=host \
  -e "HF_TOKEN=$HF_TOKEN" \
  -e VLLM_ROCM_USE_AITER=1 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai-rocm:v0.14.0 \
  --model Qwen/Qwen3-VL-Embedding-2B \
  --runner pooling \
  --max-model-len 4096 \
  --port 8001 \
  --gpu-memory-utilization 0.25

## **3\. Coding Model — Ollama on CPU (New)**

Qwen2.5-Coder-7B served via Ollama directly on the host (not in Docker). Exposes an OpenAI-compatible REST API on localhost:11434. Used exclusively by the reflective agent for statistical computation and data analysis code generation.

Rationale for CPU / Ollama rather than a third Docker container:

* CPU models do not require ROCm device passthrough — the Docker complexity buys nothing for a CPU workload
* Ollama manages model downloads, storage, and serving itself — single install, single command
* The reflective agent runs during idle periods, so 10–30 second generation latency is acceptable
* 32GB RAM on the Ryzen 9 9900 comfortably fits a 7B model leaving headroom for the OS and other processes
* No VRAM consumed — the GPU remains fully available for the vision and embedding containers

  \# Install Ollama on host
  curl \-fsSL https://ollama.com/install.sh | sh
  \# Pull the model
  ollama pull qwen2.5-coder:7b
  \# Serves automatically on localhost:11434
| Property | Value |
| :---- | :---- |
| Model | Qwen2.5-Coder-7B |
| Runtime | Ollama on host (no Docker) |
| Port | localhost:11434 |
| Hardware | CPU — Ryzen 9 9900 · 64GB RAM |
| VRAM consumed | None |
| Latency | 10–30s per generation (acceptable for reflective agent) |
| Protocol | OpenAI-compatible REST API |

## **4\. Vision Tool — tools/vision.py**

Sends a single image frame to the vLLM endpoint and returns a structured JSON description of any cats present. This is the agent's primary sense — what it uses to see.

**Key design decisions:**

* Structured output with controlled vocabularies — body\_condition is poor/fair/good/excellent
* Camera name read from overlay text in the image itself
* Validation rules in the prompt prevent non-cats being classified as cats
* health\_flags and distinctive\_markings are arrays — empty if nothing notable
* confidence field set to low if lighting is poor or cat partially obscured

## **5\. Frame Extractor — tools/frames.py**

Extracts the best frames from a video clip using an async pipeline. Uses the vision model to score each frame for quality before selecting the top 4\.

* FFmpeg extracts 1 frame per second and resizes to 640px wide in one pass
* All frames scored concurrently with asyncio (MAX\_CONCURRENT=1 for single vLLM instance)
* Each frame scored 1–10: cat visibility, sharpness, angle, lighting
* Top 4 selected with minimum 3-second spacing for temporal diversity
* Best frame path stored with visit record for identity engine use

*Important: 4K Reolink frames consume \~8000 tokens at full resolution. Resizing to 640px wide drops this to \~238 tokens. Always resize before sending to the model.*

## **6\. Cat Identity Engine — db/store.py**

The core of autonomous cat identification. Uses two databases working together — SQLite for structured facts and ChromaDB for vector-based identity matching.

**SQLite Schema**

 CREATE TABLE IF NOT EXISTS cats (
            cat_id      TEXT PRIMARY KEY,
            first_seen  TEXT NOT NULL,
            last_seen   TEXT NOT NULL,
            visit_count INTEGER DEFAULT 1,
            description TEXT,
            health_flags TEXT DEFAULT '[]'
        );

visits  
CREATE TABLE IF NOT EXISTS visits (
            visit_id         INTEGER PRIMARY KEY AUTOINCREMENT,
            cat_id           TEXT NOT NULL,
            timestamp        TEXT NOT NULL,
            camera           TEXT,
            clip_path        TEXT,
            behavior         TEXT,
            body_condition   TEXT,
            temperature_f    REAL,
            precipitation_mm REAL,
            windspeed_mph    REAL,
            weather_code     INTEGER,
            health_flags     TEXT DEFAULT '[]',
            lighting         TEXT,
            confidence       REAL,
            notes            TEXT,
            raw_json         TEXT,
            FOREIGN KEY (cat_id) REFERENCES cats(cat_id)
        );

alerts  
    CREATE TABLE IF NOT EXISTS alerts (
            alert_id    INTEGER PRIMARY KEY AUTOINCREMENT,
            cat_id      TEXT,
            timestamp   TEXT NOT NULL,
            alert_type  TEXT NOT NULL,
            detail      TEXT,
            resolved    INTEGER DEFAULT 0
        );

links  
   CREATE TABLE IF NOT EXISTS links (
            link_id           INTEGER PRIMARY KEY AUTOINCREMENT,
            candidate_cat_id  TEXT NOT NULL,
            linked_to_cat_id  TEXT NOT NULL,
            link_confidence   REAL NOT NULL,
            created_at        TEXT NOT NULL,
            resolved          INTEGER DEFAULT 0,
            resolution        TEXT,
            FOREIGN KEY (candidate_cat_id) REFERENCES cats(cat_id),
            FOREIGN KEY (linked_to_cat_id) REFERENCES cats(cat_id)
        );
        
**ChromaDB Identity Matching**

Each cat is stored as a vector embedding of their best frame image, produced by Qwen3-VL-Embedding-2B. Embeddings are updated over time by averaging confirmed visit embeddings.

**Matching Logic**

* Hard filter: coat_color AND eye_color must match exactly in ChromaDB metadata (when both are known)
* If either field is unknown, fall back to image embedding similarity alone
* Vector similarity search on candidates passing the filter
* Distance \< 0.25 → known cat, update profile and average embedding
* Distance in middle range (0.25–0.45) → uncertain match, flagged for review
* Distance ≥ 0.45 → new cat, create profile with UUID
* No cats pass hard filter → automatically new cat

**Embedding Averaging**

After each confirmed visit, the prototype embedding is updated by averaging the new frame embedding with the existing prototype. A single photo in poor lighting cannot corrupt a prototype built from many visits.

## **7\. Agent Loop — agent/loop.py**

The reasoning core. Implements a tool-use loop where the model decides what to do, we execute it, feed the result back, and repeat until the model signals it is done.

| Tool | Description |  |
| :---- | :---- | :---- |
| extract\_best\_frames(video\_path) | Extract and score frames from a clip. Always called first. |  |
| analyze\_cat\_image(image\_path) | Analyze a single frame. Returns structured cat description. |  |
| process\_visit(analysis, clip\_path) | Log visit to database. Returns known/new/uncertain cat status. |  |

max\_iterations \= 10 prevents runaway loops. Safety limit — if the agent hasn't finished in 10 iterations something has gone wrong.

## **8\. Folder Watcher — watcher.py**

Monitors the FTP drop folder for new video files. Waits for upload to complete (file size stable for 3 seconds) then triggers the agent loop. Runs indefinitely, logs all activity to catmonitor.log.

## **9\. Batch Processor — process\_existing.py**

Processes clips already in the FTP folder.

\# See what's there without processing

python3 process\_existing.py /path/to/uploads \--dry-run

\# Process only a specific camera and date

python3 process\_existing.py /path/to/uploads \\

  \--camera "Platform Right" \\

  \--date "20260327"

\# Process everything

python3 process\_existing.py /path/to/uploads

# **Cat Population Model**

The feeding station serves a small, fluid population — typically no more than 4 active cats at any time, cycling over months and years. The system must model this correctly.

| Status | Description |
| :---- | :---- |
| Active | Visiting regularly now |
| Absent | Known cat that has stopped coming — could return |
| New | Never seen before — needs a new identity |
| Returning | Was absent, now back — same identity, not a new cat |
| Provisional Link | Possible returning cat — evidence accumulating, not yet confirmed |

## **Returning Cat Handling**

A cat absent for months may not be recognizable by appearance alone. The image embedding prototype built from confirmed visits provides a memory the system can compare against even when humans cannot.

*Key design decision: a returning cat gets a new provisional profile (e.g. cat147) with a possible\_match link to the original (cat123). The original prototype is not altered. Evidence accumulates independently in the new profile. The reflective agent evaluates the evidence and either confirms the link — merging cat147 into cat123's history — or dismisses it.*

**Link confidence decays with absence duration:**
* Gone 2 weeks → high confidence if returns
* Gone 3 months → moderate, flag for review
* Gone 1 year → low, treat almost as new cat but preserve possible link

## **Baselines**

Without a baseline, the reflective agent can see that a cat was logged as poor but has no way to know if that's a change or if that cat has always been poor.

* After N confirmed visits, compute baseline: typical body\_condition, behavior, visit frequency
* Flag deviations from the baseline, not just absolute values
* Baseline updates slowly — a genuinely improving cat shifts their baseline over time
* Baseline per cat — a cat visiting once a week needs longer to establish one than a daily visitor
* The averaged image embedding prototype is the visual baseline — unusual drift from it is a signal even when categorical fields look normal

# **Reflective Agent — Design**

A second agent mode that runs during idle periods and reasons over visit history rather than individual clips. The visit agent has a narrow context — one clip, right now. The reflective agent's value is the opposite: it holds a cat's history in context to reason about patterns across time.

## **Division of Responsibility**

The reflective agent uses Qwen3-VL (GPU) for reasoning, pattern recognition, and tool selection — the same model as the visit agent. When quantitative analysis is needed, it delegates to the coding model running on CPU via Ollama. Neither model needs to know the other exists; the coding model is just another tool from the reflective agent's perspective.

| Responsibility | Model |
| :---- | :---- |
| Vision analysis | Qwen3-VL-8B (GPU, localhost:8000) |
| Reasoning and pattern detection | Qwen3-VL-8B (GPU, localhost:8000) |
| Tool selection and orchestration | Qwen3-VL-8B (GPU, localhost:8000) |
| Image embeddings | Qwen3-VL-Embedding-2B (GPU, localhost:8001) |
| Statistical computation | Qwen2.5-Coder-7B (CPU, localhost:11434) |
| Data analysis code generation | Qwen2.5-Coder-7B (CPU, localhost:11434) |

## **How the Coding Model Integrates**

The reflective agent (Qwen3-VL) reasons about cat history and identifies questions that require quantitative answers — for example, whether visit frequency has changed significantly, or whether body condition decline is statistically meaningful against a baseline. Rather than attempting arithmetic in-context (which is unreliable), it calls a generateand_run_analysis(task_description) tool.

The tool:
* Sends the plain-language task description to Qwen2.5-Coder-7B on localhost:11434
* Receives Python code back (pandas, numpy, scipy against the SQLite database)
* Executes the code in a sandboxed subprocess with a timeout
* Returns the result to Qwen3-VL as a tool result

Qwen3-VL then continues reasoning with the precise numerical result. The coding model only generates code — it has no knowledge of the broader agent context.

## **Cat Dossier**

The reflective agent needs a structured context it can reason over, not raw database rows. A get_cat_dossier(cat_id) tool assembles everything known about a cat:

* Prototype embedding and how it was built (visit count, confidence history)
* Recent best frames — visual history
* Visit history summary — frequency, timing, camera patterns
* Confidence trajectory across visits
* Gap periods — when they stopped and started visiting
* Any human corrections made
* Provisional links to other cat profiles if any

## **Reflective Agent Tools**

| Tool | Description |
| :---- | :---- |
| get_cat_dossier(cat_id) | Full assembled context for one cat — history, embeddings, gaps, links |
| get_cat_history(cat_id, limit) | Recent visits for a known cat |
| get_absent_cats(days) | Cats not seen in N days |
| get_health_trends(cat_id) | Body condition across visits |
| get_overnight_summary() | All visits in last 12 hours |
| get_uncertain_matches() | Visits flagged as uncertain identity |
| get_provisional_links() | All pending returning-cat link candidates |
| resolve_link(candidate_id, confirmed) | Confirm or dismiss a provisional link |
| recompute_prototype(cat_id) | Rebuild embedding average after correction |
| generate_and_run_analysis(task_description) | Delegate statistical computation to coding model on CPU. Returns exact numerical result. |

## **What the Reflective Agent Detects**

* Body condition declining across visits
* Cats that have stopped visiting — flag absence after N days
* Changes in visit frequency or timing
* Health flags appearing for the first time
* New cats that visited once and never returned
* Low-confidence identity matches that warrant a second look
* Provisional links where evidence is accumulating toward confirmation or dismissal
* Visits where confidence was low and a later visit might resolve identity
* Statistical deviations from established baselines (via coding model)
* Embedding drift correlating with health or behavior changes (via coding model)

## **What the Coding Model Computes**

Examples of tasks delegated to Qwen2.5-Coder-7B:

* Visit frequency rolling averages and trend detection
* Body condition baseline deviation — is the current value a statistically meaningful departure?
* Link confidence decay interpolation between anchor points (2 weeks, 3 months, 1 year)
* Embedding cosine distance between current prototype and a new visit
* Gap period analysis — detecting when absence patterns become unusual
* Temporal clustering of visit times to detect schedule changes

# **Human Override**

The system stays autonomous but accepts corrections. The override is not a database command — it is a low-friction interface (dashboard) that presents the evidence and lets the human decide.

**When a correction is made ("that's not cat123"):**

* Unlink the visit from cat123
* Assign to correct cat or create new profile
* Recompute cat123's prototype — remove the bad embedding from the average
* Trigger reflective agent audit of other low-confidence matches for cat123

The dashboard becomes the feedback channel that makes the system smarter over time. Human judgment captured as data, not lost.

# **Known Issues & Decisions**

| Issue | Resolution |
| :---- | :---- |

| Frame sampling | Currently fixed 1fps. Consider TARGET\_FRAMES=20 so sampling adapts to clip length. |
| Eye color | Platform Right is overhead — cat eating head-down means eye color often unknown. Hard filter relaxed to image-only when unknown. |
| Tabby stripes | Bright light on Platform Front washes out stripe detail. Physical fix: diffusion filter over camera light. Software fallback: image embedding is not affected. |


# **What This Project Teaches About Agents**

An agent is: LLM \+ Tools \+ Loop. Nothing more. The LLM reasons. Tools are functions it can call. The loop continues until the LLM decides it is done. Everything else is infrastructure.

## **Structured output matters**

Free text responses cannot be stored, compared, or reasoned about programmatically. Controlled vocabularies — poor/fair/good/excellent rather than prose — make data useful across visits.

## **Hard filters beat soft similarity for identity**

A general-purpose embedding model treats orange and black as similar because they are both colors. Hard metadata filters on coat\_color and eye\_color before vector similarity correctly separate obviously different cats. Both layers work together — they are not alternatives.

## **Image embeddings beat text embeddings for visual identity**

A text embedding of a cat description summarizes what a human wrote down. An image embedding captures what the cat actually looked like. The Qwen3-VL embedding model shares the same foundation as the reasoning model, making the two coherent. Cameras are fixed — this makes image embeddings stable and meaningful.

## **Delegate computation to the right tool**

A vision-language model is not a calculator. When the reflective agent needs to know whether a trend is statistically significant, it should ask a model trained to write that computation — not approximate it in-context. The coding model on CPU is cheap to run, tolerant of latency, and produces deterministic results. This is the same principle as tool use generally: the LLM decides what to compute, a tool does the computing.

## **Prototypes improve with data**

A single photo in poor lighting is a poor identity anchor. An averaged embedding across ten confirmed visits is a robust one. Identity matching improves automatically as cats accumulate visit history.

## **The system needs to be honest about uncertainty**

A returning cat after months of absence is not a confident match — it is a hypothesis. The system surfaces evidence and confidence, creates provisional links, and lets evidence accumulate before committing. It does not pretend to know what it doesn't.

## **Local inference changes the architecture**

A cloud API handles concurrency. A single local vLLM instance processes one request at a time. MAX\_CONCURRENT=1 is correct for a single GPU. The second container for embeddings runs independently and does not compete with the reasoning model during normal operation. The coding model on CPU runs independently of both.

## **Start simple, add on later**

Every layer added cleanly without breaking what came before. The folder watcher does not know about ChromaDB. The identity engine does not know about the agent loop. The coding model does not know about cat history. Clean separation means each piece can be replaced or extended independently.

*Cat Monitor · Project Notes v4 · March 2026 · Fully local, no cloud*