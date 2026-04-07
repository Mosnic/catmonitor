**🐱  Cat Monitor**

*Autonomous AI Agent — Project Notes*

March 2026  ·  Walter P  ·  spicysnickers

# **Project Overview**

Cat Monitor is a fully local, autonomous AI agent that processes motion-triggered security camera footage from a backyard stray cat feeding station. It identifies individual cats across visits, tracks their health over time, and reasons about what it observes — without any cloud services or human intervention.

| Hardware | AMD Ryzen 9 9900 · 64GB RAM · 1TB NVMe · 8TB SATA · 24TB External |
| :---- | :---- |
| **GPU** | AMD Radeon AI Pro R9700 · 32GB VRAM |
| **OS** | Ubuntu 24.04 LTS |
| **Vision Model** | Qwen3-VL-8B-Instruct served via vLLM on ROCm |
| **Embedding Model** | Qwen3-VL-Embedding-2B via vLLM (pooling mode) — second container, \~4GB VRAM |
| **Cameras** | Reolink · Platform Front \+ Platform Right · Motion triggered FTP upload |
| **Cloud Services** | None — fully local inference, no API costs |

# **System Architecture**

The system is structured in layers. Each layer has a single responsibility and can be developed or replaced independently.

## **Layer Stack**

| ✓  Reolink Cameras | Hardware |
| :---- | :---- |
| **✓  FTP Drop Folder** | Local network |
| **✓  Folder Watcher** | watcher.py |
| **✓  Frame Extractor** | tools/frames.py |
| **✓  Vision Model API** | vLLM \+ Qwen3-VL-8B-Instruct |
| **✓  Cat Identity Engine** | db/store.py \+ ChromaDB |
| **✓  Visit Logger** | db/store.py \+ SQLite |
| **✓  Agent Loop** | agent/loop.py |
| **○  Reflective Agent** | In design — see section below |
| **○  Human Override Interface** | Planned — dashboard correction channel |
| **○  Dashboard** | Planned — Flask or FastAPI |

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

docker run \--rm \\  
  \--group-add=video \\  
  \--cap-add=SYS\_PTRACE \\  
  \--security-opt seccomp=unconfined \\  
  \--device /dev/kfd \\  
  \--device /dev/dri \\  
  \-p 8000:8000 \\  
  \--ipc=host \\  
  \-e "HF\_TOKEN=$HF\_TOKEN" \\  
  \-e VLLM\_ROCM\_USE\_AITER=1 \\  
  \-v \~/.cache/huggingface:/root/.cache/huggingface \\  
  vllm/vllm-openai-rocm:v0.14.0 \\  
  \--model Qwen/Qwen3-VL-8B-Instruct \\  
  \--max-model-len 16384 \\  
  \--enable-auto-tool-choice \\  
  \--tool-call-parser hermes

Key flags: \--max-model-len 16384 prevents KV cache overflow. \--enable-auto-tool-choice and \--tool-call-parser hermes enable the agent tool use loop.

## **2\. Embedding Model Server (New)**

Qwen3-VL-Embedding-2B served via vLLM in a second ROCm container on localhost:8001. Uses \--runner pooling mode to expose /v1/embeddings. Built on the same Qwen3-VL foundation as the reasoning model, so visual representations are coherent between the two.

This replaces all-MiniLM-L6-v2 (CPU) for cat identity embeddings. Instead of embedding a text description of a cat, the actual best frame image is embedded directly. Cat matching becomes image-to-image similarity rather than description-to-description.

| VRAM | \~4GB — well within headroom on 32GB GPU |
| :---- | :---- |
| **Port** | localhost:8001 |
| **Runner** | \--runner pooling |
| **Input** | Best frame image (resized to 640px wide) |
| **Replaces** | all-MiniLM-L6-v2 on CPU |

## **3\. Vision Tool — tools/vision.py**

Sends a single image frame to the vLLM endpoint and returns a structured JSON description of any cats present. This is the agent’s primary sense — what it uses to see.

**Key design decisions:**

* Structured output with controlled vocabularies — body\_condition is poor/fair/good/excellent  
* Camera name read from overlay text in the image itself  
* Validation rules in the prompt prevent non-cats being classified as cats  
* health\_flags and distinctive\_markings are arrays — empty if nothing notable  
* confidence field set to low if lighting is poor or cat partially obscured

## **4\. Frame Extractor — tools/frames.py**

Extracts the best frames from a video clip using an async pipeline. Uses the vision model to score each frame for quality before selecting the top 4\.

* FFmpeg extracts 1 frame per second and resizes to 640px wide in one pass  
* All frames scored concurrently with asyncio (MAX\_CONCURRENT=1 for single vLLM instance)  
* Each frame scored 1-10: cat visibility, sharpness, angle, lighting  
* Top 4 selected with minimum 3-second spacing for temporal diversity  
* Best frame path stored with visit record for identity engine use

*Important: 4K Reolink frames consume \~8000 tokens at full resolution. Resizing to 640px wide drops this to \~238 tokens. Always resize before sending to the model.*

## **5\. Cat Identity Engine — db/store.py**

The core of autonomous cat identification. Uses two databases working together — SQLite for structured facts and ChromaDB for vector-based identity matching.

### **SQLite Schema**

cats    — cat\_id, first\_seen, last\_seen, visit\_count, description,  
           status (active|absent|provisional\_link)  
visits  — visit\_id, cat\_id, timestamp, camera, clip\_path,  
           behavior, body\_condition, health\_flags, lighting,  
           confidence, notes, raw\_json, best\_frame\_path  
alerts  — alert\_id, cat\_id, timestamp, alert\_type, detail, resolved  
links   — link\_id, candidate\_cat\_id, linked\_to\_cat\_id,  
           link\_confidence, created\_at, resolved, resolution

### **ChromaDB Identity Matching**

Each cat is stored as a vector embedding of their best frame image, produced by Qwen3-VL-Embedding-2B. Embeddings are updated over time by averaging confirmed visit embeddings — the prototype becomes more representative as more visits accumulate.

### **Matching Logic**

* Hard filter: coat\_color AND eye\_color must match exactly in ChromaDB metadata (when both are known)  
* If either field is unknown, fall back to image embedding similarity alone  
* Vector similarity search on candidates passing the filter  
* Distance \< 0.25 → known cat, update profile and average embedding  
* Distance in middle range (0.25–0.45) → uncertain match, flagged for review  
* Distance ≥ 0.45 → new cat, create profile with UUID  
* No cats pass hard filter → automatically new cat

### **Embedding Averaging**

After each confirmed visit, the prototype embedding is updated by averaging the new frame embedding with the existing prototype. This makes the identity vector more stable and representative over time. A single photo in poor lighting cannot corrupt a prototype built from many visits.

## **6\. Agent Loop — agent/loop.py**

The reasoning core. Implements a tool-use loop where the model decides what to do, we execute it, feed the result back, and repeat until the model signals it is done.

| Tool | Description |
| :---- | :---- |
| **extract\_best\_frames(video\_path)** | Extract and score frames from a clip. Always called first. |
| **analyze\_cat\_image(image\_path)** | Analyze a single frame. Returns structured cat description. |
| **process\_visit(analysis, clip\_path)** | Log visit to database. Returns known/new/uncertain cat status. |

max\_iterations \= 10 prevents runaway loops. Safety limit — if the agent hasn’t finished in 10 iterations something has gone wrong.

## **7\. Folder Watcher — watcher.py**

Monitors the FTP drop folder for new video files. Waits for upload to complete (file size stable for 3 seconds) then triggers the agent loop. Runs indefinitely, logs all activity to catmonitor.log.

## **8\. Batch Processor — process\_existing.py**

Processes clips already in the FTP folder. Each script has one job.

\# See what's there without processing  
python3 process\_existing.py /path/to/uploads \--dry-run

\# Process only a specific camera and date  
python3 process\_existing.py /path/to/uploads \\  
  \--camera "Platform Right" \\  
  \--date "20260327"

\# Process everything  
python3 process\_existing.py /path/to/uploads

# **Cat Population Model**

The feeding station serves a small, fluid population — typically no more than 4 active cats at any time, cycling over months and years. Cats stop coming and new ones appear. The system must model this correctly.

| Status | Description |
| :---- | :---- |
| **Active** | Visiting regularly now |
| **Absent** | Known cat that has stopped coming — could return |
| **New** | Never seen before — needs a new identity |
| **Returning** | Was absent, now back — same identity, not a new cat |
| **Provisional Link** | Possible returning cat — evidence accumulating, not yet confirmed |

## **Returning Cat Handling**

A cat absent for months may not be recognizable by appearance alone. The image embedding prototype built from confirmed visits provides a memory the system can compare against even when humans cannot. However, body condition and appearance may have changed.

*Key design decision: a returning cat gets a new provisional profile (e.g. cat147) with a possible\_match link to the original (cat123). The original prototype is not altered. Evidence accumulates independently in the new profile. The reflective agent evaluates the evidence and either confirms the link — merging cat147 into cat123’s history — or dismisses it.*

**Link confidence decays with absence duration:**

* Gone 2 weeks → high confidence if returns  
* Gone 3 months → moderate, flag for review  
* Gone 1 year → low, treat almost as new cat but preserve possible link

The original prototype is never contaminated by uncertain data. It earned its stability through confirmed visits.

## **Baselines**

Without a baseline, the reflective agent can see that a cat was logged as poor but has no way to know if that’s a change or if that cat has always been poor. Baselines must be established deliberately.

* After N confirmed visits, compute baseline: typical body\_condition, behavior, visit frequency  
* Flag deviations from the baseline, not just absolute values  
* Baseline updates slowly — a genuinely improving cat shifts their baseline over time  
* Baseline per cat — a cat visiting once a week needs longer to establish one than a daily visitor  
* The averaged image embedding prototype is the visual baseline — unusual drift from it is a signal even when categorical fields look normal

# **Reflective Agent — Design**

A second agent mode that runs during idle periods and reasons over visit history rather than individual clips. The visit agent has a narrow context — one clip, right now. The reflective agent’s value is the opposite: it holds a cat’s history in context to reason about patterns across time.

## **Cat Dossier**

The reflective agent needs a structured context it can reason over, not raw database rows. A get\_cat\_dossier(cat\_id) tool assembles everything known about a cat:

* Prototype embedding and how it was built (visit count, confidence history)  
* Recent best frames — visual history  
* Visit history summary — frequency, timing, camera patterns  
* Confidence trajectory across visits  
* Gap periods — when they stopped and started visiting  
* Any human corrections made  
* Provisional links to other cat profiles if any

The agent reads the dossier and reasons about it using the same loop and model as the visit agent. Same architecture, different context.

## **Reflective Agent Tools**

| Tool | Description |
| :---- | :---- |
| **get\_cat\_dossier(cat\_id)** | Full assembled context for one cat — history, embeddings, gaps, links |
| **get\_cat\_history(cat\_id, limit)** | Recent visits for a known cat |
| **get\_absent\_cats(days)** | Cats not seen in N days |
| **get\_health\_trends(cat\_id)** | Body condition across visits |
| **get\_overnight\_summary()** | All visits in last 12 hours |
| **get\_uncertain\_matches()** | Visits flagged as uncertain identity |
| **get\_provisional\_links()** | All pending returning-cat link candidates |
| **resolve\_link(candidate\_id, confirmed)** | Confirm or dismiss a provisional link |
| **recompute\_prototype(cat\_id)** | Rebuild embedding average after correction |

## **What the Reflective Agent Detects**

* Body condition declining across visits  
* Cats that have stopped visiting — flag absence after N days  
* Changes in visit frequency or timing  
* Health flags appearing for the first time  
* New cats that visited once and never returned  
* Low-confidence identity matches that warrant a second look  
* Provisional links where evidence is accumulating toward confirmation or dismissal  
* Visits where confidence was low and a later visit might resolve identity

# **Human Override**

The system stays autonomous but accepts corrections. The override is not a database command — it is a low-friction interface (dashboard) that presents the evidence and lets the human decide.

When a correction is made ("that’s not cat123"):

* Unlink the visit from cat123  
* Assign to correct cat or create new profile  
* Recompute cat123’s prototype — remove the bad embedding from the average  
* Trigger reflective agent audit of other low-confidence matches for cat123

The dashboard becomes the feedback channel that makes the system smarter over time. Human judgment captured as data, not lost.

# **Known Issues & Decisions**

| Plastic trigger | Piece of plastic moving in wind triggers motion sensor. Fix in Reolink app — draw motion zone excluding the plastic. |
| :---- | :---- |
| **Clip length** | Default 30s clips. Reduce to 15s in Reolink — sufficient for identification, halves processing time. |
| **Frame sampling** | Currently fixed 1fps. Consider TARGET\_FRAMES=20 so sampling adapts to clip length. |
| **Eye color** | Platform Right is overhead — cat eating head-down means eye color often unknown. Hard filter relaxed to image-only when unknown. |
| **Tabby stripes** | Bright light on Platform Front washes out stripe detail. Physical fix: diffusion filter over camera light. Software fallback: image embedding is not affected by this misclassification. |
| **FTP permissions** | Manually uploaded files may have wrong permissions. Fix: sudo chown \-R walterp:walterp /uploads/ |
| **GPU memory fault** | ROCm throws memory fault on unclean Docker stop. Fix: docker stop $(docker ps \-q) then check rocm-smi. |
| **Context window** | max-model-len 16384\. Full 4K frames at 8178 tokens fill this — always resize to 640px wide before sending. |

# **What This Project Teaches About Agents**

## **An agent is: LLM \+ Tools \+ Loop**

Nothing more. The LLM reasons. Tools are functions it can call. The loop continues until the LLM decides it is done. Everything else is infrastructure.

## **Structured output matters**

Free text responses cannot be stored, compared, or reasoned about programmatically. Controlled vocabularies — poor/fair/good/excellent rather than prose — make data useful across visits.

## **Hard filters beat soft similarity for identity**

A general-purpose embedding model treats orange and black as similar because they are both colors. Hard metadata filters on coat\_color and eye\_color before vector similarity correctly separate obviously different cats. Both layers work together — they are not alternatives.

## **Image embeddings beat text embeddings for visual identity**

A text embedding of a cat description summarizes what a human wrote down. An image embedding captures what the cat actually looked like. The Qwen3-VL embedding model shares the same foundation as the reasoning model, making the two coherent. Cameras are fixed — this makes image embeddings stable and meaningful.

## **Prototypes improve with data**

A single photo in poor lighting is a poor identity anchor. An averaged embedding across ten confirmed visits is a robust one. Identity matching improves automatically as cats accumulate visit history.

## **The system needs to be honest about uncertainty**

A returning cat after months of absence is not a confident match — it is a hypothesis. The system surfaces evidence and confidence, creates provisional links, and lets evidence accumulate before committing. It does not pretend to know what it doesn’t.

## **Local inference changes the architecture**

A cloud API handles concurrency. A single local vLLM instance processes one request at a time. MAX\_CONCURRENT=1 is correct for a single GPU. The second container for embeddings runs independently and does not compete with the reasoning model during normal operation.

## **Start simple, add on later**

Every layer added cleanly without breaking what came before. The folder watcher does not know about ChromaDB. The identity engine does not know about the agent loop. Clean separation means each piece can be replaced or extended independently.

*Cat Monitor · Project Notes · March 2026 · Fully local, no cloud*