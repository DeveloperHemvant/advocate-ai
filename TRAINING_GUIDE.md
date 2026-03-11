# Legal Drafting AI – Training & Deployment Guide

Step-by-step instructions to train the model, build the vector database, and run the API.

---

## 1. Collect Legal Drafts Dataset

- Gather at least **500** real or anonymised Indian legal drafts (bail applications, legal notices, affidavits, petitions, agreements).
- Ensure each item has:
  - **document_type**: One of `bail_application`, `legal_notice`, `affidavit`, `petition`, `agreement`
  - **facts**: Brief case facts or context (one or two paragraphs)
  - **draft**: Full formatted legal draft (court-ready text)
- Prefer a balanced distribution (e.g. ~100 per document type).

---

## 2. Format Dataset as JSONL

Save the dataset as `datasets/legal_drafts.jsonl`: one JSON object per line, UTF-8 encoding.

**Example line:**

```json
{"document_type": "bail_application", "facts": "Client arrested under IPC 420, first offense.", "draft": "IN THE COURT OF ...\n\nBAIL APPLICATION\n..."}
```

See `training/dataset_schema.md` for the full schema and optional fields.

---

## 3. Run Dataset Preparation

Validate and clean the dataset; optionally create train/validation split:

```bash
cd legal-ai
python training/prepare_dataset.py datasets/legal_drafts.jsonl -o datasets/legal_drafts_clean.jsonl --val-ratio 0.1
```

This produces:

- Cleaned JSONL (if `-o` is given)
- `datasets/legal_drafts_train.jsonl` and `datasets/legal_drafts_val.jsonl` (if `--val-ratio` > 0)

---

## 4. Run LoRA Training

**Requirements:** GPU with at least 16 GB VRAM (e.g. 4-bit quantization). Install dependencies:

```bash
pip install -r requirements.txt
```

**Training command:**

```bash
python training/train_lora.py --dataset datasets/legal_drafts.jsonl --output-dir models/legal_llama_lora --base-model meta-llama/Llama-3-8B-Instruct --epochs 3 --batch-size 2 --grad-accum 8
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--dataset` | `datasets/legal_drafts.jsonl` | Path to JSONL dataset |
| `--output-dir` | `models/legal_llama_lora` | Where to save LoRA adapter and tokenizer |
| `--base-model` | `meta-llama/Llama-3-8B-Instruct` | Base HuggingFace model |
| `--epochs` | 3 | Number of training epochs |
| `--batch-size` | 2 | Per-device batch size |
| `--grad-accum` | 8 | Gradient accumulation steps |
| `--no-4bit` | - | Disable 4-bit quantization (needs more VRAM) |

After training, the LoRA adapter and tokenizer are in `models/legal_llama_lora`. Use this path when serving the model with vLLM (merge adapter or load with PEFT, depending on your vLLM setup).

---

## 5. Build Vector Database (FAISS)

Embed all drafts and build the FAISS index for RAG:

```bash
python scripts/build_vector_db.py --dataset datasets/legal_drafts.jsonl
```

Default output:

- `vector_index/legal_faiss.index`
- `vector_index/legal_metadata.json`

To use Instructor-xl embeddings (optional):

```bash
pip install instructor-embedding
# Set LEGAL_AI_USE_INSTRUCTOR=true or:
python scripts/build_vector_db.py --instructor
```

---

## 6. Start vLLM Server (Inference)

Run a vLLM server with the base model (or with the fine-tuned adapter if your vLLM version supports it).

**Base model only:**

```bash
python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-3-8B-Instruct --port 8001
```

**With LoRA adapter:** Refer to vLLM documentation for loading PEFT adapters. Alternatively, merge the LoRA weights into the base model and serve the merged model.

Ensure the API is available at `http://localhost:8001/v1` (or set `LEGAL_AI_VLLM_BASE_URL` in `.env`).

---

## 7. Start FastAPI Server

From the project root:

```bash
cd legal-ai
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Or:

```bash
python -m app.main
```

API docs: `http://localhost:8000/docs`.

---

## 8. Optional: Evaluation

Validate the validation set and print stats:

```bash
python training/evaluation.py datasets/legal_drafts_val.jsonl
```

---

## 9. Quick Test (CLI)

Generate a draft without the API:

```bash
python scripts/run_inference.py --document-type bail_application --case-facts "Client arrested under IPC 420, first offense" --court-name "District Court Delhi" --client-name "Rajesh Kumar" --section "IPC 420"
```

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LEGAL_AI_VLLM_BASE_URL` | vLLM OpenAI-compatible API base URL | `http://localhost:8001/v1` |
| `LEGAL_AI_EMBEDDING_MODEL` | Sentence-transformers model for embeddings | `BAAI/bge-small-en-v1.5` |
| `LEGAL_AI_USE_INSTRUCTOR` | Use Instructor-xl for embeddings | `false` |
| `LEGAL_AI_RAG_TOP_K` | Number of RAG examples per request | 3 |
| `LEGAL_AI_DEBUG` | Enable debug mode | `false` |

---

## Integration with Laravel / Node Backend

- **API base:** `POST http://<host>:8000/generate-draft` with JSON body (see API docs).
- **Response:** `{ "draft": "...", "validation": { "valid": true, "errors": [], "warnings": [] }, "success": true }`.
- Call from Laravel (HTTP client) or Node (axios/fetch) and forward the response to your front end or document pipeline.
