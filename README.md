# Legal Drafting AI

Self-hosted Legal Drafting AI for Indian Advocates. Generates bail applications, legal notices, affidavits, petitions, and agreements using **Llama 3 8B** (vLLM), **FAISS** RAG, and optional **LoRA** fine-tuning. No third-party paid APIs.

## Features

- **Document types:** Bail Application, Legal Notice, Affidavit, Petition, Agreement
- **Pipeline:** Template selection → RAG retrieval (FAISS) → prompt build → local LLM → validation → formatted draft
- **Stack:** Python, FastAPI, vLLM, FAISS, BGE-small/Instructor embeddings, PEFT/LoRA
- **API:** REST endpoints for draft generation and semantic search; ready for Laravel/Node integration

## Quick Start (with Ollama)

1. **Install Ollama** from [ollama.com](https://ollama.com) and start it (it runs in the background).
2. **Pull a model** (e.g. Llama 3.2):
   ```bash
   ollama pull llama3.2
   ```
3. **Install Python deps and set env** (from the `legal-ai` folder):
   ```bash
   cd legal-ai
   pip install -r requirements.txt
   copy .env.example .env
   ```
   In `.env` set (Ollama defaults):
   ```
   LEGAL_AI_VLLM_BASE_URL=http://localhost:11434/v1
   LEGAL_AI_LLM_MODEL_NAME=llama3.2
   ```
4. **Build RAG index** (optional but recommended):
   ```bash
   python scripts/build_vector_db.py
   ```
5. **Start the API:**
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```
6. **Generate a draft:**  
   `POST http://localhost:8000/generate-draft` with body:
   ```json
   {
     "document_type": "bail_application",
     "court_name": "District Court Delhi",
     "client_name": "Rajesh Kumar",
     "section": "IPC 420",
     "case_facts": "Client falsely implicated, first offense"
   }
   ```
   Or open **http://localhost:8000/docs** and use the Swagger UI.

   **Ollama URL in `.env`:** Use `http://localhost:11434` (no `/v1`). The app will try `/api/chat` and then `/api/generate` automatically.

### If you get 404 when calling the generate-draft API

- **Browser** shows "Method Not Allowed" for `http://localhost:11434/api/chat` (GET) — that’s normal; the API expects POST.
- **App** returns 404: the process running the FastAPI app might not see the same "localhost" as your browser (e.g. Docker or WSL).
  - Run the API **on the same Windows machine** as Ollama (no Docker/WSL): use `LEGAL_AI_VLLM_BASE_URL=http://localhost:11434`.
  - If the API runs **inside Docker** on Windows/Mac, point it at the host: `LEGAL_AI_VLLM_BASE_URL=http://host.docker.internal:11434`.
  - Confirm Ollama is running and the model is pulled: `ollama list`, and set `LEGAL_AI_LLM_MODEL_NAME` to an exact name from that list (e.g. `llama3.2`).

## Quick Start (with vLLM)

1. **Install:** `pip install -r requirements.txt`
2. **Dataset:** Add at least 500 examples to `datasets/legal_drafts.jsonl` (see `training/dataset_schema.md`).
3. **Build RAG index:** `python scripts/build_vector_db.py`
4. **Start vLLM** (separate terminal):  
   `python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-3-8B-Instruct --port 8001`
5. **Set env** so the app talks to vLLM:  
   `LEGAL_AI_VLLM_BASE_URL=http://localhost:8001/v1` and `LEGAL_AI_LLM_MODEL_NAME=meta-llama/Llama-3-8B-Instruct`
6. **Start API:** `uvicorn app.main:app --host 0.0.0.0 --port 8000`
7. **Generate draft:**  
   `POST http://localhost:8000/generate-draft` with body:
   ```json
   {
     "document_type": "bail_application",
     "court_name": "District Court Delhi",
     "client_name": "Rajesh Kumar",
     "section": "IPC 420",
     "case_facts": "Client falsely implicated, first offense"
   }
   ```

## Project Structure

```
legal-ai/
├── app/                 # FastAPI app, routes, services, models, vectorstore, utils
├── training/            # Dataset schema, prepare_dataset, train_lora, evaluation
├── datasets/            # legal_drafts.jsonl
├── scripts/             # build_vector_db, run_inference
├── requirements.txt
├── TRAINING_GUIDE.md    # Step-by-step training and deployment
└── README.md
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Liveness |
| GET | `/ready` | Readiness (vLLM, FAISS config) |
| POST | `/generate-draft` | Generate legal draft (body: document_type, case_facts, court_name, client_name, section, etc.) |
| POST/GET | `/search-legal-docs` | Semantic search over legal drafts (RAG-style) |

## Training (LoRA)

See **TRAINING_GUIDE.md** for:

1. Collecting and formatting the dataset (JSONL)
2. Running `prepare_dataset.py`
3. Running `train_lora.py` (HuggingFace + PEFT)
4. Building the FAISS vector DB
5. Starting vLLM and the FastAPI server

## License

Use and modify as needed for your organisation.
