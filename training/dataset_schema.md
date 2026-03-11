# Legal Drafts Dataset Schema

## Overview

The Legal Drafting AI system uses a JSONL (JSON Lines) dataset for:
1. **Fine-tuning** the base LLM (Llama 3 8B) via LoRA on Indian legal drafts.
2. **RAG** — documents are embedded and stored in FAISS for retrieval during generation.

## File Format

- **Extension**: `.jsonl`
- **Encoding**: UTF-8
- **Line format**: One JSON object per line.

## Required Fields (per line)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `document_type` | string | Yes | One of: `bail_application`, `legal_notice`, `affidavit`, `petition`, `agreement` |
| `facts` | string | Yes | Brief case facts or context (used for retrieval and training input) |
| `draft` | string | Yes | Full formatted legal draft (target for generation and RAG storage) |

## Optional Fields

| Field | Type | Description |
|-------|------|-------------|
| `court_name` | string | Name of court |
| `section` | string | Applicable section (e.g. IPC 420, CrPC 439) |
| `client_name` | string | Applicant/Client name |
| `metadata` | object | Any extra key-value data |

## Example Entry

```json
{
  "document_type": "bail_application",
  "facts": "Client arrested under IPC 420, first offense, no prior record, family dependent on him.",
  "draft": "IN THE COURT OF DISTRICT JUDGE, DELHI\n\nBAIL APPLICATION\nUNDER SECTION 439 CrPC\n\nApplicant: Rajesh Kumar\n\nMost Respectfully Submitted:\n\n1. That the applicant has been falsely implicated in FIR No. 123/2024 under Section 420 IPC.\n2. That the applicant is a first-time offender and has deep roots in society.\n3. That the applicant undertakes to abide by all conditions imposed by this Hon'ble Court.\n\nPRAYER\n\nIt is therefore prayed that this Hon'ble Court may kindly grant bail to the applicant.\n\nPlace: Delhi\nDate: 01.01.2025\n\nApplicant\nRajesh Kumar"
}
```

## Dataset Size

- **Minimum**: 500 examples recommended for meaningful LoRA fine-tuning and RAG.
- **Distribution**: Prefer balanced distribution across `document_type` (e.g. ~100 per type).

## Validation Rules

1. `document_type` must be one of the five supported types.
2. `facts` and `draft` must be non-empty strings.
3. `draft` should be a complete, court-ready document (no placeholders in production data).
