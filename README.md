# RFT FastAPI Server

This repository contains a simple FastAPI application that exposes endpoints for running a Reinforcement Fine-Tuning (RFT) job using OpenAI's API. The server accepts training and validation files and creates the fine-tuning job for you.

## Requirements

- Python 3.10+
- `fastapi`
- `uvicorn`
- `openai`

Install the dependencies using pip:

```bash
pip install fastapi uvicorn openai
```

## Running the Server

Set the `OPENAI_API_KEY` environment variable with your OpenAI API key and start the server with `uvicorn`:

```bash
export OPENAI_API_KEY=your-key
uvicorn server:app --reload
```

The server will be available at `http://localhost:8000`.

## Endpoints

### `POST /train`
Creates an RFT job.

**Form fields:**

- `training_file` – Training dataset file (JSONL).
- `validation_file` – Validation dataset file (JSONL).
- `evaluation_prompt` – Prompt used by the score model grader.
- `model` – Base model name (default `o4-mini-2025-04-16`).
- `grader_model` – Model used for grading (default `gpt-4o-2024-08-06`).
- `n_epochs` – Number of training epochs.
- `batch_size` – Batch size.
- `reasoning_effort` – Reasoning effort hyperparameter.
- `seed` – Random seed.
- `api_key` – Optional API key if not using `OPENAI_API_KEY`.

Returns the created job ID and its initial status.

### `GET /jobs/{job_id}`
Returns the status of an existing RFT job.

```bash
curl http://localhost:8000/jobs/<job_id>
```

## Testing
Run a simple syntax check:
```bash
python -m py_compile server.py
```

