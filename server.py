from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import os
import tempfile
from openai import OpenAI
from openai.types.fine_tuning import ReinforcementMethod, ReinforcementHyperparameters
from openai.types.graders import ScoreModelGrader

app = FastAPI(title="RFT Training Server")


def get_client(api_key: str | None = None) -> OpenAI:
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise HTTPException(status_code=400, detail="OpenAI API key missing")
    return OpenAI(api_key=key)


@app.post("/train")
async def create_training_job(
    training_file: UploadFile = File(...),
    validation_file: UploadFile = File(...),
    evaluation_prompt: str = Form(...),
    model: str = Form("o4-mini-2025-04-16"),
    grader_model: str = Form("gpt-4o-2024-08-06"),
    n_epochs: int = Form(6),
    batch_size: int = Form(4),
    reasoning_effort: str = Form("medium"),
    seed: int = Form(42),
    api_key: str | None = Form(None),
):
    client = get_client(api_key)

    # Persist uploaded files temporarily
    with tempfile.NamedTemporaryFile(delete=False) as tf:
        tf.write(await training_file.read())
        train_path = tf.name
    with tempfile.NamedTemporaryFile(delete=False) as vf:
        vf.write(await validation_file.read())
        val_path = vf.name

    training_resp = client.files.create(file=open(train_path, "rb"), purpose="fine-tune")
    validation_resp = client.files.create(file=open(val_path, "rb"), purpose="fine-tune")

    job = client.fine_tuning.jobs.create(
        training_file=training_resp.id,
        validation_file=validation_resp.id,
        model=model,
        method={
            "type": "reinforcement",
            "reinforcement": ReinforcementMethod(
                grader=ScoreModelGrader(
                    name="score_health",
                    type="score_model",
                    input=[{"role": "user", "type": "message", "content": evaluation_prompt}],
                    model=grader_model,
                ),
                hyperparameters=ReinforcementHyperparameters(
                    reasoning_effort=reasoning_effort,
                    n_epochs=n_epochs,
                    batch_size=batch_size,
                ),
            ),
        },
        seed=seed,
    )

    return {"job_id": job.id, "status": job.status}


@app.get("/jobs/{job_id}")
def get_job_status(job_id: str, api_key: str | None = None):
    client = get_client(api_key)
    job = client.fine_tuning.jobs.retrieve(job_id)
    return {"job_id": job.id, "status": job.status}
