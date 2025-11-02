# app/main.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
import logging
import os
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fastapi_app")

app = FastAPI(
    title="Example FastAPI + Gunicorn App",
    description="Demo microservice / REST API using FastAPI served by Gunicorn+Uvicorn workers",
    version="0.1.0",
)

# Allow CORS for common dev origins (adjust in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    text: str
    top_k: Optional[int] = 1

class PredictResponse(BaseModel):
    input: str
    top_k: int
    predictions: list

@app.on_event("startup")
async def startup_event():
    # Simulate startup tasks (DB connection, model load etc.)
    logger.info("Starting up application...")
    # Example: load ML model here
    # app.state.model = load_my_model()
    time.sleep(0.1)
    logger.info("Startup complete.")

@app.on_event("shutdown")
async def shutdown_event():
    # Clean up resources
    logger.info("Shutting down application...")
    # Example: close DB connections
    time.sleep(0.1)
    logger.info("Shutdown complete.")

@app.get("/health", tags=["health"])
async def health():
    """Simple health check"""
    return {"status": "ok", "uptime": time.time()}

@app.get("/", tags=["root"])
async def root():
    return {"message": "Hello from FastAPI + Gunicorn!"}

@app.get("/echo/{message}", tags=["demo"])
async def echo(message: str):
    """Simple path param demo"""
    return {"echo": message}

def _simple_inference(text: str, top_k: int = 1):
    """
    A placeholder inference function. Replace with actual model inference.
    For demo purposes returns the words sorted by length as 'predictions'.
    """
    words = text.split()
    # simple deterministic "scoring"
    scored = sorted(words, key=lambda w: len(w), reverse=True)
    top = scored[: max(1, top_k)]
    # return list of (token, score) style tuples to simulate model output
    return [{"token": t, "score": float(len(t))} for t in top]

@app.post("/predict", response_model=PredictResponse, tags=["inference"])
async def predict(req: PredictRequest, background_tasks: BackgroundTasks):
    """
    Example POST endpoint simulating an inference. Validates input with Pydantic.
    """
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text must not be empty")
    # run inference (could be CPU-bound — consider background tasks or job queue)
    predictions = _simple_inference(text, req.top_k)
    # Example: log asynchronously
    background_tasks.add_task(logger.info, f"Predicted for input length={len(text)} top_k={req.top_k}")
    return PredictResponse(input=text, top_k=req.top_k, predictions=predictions)

# Optional: run via `python -m app.main` for development (uses uvicorn)
if __name__ == "__main__":
    # dev server (not used when running under Gunicorn)
    uvicorn.run("app.main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=True)
