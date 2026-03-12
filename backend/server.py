import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.model import EmotionModel  # noqa: E402

DATA_DIR = ROOT / "data" / "processed"
MODEL_PATH = ROOT / "models" / "best_model.pt"


def load_arrays():
  X = np.load(DATA_DIR / "X.npy")            # (N, 168, 128, 1)
  y = np.load(DATA_DIR / "y.npy")
  classes = np.load(DATA_DIR / "classes.npy", allow_pickle=True)
  return X, y, classes


def pick_device():
  if torch.backends.mps.is_available():
    return torch.device("mps")
  if torch.cuda.is_available():
    return torch.device("cuda")
  return torch.device("cpu")


def run_inference(X: np.ndarray, model: EmotionModel, device: torch.device):
  # Returns predictions, per-sample confidences, latency stats
  probs_list: List[np.ndarray] = []
  preds_list: List[np.ndarray] = []
  per_sample_ms: List[float] = []

  BATCH = 64
  model.eval()
  with torch.no_grad():
    for i in range(0, len(X), BATCH):
      xb = torch.from_numpy(X[i:i + BATCH]).float().to(device)
      start = time.perf_counter()
      out = model(xb)
      elapsed = time.perf_counter() - start
      per_sample_ms.extend([elapsed / len(xb) * 1000] * len(xb))
      probs = torch.softmax(out, dim=1).cpu().numpy()
      probs_list.append(probs)
      preds_list.append(probs.argmax(1))

  probs = np.concatenate(probs_list, axis=0)
  preds = np.concatenate(preds_list, axis=0)
  latency_p95 = float(np.percentile(per_sample_ms, 95))
  latency_avg = float(np.mean(per_sample_ms))
  return preds, probs, latency_avg, latency_p95


def build_dashboard_payload() -> Dict:
  X_raw, y, classes = load_arrays()
  X = X_raw.transpose(0, 3, 1, 2)  # (N, 1, 168, 128)

  device = pick_device()
  model = EmotionModel(num_classes=len(classes)).to(device)
  state = torch.load(MODEL_PATH, map_location=device)
  model.load_state_dict(state)

  preds, probs, latency_avg, latency_p95 = run_inference(X, model, device)
  accuracy = float((preds == y).mean() * 100)

  # Per-class stats
  emotions = []
  palette = {
    "angry": "#FF7E6B",
    "calm": "#7CC7FF",
    "disgust": "#FF9EC4",
    "fearful": "#FFCF7D",
    "happy": "#5CF0B5",
    "neutral": "#9AA4FF",
    "sad": "#FF8FB1",
    "surprised": "#F9D65C",
  }
  for idx, name in enumerate(classes):
    mask = preds == idx
    support = int(mask.sum())
    confidence = float(probs[mask, idx].mean() * 100) if support else 0.0
    emotions.append({
      "name": name.title(),
      "color": palette.get(name, "#7CC7FF"),
      "confidence": confidence,
      "latency_ms": latency_avg,
      "support": support,
    })

  # Trend: rolling detection volume over dataset order
  window = 120
  volumes = [int(len(chunk)) for chunk in np.array_split(y, max(1, len(y)//window))]
  trend_points = []
  running = 0
  for v in volumes:
    running += v
    trend_points.append(running)

  # Events derived from real stats
  least_class = emotions[np.argmin([e["support"] for e in emotions])]
  best_class = emotions[np.argmax([e["confidence"] for e in emotions])]
  events = [
    {
      "title": f"Accuracy measured at {accuracy:.2f}%",
      "meta": f"{len(y)} labeled samples evaluated",
      "tone": "primary",
    },
    {
      "title": f"Fastest latency p95: {latency_p95:.1f} ms",
      "meta": "Measured over full dataset",
      "tone": "success",
    },
    {
      "title": f"Under-represented class: {least_class['name']}",
      "meta": f"{least_class['support']} samples available",
      "tone": "warn",
    },
    {
      "title": f"Strongest class: {best_class['name']}",
      "meta": f"Avg confidence {best_class['confidence']:.1f}%",
      "tone": "neutral",
    },
  ]

  return {
    "updated_at": time.time(),
    "summary": {
      "detections": int(len(y)),
      "accuracy": accuracy,
      "latency_p95_ms": latency_p95,
      "latency_avg_ms": latency_avg,
      "uptime": 99.4,  # placeholder until live service uptime is tracked
    },
    "trend": {
      "points": trend_points,
      "forecast_change_pct": 0.0,
    },
    "emotions": emotions,
    "events": events,
    "classes": list(classes),
  }


app = FastAPI(title="Emotion Dashboard API")
app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],
  allow_methods=["*"],
  allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
  app.state.cache = build_dashboard_payload()


@app.get("/api/dashboard")
async def read_dashboard():
  return app.state.cache


if __name__ == "__main__":
  import uvicorn

  uvicorn.run(
    "backend.server:app",
    host="0.0.0.0",
    port=8000,
    reload=False,
    workers=1,
  )
