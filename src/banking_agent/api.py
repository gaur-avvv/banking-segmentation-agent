from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .agent import run_agent


class RunRequest(BaseModel):
    query: str = Field(min_length=1)
    data_path: str | None = None
    user_id: str | None = None
    memory_db: str | None = None
    memory_consent: bool = False
    provider: str | None = None
    model: str | None = None


def create_app(default_data_path: str = "data") -> FastAPI:
    app = FastAPI(title="Banking Segmentation Agent", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "banking-segmentation-agent"}

    @app.post("/run")
    def run(request: RunRequest) -> dict[str, Any]:
        try:
            return run_agent(
                request.data_path or default_data_path,
                request.query,
                request.user_id,
                request.memory_db,
                request.memory_consent,
                request.provider,
                request.model,
            )
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Agent execution failed: {type(exc).__name__}") from exc

    return app
