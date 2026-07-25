from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse

from .agent import run_agent
from .adk_adapter import adk_status
from .datasets import discover_dataset_paths, resolve_dataset_path
from .orchestration import build_agent_trace


class RunRequest(BaseModel):
    query: str = Field(min_length=1)
    data_path: str | None = None
    user_id: str | None = None
    memory_db: str | None = None
    memory_consent: bool = False
    provider: str | None = None
    model: str | None = None


WEB_UI = """<!doctype html>
<html><head><meta charset='utf-8'><title>Banking Agent Console</title>
<style>body{font-family:system-ui;background:#0f172a;color:#e2e8f0;max-width:1100px;margin:32px auto;padding:0 20px}textarea,input,button{font:inherit;border-radius:8px;border:1px solid #334155;padding:10px}textarea{width:100%;height:90px;background:#111827;color:#fff}input{background:#111827;color:#fff}button{background:#2563eb;color:#fff;cursor:pointer;margin-top:10px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}.panel{background:#1e293b;padding:16px;border-radius:12px;margin-top:16px}pre{white-space:pre-wrap;overflow:auto;background:#020617;padding:12px;border-radius:8px}.event{border-left:4px solid #38bdf8;padding:8px 12px;margin:8px 0;background:#0f172a}.muted{color:#94a3b8}</style></head>
<body><h1>Banking Segmentation Agent</h1><p class='muted'>Multi-agent trace, tool calls, governance checks, and JSON output.</p>
<textarea id='query'>Segment customers into priority, regular, and dormant groups and find conversion candidates</textarea>
<div class='grid'><label>Dataset<br><select id='dataset'><option value=''>Loading available datasets...</option></select><br><input id='path' placeholder='Optional custom CSV, ZIP, or folder path'></label><label>Provider<br><input id='provider' placeholder='none, gemini, openrouter, groq, ollama'></label></div>
<button onclick='run()'>Run agent</button><div id='status' class='panel'>Idle</div><div class='panel'><h2>Agent/tool calls</h2><div id='events'></div></div><div class='panel'><h2>Visualizations</h2><div id='images'></div></div><div class='panel'><h2>Final JSON response</h2><pre id='json'>{}</pre></div>
<script>async function loadDatasets(){try{const r=await fetch('/datasets');const items=await r.json();const select=document.getElementById('dataset');select.innerHTML='';items.forEach((item,i)=>{const o=document.createElement('option');o.value=item.path;o.textContent=item.name+(i===0?' (default)':'');select.appendChild(o)})}catch(e){document.getElementById('dataset').innerHTML='<option value="">Automatic demo fallback</option>'}}async function run(){const q=document.getElementById('query').value,custom=document.getElementById('path').value.trim(),path=custom||document.getElementById('dataset').value||null,provider=document.getElementById('provider').value||null;document.getElementById('events').innerHTML='';document.getElementById('images').innerHTML='';document.getElementById('json').textContent='';document.getElementById('status').textContent='Running...';const res=await fetch('/run/stream',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query:q,data_path:path,provider:provider})});const reader=res.body.getReader(),decoder=new TextDecoder();let buffer='';while(true){const x=await reader.read();if(x.done)break;buffer+=decoder.decode(x.value,{stream:true});const lines=buffer.split('\n');buffer=lines.pop();for(const line of lines){if(!line.startsWith('data:'))continue;const item=JSON.parse(line.slice(5));if(item.type==='agent_tool_call'){const el=document.createElement('div');el.className='event';el.innerHTML='<b>'+item.agent+'</b> → <b>'+item.tool+'</b><br><span class="muted">'+item.detail+'</span>';document.getElementById('events').appendChild(el)}if(item.type==='final'){document.getElementById('json').textContent=JSON.stringify(item.result,null,2);(item.result.artifacts||[]).filter(p=>p.toLowerCase().endsWith('.png')).forEach(p=>{const img=document.createElement('img');img.src='/artifact?path='+encodeURIComponent(p);img.alt=p;img.style='max-width:100%;margin:8px;border-radius:8px';document.getElementById('images').appendChild(img)});document.getElementById('status').textContent='Completed'}}}}}loadDatasets();</script></body></html>"""


def create_app(default_data_path: str = "data") -> FastAPI:
    app = FastAPI(title="Banking Segmentation Agent", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "banking-segmentation-agent", "adk": str(adk_status()["installed"])}

    @app.get("/ui", response_class=HTMLResponse)
    def ui() -> str:
        return WEB_UI

    @app.get("/datasets")
    def datasets() -> list[dict[str, str]]:
        # Ensure a clone always has one safe selectable default.
        resolve_dataset_path(default_data_path)
        return discover_dataset_paths()

    @app.get("/artifact")
    def artifact(path: str) -> FileResponse:
        """Serve only generated PNGs below an artifacts directory to the local UI."""
        candidate = Path(path).expanduser().resolve()
        if candidate.suffix.lower() != ".png" or "artifacts" not in candidate.parts:
            raise HTTPException(status_code=403, detail="Only generated PNG artifacts are available")
        if not candidate.is_file():
            raise HTTPException(status_code=404, detail="Artifact not found")
        return FileResponse(candidate, media_type="image/png", filename=candidate.name)

    @app.get("/.well-known/agent.json")
    def agent_card() -> dict[str, Any]:
        return {
            "name": "banking-segmentation-orchestrator",
            "description": "Multi-agent retail banking segmentation and personalization workflow",
            "version": "0.1.0",
            "url": "/a2a",
            "capabilities": {"streaming": True, "structured_output": True},
            "defaultInputModes": ["text/plain", "application/json"],
            "defaultOutputModes": ["application/json", "text/event-stream"],
            "skills": [
                {"id": "segment_customers", "name": "Segment customers"},
                {"id": "explain_segment", "name": "Explain segment"},
                {"id": "compare_segments", "name": "Compare segments"},
                {"id": "recommend_conversion", "name": "Recommend conversion candidates"},
            ],
        }

    @app.post("/run")
    def run(request: RunRequest) -> dict[str, Any]:
        try:
            result = run_agent(
                request.data_path or default_data_path,
                request.query,
                request.user_id,
                request.memory_db,
                request.memory_consent,
                request.provider,
                request.model,
            )
            result["agent_trace"] = build_agent_trace(result.get("events", []))
            return result
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Agent execution failed: {type(exc).__name__}") from exc

    @app.post("/run/stream")
    def run_stream(request: RunRequest) -> StreamingResponse:
        def events():
            yield "data: " + json.dumps({"type": "orchestrator_status", "status": "started"}) + "\n\n"
            try:
                result = run_agent(
                    request.data_path or default_data_path,
                    request.query,
                    request.user_id,
                    request.memory_db,
                    request.memory_consent,
                    request.provider,
                    request.model,
                )
                for item in build_agent_trace(result.get("events", [])):
                    yield "data: " + json.dumps(item, default=str) + "\n\n"
                result["agent_trace"] = build_agent_trace(result.get("events", []))
                yield "data: " + json.dumps({"type": "final", "result": result}, default=str) + "\n\n"
            except Exception as exc:
                yield "data: " + json.dumps({"type": "error", "detail": str(exc)}) + "\n\n"
        return StreamingResponse(events(), media_type="text/event-stream")

    @app.post("/a2a")
    def a2a(request: dict[str, Any]) -> dict[str, Any]:
        """A2A-friendly JSON-RPC façade; install [a2a] for a full SDK server."""
        params = request.get("params", request)
        query = params.get("query") or params.get("message") or params.get("text")
        if not query:
            return {"jsonrpc": "2.0", "id": request.get("id"), "error": {"code": -32602, "message": "query is required"}}
        result = run_agent(
            params.get("data_path") or default_data_path,
            query,
            params.get("user_id"),
            params.get("memory_db"),
            bool(params.get("memory_consent", False)),
            params.get("provider"),
            params.get("model"),
        )
        result["agent_trace"] = build_agent_trace(result.get("events", []))
        return {"jsonrpc": "2.0", "id": request.get("id"), "result": result}

    return app
