"""
FastAPI main application.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from api.routes import prediction
from config import BASE_DIR

app = FastAPI(
    title="🩺 Breast Cancer Prediction API",
    description="Breast cancer diagnosis with Reinforcement Learning Trees (RLT), our Neural RLT and 5 baseline tree models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(prediction.router, prefix="/api", tags=["Prediction"])

# Serve HTML interface
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main HTML interface."""
    html_path = BASE_DIR / "templates" / "index.html"
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    return "<h1>UI not found. Please create templates/index.html</h1>"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.app:app", host="127.0.0.1", port=8000, reload=True)

