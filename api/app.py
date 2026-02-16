"""
FastAPI main application.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pathlib import Path
from api.routes import prediction

app = FastAPI(
    title="🩺 Breast Cancer Prediction API",
    description="ML-powered breast cancer diagnosis with 6 ensemble models",
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
    html_path = Path("templates/index.html")
    if html_path.exists():
        return html_path.read_text()
    return "<h1>UI not found. Please create templates/index.html</h1>"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

