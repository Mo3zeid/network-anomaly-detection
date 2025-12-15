"""
FastAPI application for Network Anomaly Detection.
"""
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import router, load_models
from .sniffer import router as sniffer_router
from .firewall import router as firewall_router
from .rules import router as rules_router
from .evaluation import router as evaluation_router
from .intelligence import router as intelligence_router
from .models import HealthResponse
from ..utils.logger import get_logger

logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Network Anomaly Detection API",
    description="""
    API for detecting network anomalies using machine learning.
    
    ## Features
    - Statistical anomaly detection (Z-score, IQR)
    - Isolation Forest ML detection
    - Attack type classification
    - Real-time alerts
    - IP Blocking via Windows Firewall
    - Configurable detection rules
    - Model evaluation metrics
    - IP Intelligence & Geolocation
    
    ## Endpoints
    - `/detect` - Analyze network flows for anomalies
    - `/stats` - Get detection statistics
    - `/alerts` - View security alerts
    - `/simulate` - Generate sample traffic for testing
    - `/sniffer` - Real-time packet capture controls
    - `/firewall` - IP blocking controls
    - `/rules` - Detection rules configuration
    - `/evaluate` - Model performance metrics
    - `/ip/{ip}/details` - IP intelligence lookup
    - `/live-stats` - Real-time dashboard data
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include router
app.include_router(router, prefix="/api", tags=["detection"])
app.include_router(sniffer_router, prefix="/api", tags=["sniffer"])
app.include_router(firewall_router, prefix="/api", tags=["firewall"])
app.include_router(rules_router, prefix="/api", tags=["rules"])
app.include_router(evaluation_router, prefix="/api", tags=["evaluation"])
app.include_router(intelligence_router, prefix="/api", tags=["intelligence"])


@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    logger.info("Starting Network Anomaly Detection API...")
    load_models()
    logger.info("API startup complete")


@app.get("/", tags=["root"])
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Network Anomaly Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check():
    """Health check endpoint."""
    from .routes import models
    
    models_loaded = any(models.values())
    
    return HealthResponse(
        status="healthy" if models_loaded else "degraded",
        models_loaded=models_loaded,
        timestamp=datetime.now()
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
