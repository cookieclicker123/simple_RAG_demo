import uvicorn
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware # For potential CORS configuration

from src.server.routes import router as api_router
from src.config import settings # For log level or other app-wide settings
# Import the lifespan event handlers
from src.server.services import initialize_global_chat_engine, shutdown_global_chat_engine 

logger = logging.getLogger(__name__)
logging.basicConfig(level=settings.log_level)

# Initialize FastAPI app
app = FastAPI(
    title="Simple RAG API",
    description="API for the Simple RAG QA Pipeline with streaming.",
    version="0.1.0",
    # You can add more OpenAPI metadata here
    # docs_url="/api/docs", # Default is /docs
    # redoc_url="/api/redoc" # Default is /redoc
)

@app.on_event("startup")
async def startup_event():
    logger.info("Application startup: Initializing resources...")
    await initialize_global_chat_engine() # Call the async initializer
    logger.info("Application startup: Resources initialized.")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Application shutdown: Cleaning up resources...")
    await shutdown_global_chat_engine() # Call the async shutdown handler
    logger.info("Application shutdown: Resources cleaned up.")

# Include middlewares (e.g., CORS)
# Adjust origins as needed for your frontend if it's on a different domain/port
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for simplicity, restrict in production
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods - Corrected
    allow_headers=["*"]  # Allows all headers - Corrected
)

# Include the API router
# All routes defined in api_router will be prefixed with /api
app.include_router(api_router, prefix="/api")

# Health check endpoint for the main application
@app.get("/health", tags=["Application Health"])
async def app_health_check():
    return {"status": "Application is healthy"}

# Main entry point to run the Uvicorn server
if __name__ == "__main__":
    logger.info("Starting Uvicorn server for Simple RAG API...")
    # It's common to configure host and port via environment variables or config
    # For now, hardcoding to localhost:8000 as requested.
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level=settings.log_level.lower())
    # Note: When running uvicorn directly like this, changes to code might not auto-reload
    # unless uvicorn is run with --reload, e.g., uvicorn src.server.web_app:app --reload
    # For development, `uvicorn src.server.web_app:app --reload --host 0.0.0.0 --port 8000` from the project root is typical. 