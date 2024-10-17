from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import logging
import os
from starlette.staticfiles import StaticFiles
from src.parsers.enhanced_parser import EnhancedParser

from contextlib import asynccontextmanager
import uvicorn

# ----------------------------
# Logging Configuration
# ----------------------------

# Configure logging to output JSON-like formatted logs
logging.basicConfig(
    format='{"time": "%(asctime)s", "name": "%(name)s", "level": "%(levelname)s", "message": "%(message)s"}',
    level=logging.INFO
)
logger = logging.getLogger("KeystoneEmailParser")

# ----------------------------
# Lifespan Context Manager
# ----------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles application startup and shutdown events.

    Initializes the EnhancedParser during startup and ensures graceful shutdown.
    """
    # Startup Phase
    logger.info("Application startup initiated.")
    try:
        # Initialize EnhancedParser
        logger.info("Initializing EnhancedParser.")
        app.state.parser = EnhancedParser()
        logger.info("EnhancedParser initialized successfully.")
    except Exception as e:
        logger.critical(f"Failed to initialize EnhancedParser: {e}", exc_info=True)
        raise
    yield
    # Shutdown Phase
    logger.info("Shutting down EnhancedParser.")
    try:
        await app.state.parser.shutdown()
        logger.info("EnhancedParser shut down successfully.")
    except Exception as e:
        logger.error(f"Error during EnhancedParser shutdown: {e}", exc_info=True)

# ----------------------------
# FastAPI Application Initialization
# ----------------------------

# Create the FastAPI app with the lifespan context manager
app = FastAPI(lifespan=lifespan)

# ----------------------------
# Static Files Mounting
# ----------------------------

# Mount static files directory with name 'static'
# This enables usage of url_for('static', filename='...') in templates
static_directory = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_directory), name="static")
logger.info(f"Static files directory mounted at '/static' from {static_directory}.")

# ----------------------------
# Templates Initialization
# ----------------------------

# Initialize Jinja2 templates
templates_directory = os.path.join(os.path.dirname(__file__), "templates")
templates = Jinja2Templates(directory=templates_directory)
logger.info(f"Templates directory initialized at {templates_directory}.")

# ----------------------------
# Routes Definitions
# ----------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """
    Renders the index.html template.

    Args:
        request (Request): The incoming request.

    Returns:
        HTMLResponse: The rendered HTML page.
    """
    logger.info("Rendering index page.")
    # Verify if the template file exists
    template_path = os.path.join(templates_directory, "index.html")
    if not os.path.exists(template_path):
        logger.critical(f"Template file not found at path: {template_path}")
        return HTMLResponse("Template file not found", status_code=500)

    try:
        response = templates.TemplateResponse("index.html", {"request": request})
        logger.info("index.html template successfully rendered.")
        return response
    except Exception as e:
        logger.critical(f"Unhandled exception while rendering index page: {e}", exc_info=True)
        raise

# ----------------------------
# Additional Routes (Optional)
# ----------------------------

# Example of an endpoint to handle email parsing
# Uncomment and modify as per your actual implementation

# @app.post("/parse-email", response_model=YourResponseModel)
# async def parse_email_endpoint(request: Request, email_content: str):
#     """
#     Endpoint to parse email content.

#     Args:
#         request (Request): The incoming request.
#         email_content (str): The raw email content to parse.

#     Returns:
#         YourResponseModel: The parsed data.
#     """
#     parser: EnhancedParser = request.app.state.parser
#     parsed_data = await parser.parse_email(email_content)
#     return parsed_data

# ----------------------------
# Main Entry Point
# ----------------------------

if __name__ == "__main__":
    """
    Runs the FastAPI application using Uvicorn.
    """
    # Read host and port from environment variables or use defaults
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    
    # Start the Uvicorn server
    uvicorn.run("app:app", host=host, port=port, log_level="info")
