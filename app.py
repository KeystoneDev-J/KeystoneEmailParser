from flask import Flask, render_template, send_from_directory, url_for, request, jsonify
import logging
import os
from src.parsers.enhanced_parser import EnhancedParser
import asyncio

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
# Flask Application Initialization
# ----------------------------

# Create the Flask app
app = Flask(__name__)

# Define base directory
base_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(base_dir, "static")
templates_dir = os.path.join(base_dir, "templates")

# Set template and static directories
app.template_folder = templates_dir
app.static_folder = static_dir

# Initialize EnhancedParser
try:
    logger.info("Initializing EnhancedParser.")
    enhanced_parser = EnhancedParser()
    logger.info("EnhancedParser initialized successfully.")
except Exception as e:
    logger.critical(f"Failed to initialize EnhancedParser: {e}", exc_info=True)
    raise

# ----------------------------
# Routes Definitions
# ----------------------------

@app.route("/")
def index():
    """
    Renders the index.html template.

    Returns:
        Rendered HTML page.
    """
    logger.info("Rendering index page.")
    template_path = os.path.join(templates_dir, "index.html")
    if not os.path.exists(template_path):
        logger.critical(f"Template file not found at path: {template_path}")
        return "Template file not found", 500

    try:
        return render_template("index.html", static_url=url_for('static', filename=''))
    except Exception as e:
        logger.critical(f"Unhandled exception while rendering index page: {e}", exc_info=True)
        return "Internal Server Error", 500

@app.route("/static/<path:filename>")
def serve_static(filename):
    """
    Serves static files.

    Args:
        filename (str): The filename of the static file to serve.

    Returns:
        Static file.
    """
    return send_from_directory(static_dir, filename)

@app.route("/parse_email", methods=["POST"])
def parse_email():
    """
    Parses the provided email content.

    Returns:
        JSON response with parsed data or an error message.
    """
    logger.info("Parsing email content.")
    try:
        email_content = request.form.get("email_content")
        if not email_content:
            return jsonify({"error": "No email content provided"}), 400

        # Use EnhancedParser to parse the email content
        parsed_data = asyncio.run(enhanced_parser.parse(email_content))
        return jsonify(parsed_data)
    except Exception as e:
        logger.critical(f"Unhandled exception while parsing email: {e}", exc_info=True)
        return jsonify({"error": "Internal Server Error"}), 500

# ----------------------------
# Main Entry Point
# ----------------------------

if __name__ == "__main__":
    """
    Runs the Flask application.
    """
    # Read host and port from environment variables or use defaults
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))

    # Start the Flask server
    app.run(host=host, port=port, debug=True)
