print("Starting imports...")
import os
import logging
import json_log_formatter
from flask import Flask, render_template, request, jsonify, send_from_directory
from src.parsers.enhanced_parser import EnhancedParser

print("Flask imported successfully")

from dotenv import load_dotenv

print("All imports completed")

print("Initializing app...")

# Load environment variables from .env file
load_dotenv()
print(".env file loaded")

app = Flask(__name__)
print("Flask app created")

parser = EnhancedParser()
print("Parser payload initialized")

# ----------------------------
# Centralized Logging Setup
# ----------------------------

def create_logger():
    logger = logging.getLogger(__name__)
    json_handler = logging.StreamHandler()
    formatter = json_log_formatter.JSONFormatter()
    json_handler.setFormatter(formatter)
    logger.addHandler(json_handler)
    logger.setLevel(logging.DEBUG)
    return logger

app.logger = create_logger()

# ----------------------------
# Flask Routes
# ----------------------------

@app.route("/", methods=["GET"])
def index():
    """
    Render the main page with the email parsing form.
    """
    app.logger.info("Rendering index page.")
    return render_template("index.html")

@app.route("/favicon.ico")
def favicon():
    """
    Serve the favicon to prevent 404 errors.
    """
    favicon_path = os.path.join(app.root_path, "static", "favicon.ico")
    if os.path.exists(favicon_path):
        app.logger.info("Serving favicon.ico.")
        return send_from_directory(
            os.path.join(app.root_path, "static"),
            "favicon.ico",
            mimetype="image/vnd.microsoft.icon",
        )
    else:
        app.logger.warning("favicon.ico not found in static directory.")
        return jsonify({"error_message": "favicon.ico not found."}), 404

@app.route("/parse_email", methods=["POST"])
def parse_email_route():
    """
    Parse the email content and return the result.
    """
    try:
        email_content = request.form.get("email_content", "").strip()
        
        if not email_content:
            app.logger.warning("Empty email content received.")
            return jsonify({"error_message": "Please provide the email content to parse."}), 400

        parsed_data = parser.parse_email(email_content)
        app.logger.info("Email parsed successfully.")
        return jsonify(parsed_data), 200

    except ValueError as ve:
        app.logger.warning(f"Invalid input: {ve}")
        return jsonify({"error_message": str(ve)}), 400
    except Exception as e:
        app.logger.error(f"Error during parsing: {e}", exc_info=True)
        return jsonify({"error_message": "An unexpected error occurred during parsing."}), 500

# ----------------------------
# Error Handlers
# ----------------------------

@app.errorhandler(Exception)
def handle_exception(e):
    """
    Global exception handler to catch and log unexpected errors.
    """
    app.logger.error(f"Unhandled exception: {e}", exc_info=True)
    return jsonify({"error_message": "An internal error occurred."}), 500

@app.errorhandler(404)
def page_not_found(e):
    """
    Custom 404 error handler.
    """
    app.logger.warning(f"404 error: {request.url} not found.")
    return jsonify({"error_message": "The requested URL was not found on the server."}), 404

# ----------------------------
# Application Entry Point
# ----------------------------

print("App initialized, about to start server...")

if __name__ == "__main__":
    print("Starting Flask server...")
    try:
        # Optionally, get host and port from environment variables
        host = os.getenv("HOST", "127.0.0.1")
        port = int(os.getenv("PORT", 5000))

        # Ensure that the 'static' directory exists for favicon
        static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
        if not os.path.exists(static_dir):
            os.makedirs(static_dir)
            app.logger.info(
                f"Created 'static' directory at {static_dir}. Please add a favicon.ico file to this directory."
            )

        print(f"About to start Flask app on {host}:{port}")
        app.run(host=host, port=port, debug=True)  # Enable debug mode
    except Exception as e:
        print(f"Error starting Flask application: {e}")
        app.logger.critical(f"Failed to start the Flask application: {e}", exc_info=True)
        raise e