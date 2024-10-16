import os
import logging
import json_log_formatter
from flask import Flask, render_template, request, jsonify, send_from_directory
from src.parsers.parser_options import ParserOption
from src.parsers.parser_registry import ParserRegistry
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# ----------------------------
# Centralized Logging Setup
# ----------------------------

# Initialize JSON formatter for structured logging
formatter = json_log_formatter.JSONFormatter()

# Create a stream handler with the JSON formatter
json_handler = logging.StreamHandler()
json_handler.setFormatter(formatter)

# Get the root logger and set the level to DEBUG
logger = logging.getLogger()
logger.addHandler(json_handler)
logger.setLevel(logging.DEBUG)  # Set to DEBUG for detailed logs

# ----------------------------
# Flask Routes
# ----------------------------


@app.route("/", methods=["GET"])
def index():
    """
    Render the main page with the email parsing form.
    """
    logger.info("Rendering index page.")
    return render_template("index.html")


@app.route("/favicon.ico")
def favicon():
    """
    Serve the favicon to prevent 404 errors.
    """
    favicon_path = os.path.join(app.root_path, "static", "favicon.ico")
    if os.path.exists(favicon_path):
        logger.info("Serving favicon.ico.")
        return send_from_directory(
            os.path.join(app.root_path, "static"),
            "favicon.ico",
            mimetype="image/vnd.microsoft.icon",
        )
    else:
        logger.warning("favicon.ico not found in static directory.")
        return jsonify({"error_message": "favicon.ico not found."}), 404


@app.route("/parse_email", methods=["POST"])
def parse_email_route():
    """
    Handle the email parsing request.

    Expects form data with:
    - 'email_content': The raw email content to parse.
    - 'parser_option': The selected parser option.

    Returns:
    - JSON response with parsed data or error messages.
    """
    try:
        # Retrieve form data
        email_content = request.form.get("email_content", "").strip()
        parser_option = request.form.get("parser_option", "").strip()

        # Input Validation
        if not email_content:
            logger.warning("No email content provided.")
            return (
                jsonify(
                    {"error_message": "Please provide the email content to parse."}
                ),
                400,
            )

        if not parser_option:
            logger.warning("No parser option selected.")
            return jsonify({"error_message": "Please select a parser option."}), 400

        logger.debug(
            f"Received email content: {email_content[:100]}..."
        )  # Log first 100 chars
        logger.debug(
            f"Raw parser_option value: {parser_option} ({type(parser_option)})"
        )

        # Parse the parser_option into the Enum value
        try:
            selected_parser_option = ParserOption(parser_option)
            logger.debug(
                f"Selected parser option Enum: {selected_parser_option} ({type(selected_parser_option)})"
            )
        except ValueError:
            logger.error(f"Invalid parser option selected: {parser_option}")
            return (
                jsonify({"error_message": f"Invalid parser option: {parser_option}"}),
                400,
            )

        # Retrieve the appropriate parser from the registry
        parser = ParserRegistry.get_parser(selected_parser_option)
        logger.info(
            f"Using parser: {parser.__class__.__name__} for option: {selected_parser_option}"
        )

        # Parse the email content
        parsed_data = parser.parse_email(email_content, selected_parser_option)

        logger.debug(f"Parsed data: {parsed_data}")

        return jsonify(parsed_data), 200

    except ValueError as ve:
        # Handle known value errors (e.g., invalid parser option)
        logger.error(f"ValueError during parsing: {ve}")
        return jsonify({"error_message": str(ve)}), 400
    except Exception as e:
        # Handle unexpected errors
        logger.error(f"Unexpected error during parsing: {e}", exc_info=True)
        return (
            jsonify({"error_message": "An unexpected error occurred during parsing."}),
            500,
        )


# ----------------------------
# Global Exception Handler
# ----------------------------


@app.errorhandler(Exception)
def handle_exception(e):
    """
    Global exception handler to catch and log unexpected errors.
    """
    logger.error(f"Unhandled exception: {e}", exc_info=True)
    return jsonify({"error_message": "An internal error occurred."}), 500


# ----------------------------
# Custom 404 Error Handler
# ----------------------------


@app.errorhandler(404)
def page_not_found(e):
    """
    Custom 404 error handler.
    """
    logger.warning(f"404 error: {request.url} not found.")
    return (
        jsonify({"error_message": "The requested URL was not found on the server."}),
        404,
    )


# ----------------------------
# Application Entry Point
# ----------------------------

if __name__ == "__main__":
    try:
        # Optionally, get host and port from environment variables
        host = os.getenv("HOST", "127.0.0.1")
        port = int(os.getenv("PORT", 5000))

        # Ensure that the 'static' directory exists for favicon
        static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
        if not os.path.exists(static_dir):
            os.makedirs(static_dir)
            logger.info(
                f"Created 'static' directory at {static_dir}. Please add a favicon.ico file to this directory."
            )

        logger.info(f"Starting Flask app on {host}:{port}")
        app.run(host=host, port=port, debug=True)
    except Exception as e:
        logger.critical(f"Failed to start the Flask application: {e}", exc_info=True)
        raise e
