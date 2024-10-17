print("Starting imports...")
import os
import logging
import json_log_formatter
from flask import Flask, render_template, request, jsonify, send_from_directory

print("Flask imported successfully")
from src.parsers.parser_options import ParserOption

print("ParserOption imported successfully")
from src.parsers.parser_registry import ParserRegistry

print("ParserRegistry imported successfully")
from dotenv import load_dotenv

print("All imports completed")

print("Initializing app...")

# Load environment variables from .env file
load_dotenv()
print(".env file loaded")

app = Flask(__name__)
print("Flask app created")

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
    try:
        # Retrieve form data
        email_content = request.form.get("email_content", "").strip()
        parser_option_str = request.form.get("parser_option", "").strip()

        # Input Validation
        if not email_content:
            logger.warning("No email content provided.")
            return (
                jsonify(
                    {"error_message": "Please provide the email content to parse."}
                ),
                400,
            )

        if not parser_option_str:
            logger.warning("No parser option selected.")
            return jsonify({"error_message": "Please select a parser option."}), 400

        logger.debug(f"Received email content: {email_content[:100]}...")
        logger.debug(
            f"Raw parser_option value: {parser_option_str} ({type(parser_option_str)})"
        )

        # Parse the parser_option into the Enum value
        try:
            parser_option = ParserOption[parser_option_str.upper()]
            logger.debug(
                f"Selected parser option Enum: {parser_option} ({type(parser_option)})"
            )
        except KeyError:
            logger.error(f"Invalid parser option selected: {parser_option_str}")
            return (
                jsonify(
                    {"error_message": f"Invalid parser option: {parser_option_str}"}
                ),
                400,
            )

        # Retrieve the appropriate parser from the registry
        parser = ParserRegistry.get_parser(parser_option)
        logger.info(
            f"Using parser: {parser.__class__.__name__} for option: {parser_option}"
        )

        # Parse the email content
        parsed_data = parser.parse_email(email_content, parser_option)

        logger.debug(f"Parsed data: {parsed_data}")

        return jsonify(parsed_data), 200

    except ValueError as ve:
        logger.error(f"ValueError during parsing: {ve}")
        return jsonify({"error_message": str(ve)}), 400
    except Exception as e:
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
            logger.info(
                f"Created 'static' directory at {static_dir}. Please add a favicon.ico file to this directory."
            )

        print(f"About to start Flask app on {host}:{port}")
        app.run(host=host, port=port, debug=True)  # Enable debug mode
    except Exception as e:
        print(f"Error starting Flask application: {e}")
        logger.critical(f"Failed to start the Flask application: {e}", exc_info=True)
        raise e
