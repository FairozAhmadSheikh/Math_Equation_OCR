import os
import io
import base64
import subprocess
import platform
import shutil
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from pymongo import MongoClient
from PIL import Image
import pytesseract
import google.generativeai as genai
from sympy import sympify, solve, Symbol
from dotenv import load_dotenv

# -------------------------------
# Load environment variables
# -------------------------------
load_dotenv()

# Initialize Flask
app = Flask(__name__)

# -------------------------------
# MongoDB Setup
# -------------------------------
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
client = MongoClient(MONGO_URI)
db = client["math_solver"]
collection = db["equations"]

# -------------------------------
# Gemini Setup
# -------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")
else:
    gemini_model = None

# -------------------------------
# Tesseract Detection & Setup
# -------------------------------
def locate_tesseract_executable():
    """Find a working tesseract binary across platforms."""
    # 1) Check PATH
    which_path = shutil.which("tesseract")
    if which_path:
        return which_path

    # 2) Common Windows paths
    if platform.system().lower().startswith("win"):
        common_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files\Tesseract\tesseract.exe",
            r"C:\Tesseract-OCR\tesseract.exe"
        ]
        for p in common_paths:
            if os.path.isfile(p):
                return p

    # 3) macOS Homebrew paths
    if platform.system().lower().startswith("darwin"):
        for p in ["/usr/local/bin/tesseract", "/opt/homebrew/bin/tesseract"]:
            if os.path.isfile(p):
                return p

    # 4) Linux path
    if os.path.isfile("/usr/bin/tesseract"):
        return "/usr/bin/tesseract"

    return None


def check_and_configure_tesseract():
    """Configure pytesseract to use the located binary."""
    tpath = locate_tesseract_executable()
    if not tpath:
        return False
    pytesseract.pytesseract.tesseract_cmd = tpath
    try:
        proc = subprocess.run([tpath, "--version"], capture_output=True, text=True, timeout=5)
        return proc.returncode == 0
    except Exception:
        return False


TESSERACT_AVAILABLE = check_and_configure_tesseract()
if TESSERACT_AVAILABLE:
    print(f"Tesseract found and configured: {pytesseract.pytesseract.tesseract_cmd}")
else:
    print("⚠️ Tesseract not found. Please install it or add it to PATH.")

# -------------------------------
# Flask Routes
# -------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    try:
        name = request.form.get("name")
        if not name:
            return jsonify({"error": "Name is required."}), 400

        image = request.files["image"]
        if not image:
            return jsonify({"error": "No image uploaded."}), 400

        # Read the uploaded image
        img_bytes = image.read()
        img = Image.open(io.BytesIO(img_bytes))

        # Perform OCR
        if not TESSERACT_AVAILABLE:
            return jsonify({"error": "Tesseract not available."}), 500

        extracted_text = pytesseract.image_to_string(img, config="--psm 6")
        if not extracted_text.strip():
            return jsonify({"error": "No text detected in image."}), 400

        print("Extracted equation:", extracted_text)

        # Try Gemini first
        result = None
        if gemini_model:
            try:
                prompt = f"Solve this mathematical equation: {extracted_text}"
                gemini_response = gemini_model.generate_content(prompt)
                result = gemini_response.text.strip()
            except Exception as e:
                print(f"Gemini error: {e}")
                result = None

        # Fallback to sympy if Gemini fails
        if not result:
            try:
                expr = sympify(extracted_text)
                x = Symbol("x")
                sol = solve(expr, x)
                result = f"Solution: {sol}"
            except Exception as e:
                result = f"Unable to solve using fallback: {str(e)}"

        # Save in MongoDB
        record = {
            "name": name,
            "equation": extracted_text,
            "solution": result,
            "timestamp": datetime.utcnow()
        }
        collection.insert_one(record)

        return jsonify({"equation": extracted_text, "result": result})

    except Exception as e:
        print("Error:", e)
        return jsonify({
            "error": "Failed to process image. Ensure Gemini key is valid or Tesseract is installed."
        }), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
