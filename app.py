import os
import base64
import io
import json
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from pymongo import MongoClient
from bson import ObjectId
from dotenv import load_dotenv
from PIL import Image
import pytesseract
from sympy import sympify, Eq, solve, Symbol
import requests

# Load environment variables
load_dotenv()

# Config
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# MongoDB setup
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
mongo_client = MongoClient(MONGO_URI)
db = mongo_client["mathvision"]
collection = db["equations"]

# Gemini config (optional)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # set this to use Gemini
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "models/gemini-1.5-mini")  # change as needed
# Example endpoint template — you may need to adjust depending on your access & region
GEMINI_ENDPOINT = os.getenv(
    "GEMINI_ENDPOINT",
    "https://generativelanguage.googleapis.com/v1beta2/"  # placeholder prefix
)

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def save_file(file_storage):
    filename = secure_filename(file_storage.filename)
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    filename = f"{timestamp}_{filename}"
    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file_storage.save(path)
    return filename, path

def image_to_base64(path):
    with open(path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode("utf-8")


def call_gemini_extract_and_solve(image_b64: str):
    """
    Example template to call Gemini. Exact request shape may need adjustment
    depending on the Gemini API version you have access to.

    This function attempts to POST to an endpoint with API key in header.
    If your Gemini endpoint or model naming differs, update GEMINI_ENDPOINT / GEMINI_MODEL.
    """
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set")

    # Build a prompt: ask Gemini to return a JSON with fields `latex` and `solution`.
    prompt_text = (
        "You are a math assistant. Given the image attached, extract the mathematical "
        "equation in LaTeX format (key: latex) and solve it (key: solution). "
        "Return ONLY a JSON object with keys: latex, solution. Do not add extra text."
    )

    # NOTE: The exact JSON below is a template. Modify according to the Gemini API you're using.
    # Many Gemini endpoints accept a `prompt` and `image` or multimodal content; if yours differs,
    # adapt the payload accordingly.
    url = f"{GEMINI_ENDPOINT}models/{GEMINI_MODEL}:generate"

    headers = {
        "Authorization": f"Bearer {GEMINI_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "prompt": {
            "text": prompt_text,
            # If the API supports including images as base64 in the request structure, include it here.
            # Some Google endpoints accept multimodal content blocks; adjust to your model's API.
            "image_base64": image_b64
        },
        # optional parameters you might have access to:
        "maxOutputTokens": 1024,
        "temperature": 0.0
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    # The response shape depends on your Gemini model. Attempt a couple of common places to read text.
    # Adapt these fields if your response is different.
    text = None
    if isinstance(data, dict):
        # Try some likely fields
        if "candidates" in data and isinstance(data["candidates"], list) and len(data["candidates"]) > 0:
            # some endpoints return text in candidates[0]["content"]
            cand = data["candidates"][0]
            text = cand.get("content") or cand.get("text") or cand.get("output")
        if not text and "outputs" in data and isinstance(data["outputs"], list):
            # other endpoints return outputs[0].content
            out = data["outputs"][0]
            # a nested structure may exist
            if isinstance(out, dict):
                # attempt to flatten
                text = json.dumps(out)
            else:
                text = str(out)

    if not text:
        # Last attempt: try top-level "text" or "content"
        text = data.get("text") or data.get("content") or ""

    # Try to parse JSON out of the returned text
    latex = ""
    solution = ""
    try:
        # Sometimes the model returns inline JSON; try to find it
        parsed = json.loads(text)
        latex = parsed.get("latex", "")
        solution = parsed.get("solution", "")
    except Exception:
        # If not valid JSON, as a fallback, just return the raw text for inspection
        latex = ""
        solution = text.strip()

    return {"latex": latex, "solution": solution, "raw": data}
def fallback_ocr_and_sympy(path):
    """
    Fallback pipeline: use pytesseract to extract text, then attempt to sympify and solve.
    This is best-effort and works well for simple algebraic equations.
    """
    im = Image.open(path).convert("L")
    # Improve OCR with simple thresholding (optional)
    # im = im.point(lambda p: p > 140 and 255)

    ocr_text = pytesseract.image_to_string(im, config="--psm 6")
    ocr_text = ocr_text.strip()

    # Attempt to find an equation in the OCR text (simple heuristics)
    # Replace unicode division signs, etc.
    cleaned = ocr_text.replace("−", "-").replace("×", "*").replace("^", "**")
    cleaned = cleaned.replace(" ", "")

    latex = cleaned  # best effort (not actual LaTeX)
    solution = ""
    try:
        # If string contains '=' try to solve for a variable, else try to evaluate/simplify
        if "=" in cleaned:
            lhs, rhs = cleaned.split("=", 1)
            expr_l = sympify(lhs)
            expr_r = sympify(rhs)
            # attempt to solve symbolically; find symbols used
            symbols = list(expr_l.free_symbols.union(expr_r.free_symbols))
            if symbols:
                var = symbols[0]
                sol = solve(Eq(expr_l, expr_r), var)
                solution = str(sol)
            else:
                solution = str(sympify(lhs - expr_r))
        else:
            # try to evaluate / simplify expression
            expr = sympify(cleaned)
            solution = str(expr.simplify())
    except Exception as e:
        solution = f"Could not solve with SymPy: {e}"

    return {"latex": latex, "solution": solution, "ocr_text": ocr_text}

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/upload", methods=["POST"])
def upload():
    """
    Handle the form:
      - required field: name
      - required: file (image)
    """
    name = request.form.get("name", "").strip()
    if not name:
        return jsonify({"error": "Name is required."}), 400

    if "image" not in request.files:
        return jsonify({"error": "Image file missing."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No selected file."}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed."}), 400

    filename, path = save_file(file)
    record = {
        "name": name,
        "filename": filename,
        "createdAt": datetime.utcnow()
    }

    # Try Gemini first if key provided
    gemini_result = None
    try:
        if GEMINI_API_KEY:
            image_b64 = image_to_base64(path)
            gemini_result = call_gemini_extract_and_solve(image_b64)
            # If gemini_result returns solution/latex, use them
            if gemini_result and (gemini_result.get("latex") or gemini_result.get("solution")):
                record["latex"] = gemini_result.get("latex", "")
                record["solution"] = gemini_result.get("solution", "")
                record["raw_gemini"] = gemini_result.get("raw")
                inserted_id = collection.insert_one(record).inserted_id
                return jsonify({
                    "id": str(inserted_id),
                    "name": name,
                    "filename": filename,
                    "latex": record["latex"],
                    "solution": record["solution"],
                    "source": "gemini"
                })
            # else fall through to fallback
    except Exception as e:
        # Log but don't crash — use fallback
        print("Gemini error:", e)

    # Fallback: local OCR (Tesseract) + SymPy
    try:
        fallback = fallback_ocr_and_sympy(path)
        record["latex"] = fallback.get("latex", "")
        record["solution"] = fallback.get("solution", "")
        record["ocr_text"] = fallback.get("ocr_text", "")
        record["source"] = "fallback"
        inserted_id = collection.insert_one(record).inserted_id
        return jsonify({
            "id": str(inserted_id),
            "name": name,
            "filename": filename,
            "latex": record["latex"],
            "solution": record["solution"],
            "source": "fallback"
        })
    except Exception as e:
        print("Fallback error:", e)
        return jsonify({"error": "Failed to process image"}), 500

