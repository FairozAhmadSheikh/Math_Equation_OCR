# app.py — Complete updated Flask app (Gemini image + fallback + preprocessing + model listing)
import os
import io
import json
import base64
import shutil
import subprocess
import platform
from datetime import datetime

from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from pymongo import MongoClient
from PIL import Image, ImageFilter, ImageOps

import pytesseract
from sympy import sympify, Eq, solve
import google.generativeai as genai

# -------------------------
# Load env
# -------------------------
load_dotenv()

# -------------------------
# Config
# -------------------------
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")            # required to use Gemini
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-mini")  # set to an available model ID
PORT = int(os.getenv("PORT", 5000))

# -------------------------
# App + DB init
# -------------------------
app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

mongo_client = MongoClient(MONGO_URI)
db = mongo_client["mathvision"]
collection = db["equations"]

# -------------------------
# Configure Gemini SDK (if key present)
# -------------------------
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# -------------------------
# Tesseract detection/config
# -------------------------
def locate_tesseract_executable():
    # 1) check PATH
    which_path = shutil.which("tesseract")
    if which_path:
        return which_path

    # 2) common Windows paths
    if platform.system().lower().startswith("win"):
        common_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            r"C:\Tesseract-OCR\tesseract.exe",
        ]
        for p in common_paths:
            if os.path.isfile(p):
                return p

    # 3) macOS Homebrew
    for p in ["/usr/local/bin/tesseract", "/opt/homebrew/bin/tesseract"]:
        if os.path.isfile(p):
            return p

    # 4) linux
    if os.path.isfile("/usr/bin/tesseract"):
        return "/usr/bin/tesseract"

    return None

def check_and_configure_tesseract():
    tpath = locate_tesseract_executable()
    if not tpath:
        return None
    try:
        pytesseract.pytesseract.tesseract_cmd = tpath
    except Exception:
        # fallback attribute
        try:
            pytesseract.tesseract_cmd = tpath
        except Exception:
            pass
    # verify
    try:
        proc = subprocess.run([tpath, "--version"], capture_output=True, text=True, timeout=5)
        if proc.returncode == 0:
            return tpath
    except Exception:
        pass
    return None

TESSERACT_PATH = check_and_configure_tesseract()
if TESSERACT_PATH:
    print(f"[INFO] Tesseract configured: {TESSERACT_PATH}")
else:
    print("[WARN] Tesseract not found. Fallback OCR will not be available until it's installed or added to PATH.")

# -------------------------
# Helpers: files, images
# -------------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def save_file(storage):
    filename = secure_filename(storage.filename)
    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    filename = f"{ts}_{filename}"
    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    storage.save(path)
    return filename, path

def image_to_base64_bytes(pil_image):
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# -------------------------
# Image preprocessing (improves OCR & Gemini)
# -------------------------
def preprocess_image_for_ocr(pil_img, resize_scale=2, median_filter=3, autocontrast=True):
    """
    Convert to grayscale, optionally upscale, denoise, and autocontrast.
    Returns a PIL.Image.
    """
    img = pil_img.convert("L")
    if resize_scale and resize_scale > 1:
        w, h = img.size
        img = img.resize((w * resize_scale, h * resize_scale), Image.LANCZOS)
    if median_filter and median_filter > 0:
        img = img.filter(ImageFilter.MedianFilter(size=median_filter))
    if autocontrast:
        img = ImageOps.autocontrast(img)
    return img

# -------------------------
# Gemini: list models helper (best-effort; SDK shapes vary)
# -------------------------
def list_gemini_models():
    if not GEMINI_API_KEY:
        return {"error": "GEMINI_API_KEY not set"}
    try:
        # Try SDK helper
        if hasattr(genai, "list_models"):
            models = genai.list_models()
            return {"models": models}
        # Some SDKs expose models.list or genai.models.list()
        if hasattr(genai, "models") and hasattr(genai.models, "list"):
            models = genai.models.list()
            return {"models": models}
    except Exception as e:
        return {"error": str(e)}
    return {"error": "Could not list models via SDK; check AI Studio web UI."}

# Expose a simple route to list available models (handy for debugging)
@app.route("/list_models", methods=["GET"])
def http_list_models():
    res = list_gemini_models()
    return jsonify(res)

# -------------------------
# Gemini: send image + prompt, request JSON in return
# -------------------------
def call_gemini_image_to_json(image_b64: str, model: str = None, timeout: int = 30):
    """
    Uses google-generativeai SDK models.generate_content to send a prompt + image.
    Tries to parse a JSON response (with keys latex, solution). Returns dict.
    """
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not configured")

    model_to_use = model or GEMINI_MODEL

    prompt_text = (
        "You are a math assistant. Given the attached image, extract the mathematical equation in LaTeX "
        "(JSON key: latex) and provide a concise solution (JSON key: solution)."
        " Return ONLY a JSON object with keys: latex, solution and no additional commentary."
    )

    try:
        response = genai.models.generate_content(
            model=model_to_use,
            content=[
                {"type": "text", "text": prompt_text},
                {"type": "image", "image_base64": image_b64}
            ],
            temperature=0.0,
            max_output_tokens=1024,
        )
    except Exception as e:
        raise

    # Extract text blocks — SDK response shapes vary; try multiple fallbacks
    text_output = ""
    try:
        # Many SDKs return response.candidates[0].content which is a list of blocks
        if getattr(response, "candidates", None):
            candidate = response.candidates[0]
            blocks = getattr(candidate, "content", [])
            parts = []
            for b in blocks:
                # block may be dict or object
                if isinstance(b, dict):
                    if "text" in b:
                        parts.append(b["text"])
                else:
                    t = getattr(b, "text", None)
                    if t:
                        parts.append(t)
                    else:
                        parts.append(str(b))
            text_output = "\n".join(parts).strip()
        else:
            # fallback to string form of response
            text_output = str(response)
    except Exception:
        text_output = str(response)

    # Try parsing JSON directly
    latex = ""
    solution = ""
    raw_json = None
    try:
        parsed = json.loads(text_output)
        raw_json = parsed
        latex = parsed.get("latex", "")
        solution = parsed.get("solution", "")
    except Exception:
        # try to find a JSON substring
        try:
            start = text_output.index("{")
            end = text_output.rindex("}") + 1
            snippet = text_output[start:end]
            parsed = json.loads(snippet)
            raw_json = parsed
            latex = parsed.get("latex", "")
            solution = parsed.get("solution", "")
        except Exception:
            # no JSON found — set solution to entire text for inspection
            solution = text_output

    return {"latex": latex, "solution": solution, "raw_text": text_output, "raw_json": raw_json, "raw_response": str(response)}

# -------------------------
# Fallback OCR + SymPy solver
# -------------------------
def fallback_ocr_and_sympy(pil_img):
    if not TESSERACT_PATH:
        raise RuntimeError("Tesseract not configured/available for fallback")

    proc_img = preprocess_image_for_ocr(pil_img, resize_scale=2)
    ocr_text = pytesseract.image_to_string(proc_img, config="--psm 6").strip()
    cleaned = ocr_text.replace("−", "-").replace("×", "*").replace("^", "**")
    cleaned_nospace = cleaned.replace(" ", "")

    latex = cleaned
    solution = ""
    try:
        if "=" in cleaned_nospace:
            lhs, rhs = cleaned_nospace.split("=", 1)
            expr_l = sympify(lhs)
            expr_r = sympify(rhs)
            # attempt to solve for first symbol found
            syms = list(expr_l.free_symbols.union(expr_r.free_symbols))
            if syms:
                var = syms[0]
                sol = solve(Eq(expr_l, expr_r), var)
                solution = str(sol)
            else:
                solution = str(expr_l - expr_r)
        else:
            expr = sympify(cleaned_nospace)
            solution = str(expr.simplify())
    except Exception as e:
        solution = f"SymPy error: {e}"

    return {"latex": latex, "solution": solution, "ocr_text": ocr_text}

# -------------------------
# Routes
# -------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

@app.route("/upload", methods=["POST"])
def upload():
    """
    Expected form fields:
      - name (required)
      - image (file, required)
    """
    try:
        name = (request.form.get("name") or "").strip()
        if not name:
            return jsonify({"error": "Name is required."}), 400

        if "image" not in request.files:
            return jsonify({"error": "Image file missing."}), 400

        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "No selected file."}), 400
        if not allowed_file(file.filename):
            return jsonify({"error": "File type not allowed."}), 400

        # Save file
        filename, path = save_file(file)

        # Open image and preprocess
        raw_img = Image.open(path).convert("RGB")
        proc_img = preprocess_image_for_ocr(raw_img, resize_scale=2)

        record = {
            "name": name,
            "filename": filename,
            "createdAt": datetime.utcnow(),
            "source": None
        }

        # First: try Gemini (image -> JSON)
        gemini_ok = False
        gemini_result = None
        if GEMINI_API_KEY:
            try:
                image_b64 = image_to_base64_bytes(proc_img)
                gemini_result = call_gemini_image_to_json(image_b64, model=GEMINI_MODEL)
                # if gemini_result contains latex or solution, accept it
                if (gemini_result.get("latex") or gemini_result.get("solution")):
                    gemini_ok = True
                    record["latex"] = gemini_result.get("latex", "")
                    record["solution"] = gemini_result.get("solution", "")
                    record["raw_gemini_text"] = gemini_result.get("raw_text", "")
                    record["source"] = "gemini"
            except Exception as e:
                # log and fall through to fallback
                print("[ERROR] Gemini call failed:", repr(e))

        # If Gemini not available or didn't provide results: fallback to Tesseract+SymPy
        if not gemini_ok:
            try:
                fallback = fallback_ocr_and_sympy(raw_img)
                record["latex"] = fallback.get("latex", "")
                record["solution"] = fallback.get("solution", "")
                record["ocr_text"] = fallback.get("ocr_text", "")
                record["source"] = "fallback"
            except Exception as e:
                print("[ERROR] Fallback failed:", repr(e))
                return jsonify({"error": "Failed to process image. Ensure Gemini key is valid or Tesseract is installed."}), 500

        # Save record
        inserted = collection.insert_one(record).inserted_id

        # Return result
        return jsonify({
            "id": str(inserted),
            "name": name,
            "filename": filename,
            "latex": record.get("latex", ""),
            "solution": record.get("solution", ""),
            "source": record.get("source", "")
        })

    except Exception as e:
        print("[ERROR] Upload exception:", repr(e))
        return jsonify({"error": "Internal server error"}), 500

@app.route("/history", methods=["GET"])
def history():
    docs = []
    for doc in collection.find().sort("createdAt", -1).limit(200):
        doc["_id"] = str(doc["_id"])
        # convert datetime to iso string if present
        if isinstance(doc.get("createdAt"), datetime):
            doc["createdAt"] = doc["createdAt"].isoformat()
        docs.append(doc)
    return jsonify(docs)

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    print("[INFO] Starting MathVision app...")
    if GEMINI_API_KEY:
        print("[INFO] GEMINI_API_KEY provided. GEMINI_MODEL:", GEMINI_MODEL)
    else:
        print("[INFO] No GEMINI_API_KEY; falling back to local OCR+SymPy only.")
    app.run(host="0.0.0.0", port=PORT, debug=True)
