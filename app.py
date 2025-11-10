import os
import io
import re
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
import logging

# optional latex->sympy converter (better if installed)
try:
    from latex2sympy2 import latex2sympy
    HAVE_LATEX2SYMPY2 = True
except Exception:
    HAVE_LATEX2SYMPY2 = False

# -------------------------
# Load environment
# -------------------------
load_dotenv()
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
# MathPix credentials (preferred for math OCR)
MATHPIX_APP_ID = os.getenv("MATHPIX_APP_ID")
MATHPIX_APP_KEY = os.getenv("MATHPIX_APP_KEY")
MATHPIX_ENDPOINT = "https://api.mathpix.com/v3/text"

# Optional Gemini REST fallback
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-mini")
GEMINI_ENDPOINT_BASE = os.getenv("GEMINI_ENDPOINT", "https://generativelanguage.googleapis.com/v1beta2").rstrip("/")

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
# Tesseract detection/config (fallback)
# -------------------------
def locate_tesseract_executable():
    which_path = shutil.which("tesseract")
    if which_path:
        return which_path
    if platform.system().lower().startswith("win"):
        common_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            r"C:\Tesseract-OCR\tesseract.exe",
        ]
        for p in common_paths:
            if os.path.isfile(p):
                return p
    for p in ["/usr/local/bin/tesseract", "/opt/homebrew/bin/tesseract", "/usr/bin/tesseract"]:
        if os.path.isfile(p):
            return p
    return None

def check_and_configure_tesseract():
    tpath = locate_tesseract_executable()
    if not tpath:
        return None
    try:
        pytesseract.pytesseract.tesseract_cmd = tpath
    except Exception:
        try:
            pytesseract.tesseract_cmd = tpath
        except Exception:
            pass
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
    print("[WARN] Tesseract not found. Fallback OCR will not be available until installed or added to PATH.")

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
# Image preprocessing
# -------------------------
def preprocess_image_for_ocr(pil_img, resize_scale=2, median_filter=3, autocontrast=True):
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
# Text preprocessing for SymPy
# -------------------------
def preprocess_text_for_sympy(s: str) -> str:
    if not s:
        return s
    s = s.strip()
    s = s.replace("−", "-").replace("–", "-").replace("—", "-")
    s = s.replace("×", "*").replace("⋅", "*").replace("^", "**")
    s = s.replace(",", "")
    # remove weird characters but keep letters, digits, math symbols
    s = re.sub(r"[^\w\*\+\-\/=\^\(\)\.\*]", " ", s)
    s = s.strip()
    # insert explicit multiplication: digit-letter and letter-digit
    s = re.sub(r"(?<=\d)(?=[A-Za-z])", "*", s)
    s = re.sub(r"(?<=[A-Za-z])(?=\d)", "*", s)
    # handle adjacent letters: insert * except common function names
    funcs = r"(sin|cos|tan|log|exp|sqrt|ln|sec|csc|cot|asin|acos|atan)"
    def insert_between_letters(match):
        left, right = match.group(1), match.group(2)
        comb = (left + right).lower()
        if re.match(rf"^{funcs}", comb):
            return left + right
        return left + "*" + right
    s = re.sub(r"([A-Za-z])([A-Za-z])", insert_between_letters, s)
    s = s.replace(" ", "")
    return s

# -------------------------
# MathPix API call (preferred OCR for math)
# -------------------------
def call_mathpix(image_b64: str, formats: str = "latex_simplified"):
    if not (MATHPIX_APP_ID and MATHPIX_APP_KEY):
        raise RuntimeError("MathPix credentials not configured in .env (MATHPIX_APP_ID/MATHPIX_APP_KEY)")
    headers = {
        "app_id": MATHPIX_APP_ID,
        "app_key": MATHPIX_APP_KEY,
        "Content-type": "application/json"
    }
    payload = {
        "src": f"data:image/png;base64,{image_b64}",
        "formats": [formats, "text"],
        "ocr": ["math", "text"]
    }
    r = requests.post(MATHPIX_ENDPOINT, headers=headers, json=payload, timeout=30)
    r.raise_for_status()
    data = r.json()
    latex = data.get("latex_simplified") or data.get("latex") or ""
    text = data.get("text", "")
    return {"latex": latex, "text": text, "raw": data}

# -------------------------
# Try to convert LaTeX -> SymPy and solve
# -------------------------
def solve_latex_with_sympy(latex_str: str):
    if not latex_str:
        return {"success": False, "error": "Empty LaTeX"}
    s = latex_str.strip()
    s = s.replace(r"\cdot", "*").replace(r"\times", "*").replace(r"\,", "")
    s = re.sub(r"\\\[|\\\]|\\\(|\\\)", "", s)
    s = re.sub(r"^\$+|\$+$", "", s)
    # Try latex2sympy2 if available
    if HAVE_LATEX2SYMPY2:
        try:
            expr = latex2sympy(s)
            if "=" in s:
                lhs, rhs = s.split("=", 1)
                L = latex2sympy(lhs); R = latex2sympy(rhs)
                syms = list(L.free_symbols.union(R.free_symbols))
                if syms:
                    sol = solve(Eq(L, R), syms[0])
                    return {"success": True, "sympy_expr": str(expr), "solution": str(sol)}
                else:
                    return {"success": True, "sympy_expr": str(expr), "solution": str(L - R)}
            else:
                return {"success": True, "sympy_expr": str(expr), "solution": str(expr.simplify())}
        except Exception as e:
            latex2_err = str(e)
    else:
        latex2_err = "latex2sympy2 not installed"
    # Try sympy.parse_latex if available
    try:
        from sympy.parsing.latex import parse_latex
        try:
            parsed = parse_latex(s)
            if "=" not in s:
                return {"success": True, "sympy_expr": str(parsed), "solution": str(parsed.simplify())}
        except Exception as e2:
            parse_err = str(e2)
    except Exception:
        parse_err = "sympy.parse_latex not available"
    # Fallback: preprocess -> sympify
    try:
        prepped = preprocess_text_for_sympy(s)
        if "=" in prepped:
            lhs, rhs = prepped.split("=", 1)
            L = sympify(lhs); R = sympify(rhs)
            syms = list(L.free_symbols.union(R.free_symbols))
            if syms:
                sol = solve(Eq(L, R), syms[0])
                return {"success": True, "sympy_expr": f"{lhs}={rhs}", "solution": str(sol)}
            else:
                return {"success": True, "sympy_expr": f"{lhs}={rhs}", "solution": str(L - R)}
        else:
            expr = sympify(prepped)
            return {"success": True, "sympy_expr": str(expr), "solution": str(expr.simplify())}
    except Exception as final_e:
        return {"success": False, "error": f"latex2_err={latex2_err}, parse_err={parse_err if 'parse_err' in locals() else None}, final_sympify_err={final_e}"}

# -------------------------
# Gemini REST fallback (optional) - normalized endpoint/model
# -------------------------
def normalized_gemini_endpoint_and_model():
    endpoint = GEMINI_ENDPOINT_BASE.rstrip("/")
    model = (GEMINI_MODEL or "").strip()
    if model.startswith("models/"):
        model = model.split("/", 1)[1]
    model = model.lstrip("/")
    return endpoint, model

def call_gemini_image_to_json_rest(image_b64: str, model: str = None, timeout: int = 30):
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not configured")
    endpoint, normalized_model = normalized_gemini_endpoint_and_model()
    model_to_use = model or normalized_model
    url = f"{endpoint}/models/{model_to_use}:generate"
    headers = {"Authorization": f"Bearer {GEMINI_API_KEY}", "Content-Type": "application/json; charset=utf-8"}
    prompt_text = (
        "You are a math assistant. Given the attached image, extract the mathematical equation in LaTeX "
        "(JSON key: latex) and provide a concise solution (JSON key: solution). Return ONLY a JSON object."
    )
    body = {
        "prompt": {
            "messages": [
                {"author": "user", "content": [{"type": "text", "text": prompt_text},
                                               {"type": "image", "image": {"imageBytes": image_b64}}]}
            ]
        },
        "temperature": 0.0,
        "maxOutputTokens": 1024
    }
    resp = requests.post(url, headers=headers, json=body, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    # Extract textual output
    text_output = ""
    if isinstance(data, dict):
        if "candidates" in data and data["candidates"]:
            cand = data["candidates"][0]
            if isinstance(cand, dict) and "content" in cand:
                blocks = cand["content"]
                parts = []
                for b in blocks:
                    if isinstance(b, dict) and "text" in b:
                        parts.append(b["text"])
                    else:
                        parts.append(str(b))
                text_output = "\n".join(parts).strip()
            else:
                text_output = json.dumps(cand)
        elif "output" in data:
            text_output = json.dumps(data["output"])
        else:
            text_output = json.dumps(data)
    else:
        text_output = str(data)
    latex = ""; solution = ""; raw_json = None
    try:
        parsed = json.loads(text_output)
        raw_json = parsed
        latex = parsed.get("latex", ""); solution = parsed.get("solution", "")
    except Exception:
        try:
            start = text_output.index("{"); end = text_output.rindex("}") + 1
            snippet = text_output[start:end]; parsed = json.loads(snippet)
            raw_json = parsed; latex = parsed.get("latex", ""); solution = parsed.get("solution", "")
        except Exception:
            solution = text_output
    return {"latex": latex, "solution": solution, "raw_text": text_output, "raw_json": raw_json, "raw_response": data}

# -------------------------
# Fallback OCR + SymPy
# -------------------------
def fallback_ocr_and_sympy(pil_img):
    if not TESSERACT_PATH:
        raise RuntimeError("Tesseract not configured/available for fallback")
    proc_img = preprocess_image_for_ocr(pil_img, resize_scale=2)
    ocr_text = pytesseract.image_to_string(proc_img, config="--psm 6").strip()
    prepped = preprocess_text_for_sympy(ocr_text)
    latex = ocr_text
    solution = ""
    try:
        if "=" in prepped:
            lhs, rhs = prepped.split("=", 1)
            L = sympify(lhs); R = sympify(rhs)
            syms = list(L.free_symbols.union(R.free_symbols))
            if syms:
                sol = solve(Eq(L, R), syms[0])
                solution = str(sol)
            else:
                solution = str(L - R)
        else:
            expr = sympify(prepped)
            solution = str(expr.simplify())
    except Exception as e:
        solution = f"SymPy error: {e} (preprocessed='{prepped}')"
    return {"latex": latex, "solution": solution, "ocr_text": ocr_text, "preprocessed": prepped}

# -------------------------
# Routes
# -------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

@app.route("/list_models", methods=["GET"])
def http_list_models():
    if not GEMINI_API_KEY:
        return jsonify({"error": "GEMINI_API_KEY not set"}), 400
    url = f"{GEMINI_ENDPOINT_BASE}/models"
    headers = {"Authorization": f"Bearer {GEMINI_API_KEY}"}
    try:
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        return jsonify(r.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/upload", methods=["POST"])
def upload():
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

        filename, path = save_file(file)
        raw_img = Image.open(path).convert("RGB")
        proc_img = preprocess_image_for_ocr(raw_img, resize_scale=2)

        record = {"name": name, "filename": filename, "createdAt": datetime.utcnow(), "source": None}

        # 1) Try MathPix (preferred)
        mathpix_ok = False
        if MATHPIX_APP_ID and MATHPIX_APP_KEY:
            try:
                image_b64 = image_to_base64_bytes(proc_img)
                mp = call_mathpix(image_b64)
                latex_mp = (mp.get("latex") or "").strip()
                text_mp = (mp.get("text") or "").strip()
                record["raw_mathpix"] = mp.get("raw")
                if latex_mp:
                    solved = solve_latex_with_sympy(latex_mp)
                    if solved.get("success"):
                        record["latex"] = latex_mp
                        record["solution"] = solved.get("solution")
                        record["sympy_expr"] = solved.get("sympy_expr")
                        record["source"] = "mathpix+sympy"
                    else:
                        record["latex"] = latex_mp
                        record["solution"] = f"Could not convert LaTeX to SymPy: {solved.get('error')}"
                        record["source"] = "mathpix"
                    mathpix_ok = True
                elif text_mp:
                    # attempt to parse MathPix plain text
                    prepped = preprocess_text_for_sympy(text_mp)
                    try:
                        if "=" in prepped:
                            lhs, rhs = prepped.split("=", 1)
                            L = sympify(lhs); R = sympify(rhs)
                            syms = list(L.free_symbols.union(R.free_symbols))
                            if syms:
                                sol = solve(Eq(L, R), syms[0])
                                record["latex"] = text_mp
                                record["solution"] = str(sol)
                                record["source"] = "mathpix_text+sympy"
                            else:
                                record["latex"] = text_mp
                                record["solution"] = str(L - R)
                                record["source"] = "mathpix_text"
                        else:
                            expr = sympify(prepped)
                            record["latex"] = text_mp
                            record["solution"] = str(expr.simplify())
                            record["source"] = "mathpix_text+sympy"
                        mathpix_ok = True
                    except Exception as e:
                        record["latex"] = text_mp
                        record["solution"] = f"MathPix text returned but SymPy conversion failed: {e}"
                        record["source"] = "mathpix_text"
                        mathpix_ok = True
            except Exception as e:
                print("[WARN] MathPix call failed:", repr(e))
                mathpix_ok = False

        # 2) If MathPix didn't provide results, try Gemini (optional) then Tesseract fallback
        gemini_ok = False
        if not mathpix_ok and GEMINI_API_KEY:
            try:
                image_b64 = image_to_base64_bytes(proc_img)
                gem = call_gemini_image_to_json_rest(image_b64, model=GEMINI_MODEL)
                if gem.get("latex") or gem.get("solution"):
                    record["latex"] = gem.get("latex", "")
                    record["solution"] = gem.get("solution", "")
                    record["raw_gemini"] = gem.get("raw_text", "")
                    record["source"] = "gemini"
                    gemini_ok = True
            except Exception as e:
                print("[WARN] Gemini call failed:", repr(e))
                gemini_ok = False

        if not mathpix_ok and not gemini_ok:
            # final fallback: tesseract + sympy
            try:
                fallback = fallback_ocr_and_sympy(raw_img)
                record["latex"] = fallback.get("latex", "")
                record["solution"] = fallback.get("solution", "")
                record["ocr_text"] = fallback.get("ocr_text", "")
                record["preprocessed_text"] = fallback.get("preprocessed", "")
                record["source"] = "fallback"
            except Exception as e:
                print("[ERROR] Fallback failed:", repr(e))
                return jsonify({"error": "Failed to process image. Ensure MathPix/Gemini keys or Tesseract installed."}), 500

        inserted = collection.insert_one(record).inserted_id
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
        if isinstance(doc.get("createdAt"), datetime):
            doc["createdAt"] = doc["createdAt"].isoformat()
        docs.append(doc)
    return jsonify(docs)

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    print("[INFO] Starting MathVision app...")
    print("[INFO] MathPix configured:" , bool(MATHPIX_APP_ID and MATHPIX_APP_KEY))
    print("[INFO] Gemini configured:" , bool(GEMINI_API_KEY))
    app.run(host="0.0.0.0", port=PORT, debug=True)
