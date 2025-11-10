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

# Optional converters
try:
    from latex2sympy2 import latex2sympy
    HAVE_LATEX2SYMPY2 = True
except Exception:
    HAVE_LATEX2SYMPY2 = False
    
# Pix2Tex import attempts (API surface varies with versions)
PIX2TEX_AVAILABLE = False
LatexOCR = None
try:
    from pix2tex.cli import LatexOCR as _LatexOCR
    LatexOCR = _LatexOCR
    PIX2TEX_AVAILABLE = True
except Exception:
    try:
        from pix2tex import LatexOCR as _LatexOCR
        LatexOCR = _LatexOCR
        PIX2TEX_AVAILABLE = True
    except Exception:
        PIX2TEX_AVAILABLE = False
# Load .env

load_dotenv()
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
PORT = int(os.getenv("PORT", 5000))

# App + DB
app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

mongo_client = MongoClient(MONGO_URI)
db = mongo_client["mathvision"]
collection = db["equations"]

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mathvision")

# Tesseract detection/config (fallback)
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
    logger.info(f"Tesseract configured: {TESSERACT_PATH}")
else:
    logger.info("Tesseract not found (fallback will be unavailable until installed).")

# -------------------------
# Helpers: files & images
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

def image_to_base64_bytes(pil_image):
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# Preprocess OCR text -> SymPy-friendly

def preprocess_text_for_sympy(s: str) -> str:
    if not s:
        return s
    s = s.strip()
    s = s.replace("−", "-").replace("–", "-").replace("—", "-")
    s = s.replace("×", "*").replace("⋅", "*").replace("^", "**")
    s = s.replace(",", "")
    s = re.sub(r"[^\w\*\+\-\/=\^\(\)\.\*]", " ", s)
    s = s.strip()
    s = re.sub(r"(?<=\d)(?=[A-Za-z])", "*", s)
    s = re.sub(r"(?<=[A-Za-z])(?=\d)", "*", s)
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
# Pix2Tex wrapper

def run_pix2tex_on_image(path_or_pil):
    if not PIX2TEX_AVAILABLE:
        raise RuntimeError("Pix2Tex not installed (pip install pix2tex).")
    # instantiate OCR (can download weights on first run)
    ocr = LatexOCR()
    try:
        if isinstance(path_or_pil, str):
            latex = ocr(path_or_pil)
        else:
            buf = io.BytesIO()
            path_or_pil.save(buf, format="PNG")
            buf.seek(0)
            # many versions accept file-like objects
            latex = ocr(buf)
        if isinstance(latex, (list, tuple)):
            latex = latex[0]
        return (latex or "").strip()
    except Exception as e:
        raise

# Convert LaTeX -> SymPy and attempt solve
def solve_latex_with_sympy(latex_str: str):
    if not latex_str:
        return {"success": False, "error": "Empty LaTeX"}
    s = latex_str.strip()
    s = s.replace(r"\cdot", "*").replace(r"\times", "*").replace(r"\,", "")
    s = re.sub(r"\\\[|\\\]|\\\(|\\\)", "", s)
    s = re.sub(r"^\$+|\$+$", "", s)
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

# Tesseract fallback solver
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

# Routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

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

        # 1) Pix2Tex (preferred)
        pix_ok = False
        if PIX2TEX_AVAILABLE:
            try:
                pix_latex = run_pix2tex_on_image(proc_img)
                if pix_latex:
                    solved = solve_latex_with_sympy(pix_latex)
                    if solved.get("success"):
                        record["latex"] = pix_latex
                        record["solution"] = solved.get("solution")
                        record["sympy_expr"] = solved.get("sympy_expr")
                        record["source"] = "pix2tex+sympy"
                    else:
                        record["latex"] = pix_latex
                        record["solution"] = f"Could not convert LaTeX to SymPy: {solved.get('error')}"
                        record["source"] = "pix2tex"
                    pix_ok = True
            except Exception as e:
                logger.warning("Pix2Tex failed: %s", e)
                pix_ok = False

        # 2) Fallback: Tesseract + SymPy
        if not pix_ok:
            try:
                if TESSERACT_PATH:
                    fallback = fallback_ocr_and_sympy(raw_img)
                    record["latex"] = fallback.get("latex", "")
                    record["solution"] = fallback.get("solution", "")
                    record["ocr_text"] = fallback.get("ocr_text", "")
                    record["preprocessed_text"] = fallback.get("preprocessed")
                    record["source"] = "fallback"
                else:
                    record["latex"] = ""
                    record["solution"] = "No Pix2Tex available and Tesseract is not installed."
                    record["source"] = "none"
            except Exception as e:
                logger.exception("Fallback error")
                return jsonify({"error": "Processing failed. Check server logs."}), 500

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
        logger.exception("Upload exception")
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

if __name__ == "__main__":
    logger.info("Starting MathVision (Pix2Tex) app...")
    logger.info("PIX2TEX_AVAILABLE=%s, TESSERACT_PATH=%s", PIX2TEX_AVAILABLE, TESSERACT_PATH)
    app.run(host="0.0.0.0", port=PORT, debug=True)