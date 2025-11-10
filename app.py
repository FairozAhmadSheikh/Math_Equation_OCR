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
