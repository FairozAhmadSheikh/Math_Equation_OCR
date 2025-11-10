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
