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
