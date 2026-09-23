import json
import importlib
import os
import sqlite3
import csv
import re
from datetime import datetime

import numpy as np
from flask import Flask, flash, redirect, render_template, request, session, url_for
from PIL import Image
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

# TensorFlow dynamic import
try:
    tf = importlib.import_module("tensorflow")
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
except ModuleNotFoundError:
    tf = None
    preprocess_input = None

app = Flask(__name__)
app.secret_key = "secret123"

UPLOAD_FOLDER = os.path.join("static", "uploads")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
DB_PATH = "predictions.db"

MODEL_PATH = os.path.join("model", "skin_cancer_model.h5")
CLASS_NAMES_PATH = os.path.join("model", "class_names.json")
METADATA_PATH = os.path.join("HAM10000_metadata.csv", "HAM10000_metadata.csv")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# 🔥 Disease mapping
CLASS_DETAILS = {
    "nv": {
        "full_name": "Melanocytic Nevus (Benign Mole)",
        "risk_level": "Low Risk",
        "risk_color": "green",
        "description": "A common benign mole.",
        "recommendation": "Monitor the lesion.",
    },
    "mel": {
        "full_name": "Melanoma",
        "risk_level": "High Risk",
        "risk_color": "red",
        "description": "Dangerous skin cancer.",
        "recommendation": "Consult dermatologist immediately.",
    },
    "bcc": {
        "full_name": "Basal Cell Carcinoma",
        "risk_level": "Medium Risk",
        "risk_color": "orange",
        "description": "Slow growing cancer.",
        "recommendation": "Medical check required.",
    },
    "bkl": {
        "full_name": "Benign Keratosis",
        "risk_level": "Low Risk",
        "risk_color": "green",
        "description": "Usually non-cancerous.",
        "recommendation": "Monitor changes.",
    },
    "akiec": {
        "full_name": "Actinic Keratosis",
        "risk_level": "Medium Risk",
        "risk_color": "orange",
        "description": "Sun damage lesion.",
        "recommendation": "Consult doctor.",
    },
    "df": {
        "full_name": "Dermatofibroma",
        "risk_level": "Low Risk",
        "risk_color": "green",
        "description": "Benign lesion.",
        "recommendation": "No major concern.",
    },
    "vasc": {
        "full_name": "Vascular Lesion",
        "risk_level": "Low Risk",
        "risk_color": "green",
        "description": "Blood vessel lesion.",
        "recommendation": "Check if needed.",
    },
}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_model_and_classes():
    model = None
    class_names = []
    model_path = MODEL_PATH
    class_names_path = CLASS_NAMES_PATH

    # Handle duplicate download names like "file (1).ext"
    if not os.path.exists(model_path):
        alt_model_path = os.path.join("model", "skin_cancer_model (1).h5")
        if os.path.exists(alt_model_path):
            model_path = alt_model_path

    if not os.path.exists(class_names_path):
        alt_class_names_path = os.path.join("model", "class_names (1).json")
        if os.path.exists(alt_class_names_path):
            class_names_path = alt_class_names_path

    if tf is not None and os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)

    if os.path.exists(class_names_path):
        with open(class_names_path, "r") as f:
            class_names = json.load(f)

    return model, class_names


model, class_names = load_model_and_classes()


def load_ground_truth_labels():
    labels = {}
    if not os.path.exists(METADATA_PATH):
        return labels

    with open(METADATA_PATH, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row.get("image_id", "").strip()
            dx = row.get("dx", "").strip()
            if image_id and dx:
                labels[image_id] = dx
    return labels


GROUND_TRUTH_LABELS = load_ground_truth_labels()


# 🔥 IMPORTANT: MobileNet preprocessing
def preprocess_image(path):
    image = Image.open(path).convert("RGB")
    image = image.resize((224, 224))
    arr = np.array(image, dtype=np.float32)
    if preprocess_input is not None:
        arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    return arr


def get_class_info(code):
    return CLASS_DETAILS.get(code, {
        "full_name": code,
        "risk_level": "Medium Risk",
        "risk_color": "orange",
        "description": "Unknown condition",
        "recommendation": "Consult doctor",
    })


def extract_isic_id(filename):
    match = re.search(r"(ISIC_\d+)", filename)
    return match.group(1) if match else None


def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT NOT NULL,
            class_code TEXT NOT NULL,
            full_disease_name TEXT NOT NULL,
            confidence REAL NOT NULL,
            risk_level TEXT NOT NULL,
            created_at TEXT NOT NULL,
            user_id INTEGER,
            recommendation TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
init_db()


def get_user_by_email(email):
    conn = get_db_connection()
    cur = conn.cursor()
    user = cur.execute(
        "SELECT id, username, email, password_hash FROM users WHERE email = ?",
        (email,),
    ).fetchone()
    conn.close()
    return user


def verify_password(stored_hash, password):
    try:
        return check_password_hash(stored_hash, password)
    except ValueError:
        # Backward compatibility for any older plain-text values.
        return stored_hash == password


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        if not email or not password:
            flash("Email and password are required.", "danger")
            return render_template("login.html", error=None)

        user = get_user_by_email(email)
        if user and verify_password(user["password_hash"], password):
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            flash("Login successful.", "success")
            return redirect(url_for("home"))

        flash("Invalid email or password.", "danger")
    return render_template("login.html", error=None)


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")

        if not username or not email or not password:
            flash("Username, email, and password are required.", "danger")
            return render_template("register.html", error=None)

        if get_user_by_email(email):
            flash("Email already registered. Please login.", "danger")
            return redirect(url_for("login"))

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
            (username, email, generate_password_hash(password)),
        )
        conn.commit()
        conn.close()
        flash("Registration successful. Please login.", "success")
        return redirect(url_for("login"))
    return render_template("register.html", error=None)


@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully.", "success")
    return redirect(url_for("home"))


@app.route("/history")
def history():
    if not session.get("user_id"):
        flash("Please login to view your history.", "danger")
        return redirect(url_for("login"))

    conn = get_db_connection()
    cur = conn.cursor()
    user_id = session.get("user_id")
    rows = cur.execute(
        """
        SELECT image_path, class_code, full_disease_name, confidence, risk_level, created_at
        FROM predictions
        WHERE user_id = ?
        ORDER BY id DESC
        """,
        (user_id,),
    ).fetchall()

    conn.close()
    predictions = [dict(row) for row in rows]
    return render_template("history.html", predictions=predictions)


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return redirect("/")

    file = request.files["image"]

    if file.filename == "":
        return redirect("/")

    if not allowed_file(file.filename):
        return redirect("/")

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)
    image_path = f"uploads/{filename}"

    processed = preprocess_image(filepath)

    predicted_class = "nv"
    confidence = 0

    if model is not None:
        preds = model.predict(processed)
        idx = np.argmax(preds[0])
        if class_names and idx < len(class_names):
            predicted_class = class_names[idx]
        confidence = float(np.max(preds[0]) * 100)

    info = get_class_info(predicted_class)
    image_id = extract_isic_id(filename)
    true_class_code = GROUND_TRUTH_LABELS.get(image_id) if image_id else None
    true_class_info = get_class_info(true_class_code) if true_class_code else None
    is_prediction_correct = (
        predicted_class == true_class_code if true_class_code else None
    )
    image_accuracy = (
        100.0 if is_prediction_correct else 0.0 if is_prediction_correct is not None else None
    )

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO predictions (
            image_path, class_code, full_disease_name, confidence, risk_level, created_at, user_id, recommendation
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            image_path,
            predicted_class,
            info["full_name"],
            round(confidence, 2),
            info["risk_level"],
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            session.get("user_id"),
            info["recommendation"],
        ),
    )
    conn.commit()
    conn.close()

    return render_template(
        "result.html",
        image_path=image_path,
        predicted_class=info["full_name"],
        class_code=predicted_class,
        confidence=round(confidence, 2),
        risk_level=info["risk_level"],
        risk_color=info["risk_color"],
        disease_description=info["description"],
        recommendation=info["recommendation"],
        image_id=image_id,
        true_class_code=true_class_code,
        true_disease_name=true_class_info["full_name"] if true_class_info else None,
        is_prediction_correct=is_prediction_correct,
        image_accuracy=image_accuracy,
    )


if __name__ == "__main__":
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    init_db()
    app.run(debug=True)
