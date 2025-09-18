import os
os.environ["KERAS_BACKEND"] = "torch"

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import io
import base64
import logging
from datetime import datetime
from functools import wraps
import time

from flask import Flask, render_template, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename

import flask_monitoringdashboard as dashboard

import numpy as np
import keras

from pymongo import MongoClient
import urllib.parse
from PIL import Image

# ---------------- Config ----------------
UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXT = {"png", "jpg", "jpeg", "webp"}
CLASSES = ['desert', 'forest', 'meadow', 'mountain']

app = Flask(__name__)

# ✅ Créer le dossier de logs AVANT de configurer le logging
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
logs_dir = os.path.join(BASE_DIR, 'logs')
os.makedirs(logs_dir, exist_ok=True)

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(logs_dir, 'app.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ✅ Ajouter un formateur de log avec contexte requête
class RequestFormatter(logging.Formatter):
    def format(self, record):
        from flask import has_request_context, request
        if has_request_context():
            record.url = request.url
            record.remote_addr = request.remote_addr
        else:
            record.url = record.remote_addr = "N/A"
        return super().format(record)

# Appliquer le formateur à tous les handlers
for handler in logger.handlers:
    handler.setFormatter(RequestFormatter(
        '[%(asctime)s] %(levelname)s | %(remote_addr)s | %(url)s | %(message)s'
    ))

# ✅ Charger les variables d'environnement depuis .env en local (optionnel)
env_path = os.path.join(os.path.dirname(BASE_DIR), '.env')  # ← Chemin vers la racine du projet
if os.path.exists(env_path):
    from dotenv import load_dotenv
    load_dotenv(env_path)  # ← Charge le .env depuis la racine
    logger.info("✅ Variables d'environnement chargées depuis .env")

# ✅ Alerting par email (uniquement en production)
if os.getenv('FLASK_ENV') == 'production':
    # Vérifier que toutes les variables nécessaires sont présentes
    required_vars = ['ALERT_EMAIL', 'ALERT_EMAIL_PASSWORD']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"❌ Variables d'environnement manquantes pour l'alerting: {missing_vars}")
    else:
        try:
            from logging.handlers import SMTPHandler
            mail_handler = SMTPHandler(
                mailhost=('smtp.gmail.com', 587),  # ✅ Tuple avec port
                fromaddr=os.getenv('ALERT_EMAIL'),
                toaddrs=[os.getenv('ALERT_EMAIL_RECIPIENT', os.getenv('ALERT_EMAIL'))],
                subject='🚨 ERREUR CRITIQUE - App Classification',
                credentials=(os.getenv('ALERT_EMAIL'), os.getenv('ALERT_EMAIL_PASSWORD')),
                secure=()  # ✅ Utilise STARTTLS
            )
            mail_handler.setLevel(logging.ERROR)
            mail_handler.setFormatter(logging.Formatter(
                '[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
            ))
            logger.addHandler(mail_handler)  # ✅ Ajouté au logger configuré, pas app.logger
            logger.info("✅ Alerting par email activé pour les erreurs critiques.")
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la configuration de l'alerting: {e}", exc_info=True)
else:
    logger.info("ℹ️ Mode développement - Alerting email désactivé")

MONGO_URI = os.getenv("MONGO_URI")
if MONGO_URI:
    try:
        client = MongoClient(MONGO_URI)
        db = client.flask_feedback
        feedback_collection = db.user_feedback
        logger.info("✅ Connexion à MongoDBAtlas établie.")
    except Exception as e:
        logger.error(f"❌ Erreur de connexion àMongoDBAtlas : {e}")
else:
    logger.warning("⚠️ MONGO_URI non définie — feedback non enregistré.")

# Configuration explicite avant bind
def get_or_generate_security_token():
    """Récupère ou génère un token de sécurité"""
    token = os.getenv('FLASK_MONITORING_TOKEN')
    if not token:
        import secrets
        token = secrets.token_urlsafe(48)
        logger.warning(f"⚠️ Token de sécurité généré automatiquement: {token}")
        logger.warning("⚠️ Ajoutez FLASK_MONITORING_TOKEN={} à vos variables d'environnement pour la production!".format(token))
    return token

# Configuration du dashboard
config_path = os.path.join(BASE_DIR, 'config.cfg')
if os.path.exists(config_path):
    # Charger d'abord la configuration depuis le fichier
    dashboard.config.init_from(file=config_path)
    logger.info("✅ Configuration dashboard chargée depuis config.cfg")
    
    # Puis override les paramètres sensibles depuis les variables d'environnement
    dashboard.config.username = os.getenv('DASHBOARD_USERNAME', dashboard.config.username)
    dashboard.config.password = os.getenv('DASHBOARD_PASSWORD', dashboard.config.password)
else:
    # Configuration par défaut si pas de fichier
    dashboard.config.username = os.getenv('DASHBOARD_USERNAME', 'admin')
    dashboard.config.password = os.getenv('DASHBOARD_PASSWORD', 'admin')
    logger.info("⚠️ Configuration dashboard par défaut appliquée")

# Le token de sécurité est toujours géré programmatiquement (priorité aux variables d'env)
dashboard.config.security_token = get_or_generate_security_token()

# ✅ CORRECTION: Configuration base de données SQLite pour le dashboard de monitoring
# Flask-MonitoringDashboard utilise SQLAlchemy et ne supporte que les DB relationnelles
dashboard_db_path = os.path.join(BASE_DIR, 'monitoring_dashboard.db')
dashboard.config.database_name = f'sqlite:///{dashboard_db_path}'
dashboard.config.table_prefix = 'fmd_'
logger.info(f"✅ Configuration SQLite dashboard appliquée: {dashboard_db_path}")

# Vérification de la configuration avant bind
logger.info(f"Dashboard config - Username: {dashboard.config.username}")
logger.info(f"Dashboard config - Database: {getattr(dashboard.config, 'database_name', 'Not set')}")
logger.info(f"Dashboard config - Token configuré: {'Oui' if dashboard.config.security_token else 'Non'}")

dashboard.bind(app)
logger.info("✅ Flask-MonitoringDashboard initialisé")

# Note: MongoDB reste utilisé pour votre application (feedback utilisateur)
# SQLite est utilisé uniquement pour les métriques de monitoring

# ---------------- Model ----------------
# Chemin absolu vers le modèle (relatif au fichier app.py)
MODEL_PATH = os.path.join(BASE_DIR, "models", "final_cnn.keras")

# ✅ Optimisations mémoire PyTorch (à ajouter AVANT d'importer keras ou de charger le modèle)
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# Désactiver les optimisations CUDA (même si GPU absent)
if torch.cuda.is_available():
    torch.cuda.empty_cache()

try:
    model = keras.saving.load_model(MODEL_PATH, compile=False)
    logger.info(f"Modèle chargé avec succès depuis {MODEL_PATH}")
    
    # Vérification des dimensions d'entrée du modèle
    input_shape = model.input_shape
    logger.info(f"Dimensions d'entrée du modèle: {input_shape}")
    
    if len(input_shape) >= 3:
        MODEL_HEIGHT = input_shape[1] if input_shape[1] is not None else 224
        MODEL_WIDTH = input_shape[2] if input_shape[2] is not None else 224
    else:
        MODEL_HEIGHT = MODEL_WIDTH = 224
        
    logger.info(f"Dimensions cibles pour le preprocessing: {MODEL_HEIGHT}x{MODEL_WIDTH}")

    # ✅ "Warm up" du modèle (à ajouter APRES le chargement)
    # Crée un tenseur vide de la bonne forme et fait une prédiction factice
    logger.info("Warming up model...")
    dummy_input = np.zeros((1, MODEL_HEIGHT, MODEL_WIDTH, 3), dtype=np.float32)
    _ = model.predict(dummy_input, verbose=0)
    logger.info("Model warmed up successfully.")

except Exception as e:
    logger.error(f"Erreur lors du chargement du modèle: {e}")
    raise

# ---------------- Utils ----------------
def allowed_file(filename: str) -> bool:
    """Vérifie si le nom de fichier possède une extension autorisée.
    La vérification est **insensible à la casse** et ne regarde que la sous-chaîne
    après le dernier point. Dépend de la constante globale `ALLOWED_EXT`.

    Args:
        filename: Nom du fichier soumis (ex. "photo.PNG").

    Returns:
        True si l'extension (ex. "png", "jpg") est dans `ALLOWED_EXT`, sinon False.

    Examples:
        >>> ALLOWED_EXT = {"png", "jpg", "jpeg", "webp"}
        >>> allowed_file("img.JPG")
        True
        >>> allowed_file("archive.tar.gz")
        False
    """
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def to_data_url(pil_img: Image.Image, fmt="JPEG") -> str:
    """Convertit une image PIL en Data URL base64 affichable dans un <img src="...">.
    L'image est encodée en mémoire (sans I/O disque), sérialisée en base64, puis
    encapsulée comme `data:<mime>;base64,<payload>`. Le type MIME est déduit de `fmt`.

    Args:
        pil_img: Image PIL à encoder.
        fmt: Format d'encodage PIL (ex. "JPEG", "PNG"). Par défaut "JPEG".

    Returns:
        Chaîne Data URL prête à être insérée dans une balise <img>.

    Raises:
        ValueError: si la sauvegarde PIL échoue pour le format demandé.

    Examples:
        >>> url = to_data_url(Image.new("RGB", (10, 10), "red"), fmt="PNG")
        >>> url.startswith("data:image/png;base64,")
        True
    """
    buffer = io.BytesIO()
    pil_img.save(buffer, format=fmt)
    b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
    mime = "image/jpeg" if fmt.upper() == "JPEG" else f"image/{fmt.lower()}"
    return f"data:{mime};base64,{b64}"

def preprocess_from_pil(pil_img: Image.Image) -> np.ndarray:
    """Prépare une image PIL pour une prédiction Keras (redimensionnement + normalisation + batch).
    Convertit en RGB, redimensionne aux dimensions attendues par le modèle, normalise en [0, 1] (float32) et ajoute l'axe batch.

    Args:
        pil_img: Image PIL source.

    Returns:
        np.ndarray de forme (1, MODEL_HEIGHT, MODEL_WIDTH, 3), dtype float32, valeurs ∈ [0, 1].
    """
    logger.info(f"Preprocessing image - taille originale: {pil_img.size}")
    
    # Convertir en RGB
    img = pil_img.convert("RGB")
    
    # CORRECTIF: Redimensionner selon les dimensions attendues par le modèle
    img = img.resize((MODEL_WIDTH, MODEL_HEIGHT), Image.Resampling.LANCZOS)
    logger.info(f"Image redimensionnée à: {img.size}")
    
    # Convertir en array et normaliser
    img_array = np.asarray(img, dtype=np.float32) / 255.0
    
    # Ajouter l'axe batch
    img_array = np.expand_dims(img_array, axis=0)
    
    logger.info(f"Forme finale du tensor: {img_array.shape}")
    return img_array

def save_feedback_to_db(image_data, predicted_label, predicted_confidence, user_label, timestamp):
    """Sauvegarde le feedback utilisateur dansMongoDB Atlas.
    
    Args:
        image_data (str): Image encodée en base64 (ex: "image/jpeg;base64,/9j/4AAQ...")
        predicted_label (str): Classe prédite par le modèle (ex: "forest")
        predicted_confidence (float):Score de confiance du modèle (0.0 à 1.0)
        user_label (str):Classe choisie par l'utilisateur (feedback)
        timestamp (str):Timestamp ISO 8601 de la soumission (ex: "2025-09-16T13:45:22.123456")
    
    Returns:
        None
    
    Logs:
        - INFO si le feedback est sauvegardé avec succès.
        -ERROR si une exception est levée.
        -WARNING si la base de données n'est pas initialisée.
    
    Example:
        >>> save_feedback_to_db("image/...", "meadow", 0.975, "forest", "2025-09-16T13:45:22")
        # Enregistre un document dansMongoDB Atlas.
    """
    if 'feedback_collection' not in globals():
        logger.warning("Base de données non initialisée — feedback non enregistré.")
        return

    try:
        feedback_entry = {
            "timestamp": timestamp,
            "image_data": image_data,
            "predicted_label": predicted_label,
            "confidence": predicted_confidence,
            "user_label": user_label,
            "is_correct": predicted_label == user_label
        }
        
        result = feedback_collection.insert_one(feedback_entry)
        logger.info(f"✅ Feedback sauvegardé dansMongo avec ID : {result.inserted_id}")
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de la sauvegarde dansMongo : {e}", exc_info=True)

def rate_limit(max_per_minute):
    min_interval = 60.0 / max_per_minute
    def decorator(func):
        last_called = [0.0]
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            left_to_wait = min_interval - elapsed
            if left_to_wait > 0:
                time.sleep(left_to_wait)
            ret = func(*args, **kwargs)
            last_called[0] = time.time()
            return ret
        return wrapper
    return decorator

# ---------------- Routes ----------------
@app.route("/", methods=["GET"])
def index():
    """Affiche la page d'upload.

    Returns:
        Réponse HTML rendant le template "upload.html".
    """
    logger.info("Affichage de la page d'upload")
    return render_template("upload.html")

@app.route("/predict", methods=["POST"])
@rate_limit(10)  # Max 10 requêtes par minute
def predict():
    """Traite l'upload, exécute la prédiction et affiche le résultat.

    Attendu: une requête `multipart/form-data` avec le champ `file`.
    Étapes:
      1) Validation de présence et d'extension du fichier.
      2) Lecture du contenu en mémoire et ouverture en PIL.
      3) Prétraitement -> tenseur (1, MODEL_HEIGHT, MODEL_WIDTH, 3).
      4) Prédiction Keras -> probas, top-1 (label, confiance).
      5) Encodage de l'image en Data URL et rendu du template résultat.

    Redirects:
        - Redirige vers "/" si le fichier est manquant ou invalide.

    Returns:
        Réponse HTML rendant "result.html" avec:
        - `image_data_url` : image soumise encodée (base64),
        - `predicted_label` : classe prédite (str),
        - `confidence` : score softmax (float),
        - `classes` : liste des classes (pour les boutons).
    """
    try:
        logger.info("Début de la prédiction")
        
        if "file" not in request.files:
            logger.warning("Aucun fichier dans la requête")
            return redirect("/")
        
        file = request.files["file"]
        if file.filename == "" or not allowed_file(secure_filename(file.filename)):
            logger.warning(f"Fichier invalide: {file.filename}")
            return redirect("/")

        logger.info(f"Traitement du fichier: {file.filename}")

        raw = file.read()
        pil_img = Image.open(io.BytesIO(raw))
        
        # Préprocessing avec redimensionnement
        img_array = preprocess_from_pil(pil_img)

        # Prédiction avec wrapper pour capturer les erreurs spécifiques
        try:
            probs = model.predict(img_array, verbose=0)[0]
        except Exception as pred_error:
            logger.error(f"❌ ERREUR DE PRÉDICTION: {type(pred_error).__name__}: {pred_error}", exc_info=True)
            raise  # Re-lance l'erreur pour que le try/catch externe la gère

        cls_idx = int(np.argmax(probs))
        label = CLASSES[cls_idx]
        conf = float(probs[cls_idx])

        logger.info(f"Prédiction: {label} (confiance: {conf:.3f})")

        image_data_url = to_data_url(pil_img, fmt="JPEG")

        return render_template("result.html", 
                             image_data_url=image_data_url, 
                             predicted_label=label, 
                             confidence=conf, 
                             classes=CLASSES)
        
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {e}", exc_info=True)
        return jsonify({"error": "Erreur lors du traitement de l'image"}), 500

@app.route("/feedback", methods=["POST"])
def submit_feedback():
    """Enregistre le feedback utilisateur."""
    try:
        data = request.get_json()
        
        image_data = data.get('image_data')
        predicted_label = data.get('predicted_label')
        predicted_confidence = data.get('predicted_confidence')
        user_label = data.get('user_label')
        timestamp = datetime.now().isoformat()
        
        # Sauvegarde du feedback
        save_feedback_to_db(image_data, predicted_label, predicted_confidence, user_label, timestamp)
        
        logger.info(f"Feedback reçu: prédiction={predicted_label}, utilisateur={user_label}")
        
        return jsonify({"status": "success", "message": "Feedback enregistré"}), 200
        
    except Exception as e:
        logger.error(f"Erreur lors de l'enregistrement du feedback: {e}", exc_info=True)
        return jsonify({"error": "Erreur lors de l'enregistrement du feedback"}), 500

@app.route("/feedback", methods=["GET"])
def feedback_ok():
    """Affiche la page de confirmation de feedback (placeholder).

    Returns:
        Réponse HTML rendant le template "feedback_ok.html".
    """
    return render_template("feedback_ok.html")

# Gestion des erreurs
@app.errorhandler(404)
def not_found_error(error):
    logger.warning(f"404 - Page non trouvée: {request.url}")
    return jsonify({"error": "Page non trouvée"}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"500 - Erreur interne: {error}")
    return jsonify({"error": "Erreur interne du serveur"}), 500

if __name__ == "__main__":
    # Créer le dossier de logs s'il n'existe pas (chemin absolu)
    logs_dir = os.path.join(BASE_DIR, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    logger.info("Démarrage de l'application Flask")
    
    # ❌ Désactive le mode debug en production
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))