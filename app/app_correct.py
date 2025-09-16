import os
os.environ["KERAS_BACKEND"] = "torch"

import io
import base64
import logging
from datetime import datetime

from flask import Flask, render_template, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename

import numpy as np
import keras

from PIL import Image

# ---------------- Config ----------------
UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXT = {"png", "jpg", "jpeg", "webp"}
CLASSES = ['desert', 'forest', 'meadow', 'mountain']

app = Flask(__name__)

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ---------------- Model ----------------
# Chemin absolu vers le modèle (relatif au fichier app.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "final_cnn.keras")

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
    """Sauvegarde le feedback utilisateur (simulation - à remplacer par une vraie DB).
    
    Args:
        image_data: Image en base64
        predicted_label: Classe prédite par le modèle
        predicted_confidence: Confiance de la prédiction
        user_label: Classe choisie par l'utilisateur
        timestamp: Timestamp de la soumission
    """
    # Pour l'instant, on sauvegarde dans un fichier (à remplacer par une base de données)
    feedback_entry = {
        'timestamp': timestamp,
        'predicted_label': predicted_label,
        'predicted_confidence': predicted_confidence,
        'user_label': user_label,
        'image_data': image_data[:100] + "..." if len(image_data) > 100 else image_data  # Tronqué pour les logs
    }
    
    logger.info(f"Feedback enregistré: {feedback_entry}")
    
    # TODO: Implémenter la sauvegarde en base de données
    # Exemple avec SQLAlchemy:
    # feedback = Feedback(
    #     image_data=image_data,
    #     predicted_label=predicted_label,
    #     predicted_confidence=predicted_confidence,
    #     user_label=user_label,
    #     timestamp=timestamp
    # )
    # db.session.add(feedback)
    # db.session.commit()

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

        # Prédiction
        probs = model.predict(img_array, verbose=0)[0]
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
    app.run(debug=True)