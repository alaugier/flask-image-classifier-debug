import os
import logging
from logging.handlers import SMTPHandler

# ✅ Charger les variables d'environnement depuis .env en local
if os.path.exists(".env"):
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Variables d'environnement chargées depuis .env")

# ✅ Créer le dossier de logs AVANT de configurer le logging
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
logs_dir = os.path.join(BASE_DIR, 'logs')
os.makedirs(logs_dir, exist_ok=True)

# ✅ Configuration du logger racine (capture TOUTES les erreurs, même au démarrage)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(logs_dir, 'app.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()  # ← Logger racine — capture tout, même avant Flask

# ✅ Alerting par email (uniquement en production)
if os.getenv('FLASK_ENV') == 'production':
    # Vérifier que toutes les variables nécessaires sont présentes
    required_vars = ['ALERT_EMAIL', 'ALERT_EMAIL_PASSWORD']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"❌ Variables d'environnement manquantes pour l'alerting: {missing_vars}")
    else:
        try:
            mail_handler = SMTPHandler(
                mailhost=('smtp.gmail.com', 587),
                fromaddr=os.getenv('ALERT_EMAIL'),
                toaddrs=[os.getenv('ALERT_EMAIL_RECIPIENT', os.getenv('ALERT_EMAIL'))],
                subject='🚨 ERREUR CRITIQUE - App Classification',
                credentials=(os.getenv('ALERT_EMAIL'), os.getenv('ALERT_EMAIL_PASSWORD')),
                secure=()
            )
            mail_handler.setLevel(logging.ERROR)
            mail_handler.setFormatter(logging.Formatter(
                '[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
            ))
            logger.addHandler(mail_handler)
            logger.info("✅ Alerting par email activé pour les erreurs critiques.")
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la configuration de l'alerting: {e}", exc_info=True)
else:
    logger.info("ℹ️ Mode développement - Alerting email désactivé")

import io
import base64

from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename

import numpy as np
import keras

from PIL import Image

# ---------------- Config ----------------
UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXT = {"png", "jpg", "jpeg", "webp"}
CLASSES = ['desert', 'forest', 'meadow', 'mountain']

app = Flask(__name__)

# ---------------- Model ----------------
# MODEL_PATH = "models/final_cnn.keras"
MODEL_PATH = os.path.join(BASE_DIR, "models", "final_cnn.keras")
model = keras.saving.load_model(MODEL_PATH, compile=False)

# ---------------- Utils ----------------
def allowed_file(filename: str) -> bool:
    """Vérifie si le nom de fichier possède une extension autorisée.
    La vérification est **insensible à la casse** et ne regarde que la sous-chaîne
    après le dernier point. Dépend de la constante globale `ALLOWED_EXT`.

    Args:
        filename: Nom du fichier soumis (ex. "photo.PNG").

    Returns:
        True si l’extension (ex. "png", "jpg") est dans `ALLOWED_EXT`, sinon False.

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
    L’image est encodée en mémoire (sans I/O disque), sérialisée en base64, puis
    encapsulée comme `data:<mime>;base64,<payload>`. Le type MIME est déduit de `fmt`.

    Args:
        pil_img: Image PIL à encoder.
        fmt: Format d’encodage PIL (ex. "JPEG", "PNG"). Par défaut "JPEG".

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
    """Prépare une image PIL pour une prédiction Keras (normalisation + batch).
    Convertit en RGB, normalise en [0, 1] (float32) et ajoute l’axe batch.

    Args:
        pil_img: Image PIL source.

    Returns:
        np.ndarray de forme (1, H, W, 3), dtype float32, valeurs ∈ [0, 1].
    """
    img = pil_img.convert("RGB")
    img_array = np.asarray(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ---------------- Routes ----------------
@app.route("/", methods=["GET"])
def index():
    """Affiche la page d’upload.

    Returns:
        Réponse HTML rendant le template "upload.html".
    """
    return render_template("upload.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Traite l'upload, exécute la prédiction et affiche le résultat.

    Attendu: une requête `multipart/form-data` avec le champ `file`.
    Étapes:
    1) Validation de présence et d'extension du fichier.
    2) Lecture du contenu en mémoire et ouverture en PIL.
    3) Prétraitement -> tenseur (1, H, W, 3).
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

    if "file" not in request.files:
        return redirect("/")
    
    file = request.files["file"]
    if file.filename == "" or not allowed_file(secure_filename(file.filename)):
        return redirect("/")

    try:
        raw = file.read()
        pil_img = Image.open(io.BytesIO(raw))
        img_array = preprocess_from_pil(pil_img)

        # ✅ Wrapper pour capturer les erreurs de prédiction
        try:
            probs = model.predict(img_array, verbose=0)[0]
        except Exception as pred_error:
            logger.error(f"❌ ERREUR DE PRÉDICTION: {type(pred_error).__name__}: {pred_error}", exc_info=True)
            raise  # Re-lance l'erreur pour que le try/catch externe la gère
        cls_idx = int(np.argmax(probs))
        label = CLASSES[cls_idx]
        conf = float(probs[cls_idx])

        image_data_url = to_data_url(pil_img, fmt="JPEG")

        return render_template("result.html", image_data_url=image_data_url, predicted_label=label, confidence=conf, classes=CLASSES)

    except Exception as e:
        # ✅ Capture TOUTES les erreurs (y compris ValueError) et les logue
        logger.error(f"❌ ERREUR CRITIQUE - Échec de prédiction pour {file.filename}: {type(e).__name__}: {e}", exc_info=True)
        # Retourner une page d'erreur ou rediriger
        return redirect("/")  # Ou return "Erreur de prédiction", 500

@app.route("/feedback", methods=["GET"])
def feedback_ok():
    """Affiche la page de confirmation de feedback (placeholder).

    Returns:
        Réponse HTML rendant le template "feedback_ok.html".
    """
    return render_template("feedback_ok.html")

if __name__ == "__main__":
    app.run(debug=True)
