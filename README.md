# Projet de Débogage d'Application Flask - Classification d'Images Satellite

Application Flask de classification d'images satellite avec système de monitoring, logging et feedback utilisateur.

## 🎯 Objectifs du projet

- Correction d'un bug de dimension d'image dans le preprocessing
- Implémentation de tests automatisés
- Mise en place d'un système de logging avancé
- Configuration d'un dashboard de monitoring
- Système d'alerting par email
- Feedback loop avec persistance MongoDB
- Documentation complète du processus de débogage

## 🏗️ Architecture

### Classes de classification
- `desert` : Paysages désertiques
- `forest` : Zones forestières  
- `meadow` : Prairies et champs
- `mountain` : Reliefs montagneux

### Stack technique
- **Backend** : Flask + Keras/TensorFlow
- **Monitoring** : Flask-MonitoringDashboard
- **Base de données** : MongoDB Atlas (feedback) + SQLite (monitoring)
- **Logging** : Python logging + SMTPHandler
- **Tests** : unittest + pytest
- **Déploiement** : Render

## 🚀 Installation

### Prérequis
```bash
Python 3.8+
pip
Git LFS (pour le modèle)
```

### Configuration locale
```bash
# Cloner le repo
git clone https://github.com/votre-username/projet-debug-flask.git
cd projet-debug-flask

# Environnement virtuel
python -m venv env_debug
source env_debug/bin/activate  # Linux/Mac
# ou env_debug\Scripts\activate  # Windows

# Dépendances
pip install -r requirements.txt

# Variables d'environnement
cp .env.example .env
# Éditer .env avec vos configurations

# Configuration monitoring
cp app/config.cfg.example app/config.cfg
# Éditer config.cfg si nécessaire
```

### Récupérer le modèle
Le modèle Keras (`final_cnn.keras`) est trop volumineux pour Git standard.

Option 1 - Git LFS :
```bash
git lfs pull
```

Option 2 - Téléchargement manuel :
```bash
# Voir app/models/README.md pour les instructions
```

## 🏃‍♂️ Utilisation

### Lancer l'application
```bash
cd app
KERAS_BACKEND=torch python app_correct.py
```

L'application sera accessible sur `http://localhost:5000`

### Dashboard de monitoring
Accédez au dashboard sur `http://localhost:5000/dashboard`
- Identifiants par défaut : `admin` / `admin`

### Tests
```bash
# Tests unitaires
python -m unittest tests.test_app -v

# Test du monitoring (notebook)
jupyter notebook notebooks/monitoring_test.ipynb
```

## 🐛 Bug identifié et corrigé

### Problème initial
```
ValueError: Input 0 of layer "functional_1" is incompatible 
with the layer: expected shape=(None, 224, 224, 3), 
found shape=(1, 600, 600, 3)
```

### Cause
La fonction `preprocess_from_pil()` ne redimensionnait pas les images selon les attentes du modèle (224x224).

### Solution
Ajout du redimensionnement automatique basé sur `model.input_shape` :
```python
img = img.resize((MODEL_WIDTH, MODEL_HEIGHT), Image.Resampling.LANCZOS)
```

## 📊 Monitoring

### Métriques techniques (Dashboard Flask)
- Performance des endpoints
- Temps de réponse
- Charge système
- Détection d'outliers

### Métriques métier (MongoDB)
- Accuracy des prédictions
- Feedback utilisateur
- Volume d'images traitées
- Données pour réentraînement

## 🔧 Configuration

### Variables d'environnement (.env)
```env
MONGO_URI=mongodb+srv://...
FLASK_ENV=development
DASHBOARD_USERNAME=admin
DASHBOARD_PASSWORD=secure_password
FLASK_MONITORING_TOKEN=your_token
ALERT_EMAIL=admin@example.com
ALERT_EMAIL_PASSWORD=app_password
```

### Monitoring (config.cfg)
- Configuration SQLite pour les métriques techniques
- Paramètres de monitoring et alerting
- Seuils de performance

## 📁 Structure du projet

```
app/                    # Application Flask
notebooks/              # Tests de monitoring
tests/                  # Tests automatisés
images_to_test/         # Images d'exemple
docs/                   # Documentation
```

## 🧪 Tests automatisés

12 tests couvrant :
- Validation des formats de fichiers
- Preprocessing et redimensionnement
- Endpoints Flask
- Intégration modèle
- Système de feedback

## 📈 Déploiement

Voir `docs/DEPLOYMENT.md` pour :
- Configuration Render
- Variables d'environnement production
- Monitoring en production
- CI/CD avec GitHub Actions

## 🤝 Contribution

1. Fork du projet
2. Créer une branche feature
3. Tests locaux
4. Pull request avec description détaillée

## 📝 Documentation

- `docs/MONITORING.md` : Guide du système de monitoring
- `docs/API.md` : Documentation des endpoints
- `notebooks/README.md` : Utilisation des notebooks de test

## ⚠️ Notes importantes

- Les credentials ne sont jamais commitées
- Le modèle Keras nécessite Git LFS
- MongoDB Atlas requis pour le feedback
- Dashboard accessible uniquement avec authentification

---

**Auteur** : Alexandre Laugier  
**Contexte** : Projet de débogage d'application - Formation DevOps
