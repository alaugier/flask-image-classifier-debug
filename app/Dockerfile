# Dockerfile pour l'application Flask de classification d'images
FROM python:3.11-slim

# Variables d'environnement
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=app_correct.py
ENV FLASK_ENV=production

# Installation des dépendances système
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Créer un utilisateur non-root
RUN useradd --create-home --shell /bin/bash app

# Définir le répertoire de travail
WORKDIR /home/app

# Copier les fichiers de requirements
COPY app/requirements.txt requirements.txt

# Installer les dépendances Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copier l'application
COPY app/ ./app/
COPY tests/test_app.py .

# Créer les répertoires nécessaires
RUN mkdir -p logs static/uploads

# Changer la propriété des fichiers
RUN chown -R app:app /home/app

# Passer à l'utilisateur non-root
USER app

# Exposer le port
EXPOSE 5000

# Vérification de santé
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/ || exit 1

# Commande par défaut
CMD ["python", "app/app_correct.py"]