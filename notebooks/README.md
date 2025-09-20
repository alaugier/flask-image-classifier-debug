# Notebooks de Test

Ce dossier contient les notebooks Jupyter pour tester et valider l'application.

## monitoring_test.ipynb

Notebook de test du système de monitoring qui génère du trafic artificiel pour alimenter le dashboard Flask-MonitoringDashboard.

### Fonctionnalités

- **Détection automatique** de l'environnement (local/production)
- **Tests d'endpoints** de base
- **Upload d'images** réelles ou générées
- **Simulation de feedback** utilisateur
- **Tests de charge** configurable

### Configuration

Modifiez la variable `TEST_TYPE` au début du notebook :

```python
TEST_TYPE = "normal"    # 10 requêtes
TEST_TYPE = "extended"  # 30 requêtes  
TEST_TYPE = "stress"    # Test de charge
TEST_TYPE = "custom"    # Personnalisé
```

### Types de tests disponibles

| Test | Durée | Requêtes | Objectif |
|------|-------|----------|----------|
| `normal` | ~30s | 10 | Test standard |
| `extended` | ~2min | 30 | Validation complète |
| `stress` | 5min | Variable | Test de charge |
| `custom` | Variable | Configurable | Tests spécifiques |

### Utilisation

1. **Lancer l'application Flask** :
   ```bash
   cd ../app
   python app_correct.py
   ```

2. **Ouvrir le notebook** :
   ```bash
   jupyter notebook monitoring_test.ipynb
   ```

3. **Configurer le test** en modifiant `TEST_TYPE`

4. **Exécuter toutes les cellules**

5. **Vérifier le dashboard** sur `http://localhost:5000/dashboard`

### URLs détectées automatiquement

- **Local** : `http://localhost:5000`
- **Production Render** : Via variable d'environnement `RENDER_APP_URL`
- **Personnalisée** : Saisie manuelle si détection échoue

### Données générées

Le notebook génère :
- **Trafic HTTP** pour le dashboard technique
- **Images de test** si le dossier `images_to_test/` n'existe pas
- **Feedback simulé** avec variations réalistes
- **Métriques de performance** avec délais variables

### Images de test

Le notebook utilise en priorité les vraies images du dossier `../images_to_test/` :
- `desert_96.jpg`
- `meadow_89.jpg` 
- `mountain_87.jpg`

Si absent, il génère des images factices avec PIL.

### Monitoring des résultats

Après exécution, vérifiez :

1. **Dashboard technique** (`/dashboard`) :
   - Statistiques de trafic
   - Performance des endpoints
   - Détection d'anomalies

2. **Base MongoDB** (si configurée) :
   - Feedback utilisateur stocké
   - Métadonnées des images
   - Historique des prédictions

### Dépendances

```python
requests>=2.31.0
PIL>=10.0.0
jupyter>=1.0.0
beautifulsoup4>=4.12.2  # Optionnel pour parsing HTML
```

### Notes de production

En production sur Render :
- Configurez `RENDER_APP_URL` dans les variables d'environnement
- Le notebook détectera automatiquement l'URL
- Utilisez `TEST_TYPE="normal"` pour éviter de surcharger
- Surveillez les quotas de votre plan Render