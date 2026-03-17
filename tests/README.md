# Tests Unitaires

Ce dossier contient les tests automatisés pour l'application de classification d'images satellite.

## Structure des tests

### TestImageClassificationApp
Tests principaux du flux de traitement des images et des routes Flask.

**Tests de validation des fichiers :**
- `test_allowed_file_valid_extensions()` - Extensions autorisées (jpg, png, webp, jpeg)
- `test_allowed_file_invalid_extensions()` - Extensions interdites

**Tests de preprocessing (critiques) :**
- `test_preprocess_from_pil_shape_correction()` - Redimensionnement vers 224x224
- `test_preprocess_different_input_sizes()` - Différentes tailles d'entrée
- `test_regression_large_image_processing()` - Test de régression du bug 600x600

**Tests des routes Flask :**
- `test_index_route()` - Page d'accueil
- `test_predict_without_file()` - Upload sans fichier
- `test_predict_with_invalid_file()` - Upload avec fichier invalide  
- `test_predict_with_valid_image()` - Upload avec image valide
- `test_feedback_post_route()` - Soumission de feedback

**Tests utilitaires :**
- `test_to_data_url()` - Conversion en data URL base64

### TestModelIntegration
Tests d'intégration avec le modèle Keras mocké.

- `test_model_input_requirements()` - Compatibilité des entrées preprocessées

## Exécution des tests

### Méthode standard (unittest)
```bash
# Depuis la racine du projet
python -m unittest tests.test_app -v

# Ou directement
cd tests
python test_app.py
```

### Méthode alternative (pytest)
Si vous installez pytest, vous pouvez l'utiliser pour exécuter les tests unittest :
```bash
pip install pytest
python -m pytest tests/ -v
```

Note : Les tests sont écrits avec `unittest`, mais pytest peut les exécuter car il est compatible.

## Configuration des tests

### Mocking du modèle
Le modèle Keras est mocké pour éviter de charger le vrai fichier `final_cnn.keras` lors des tests :

```python
with patch('keras.saving.load_model') as mock_load_model:
    mock_model = MagicMock()
    mock_model.input_shape = (None, 224, 224, 3)
    mock_model.predict.return_value = np.array([[0.1, 0.7, 0.15, 0.05]])
```

### Images de test
Les tests génèrent des images factices avec PIL :
```python
def create_test_image(self, size=(600, 600), format='JPEG'):
    img = Image.new('RGB', size, color='red')
```

## Tests critiques

### Test de régression du bug principal
`test_regression_large_image_processing()` reproduit le bug original :
- Upload d'une image 600x600
- Vérification qu'elle est traitée sans erreur `ValueError`
- Validation que le redimensionnement fonctionne

### Test de preprocessing
`test_preprocess_from_pil_shape_correction()` vérifie :
- Redimensionnement automatique vers 224x224
- Normalisation des valeurs [0,1]
- Type de données float32
- Ajout correct de l'axe batch

## Couverture des tests

Les 12 tests couvrent :
- ✅ Validation des formats de fichiers
- ✅ Preprocessing et redimensionnement 
- ✅ Routes Flask principales
- ✅ Gestion des erreurs
- ✅ Système de feedback
- ✅ Intégration modèle
- ✅ Test de régression du bug

## Dépendances

```python
unittest          # Standard Python
unittest.mock     # Mocking 
PIL (Pillow)      # Images de test
numpy             # Arrays pour le modèle
```

## Ajout de nouveaux tests

Pour ajouter un test :

1. **Test unitaire** → Ajoutez dans `TestImageClassificationApp`
2. **Test d'intégration** → Ajoutez dans `TestModelIntegration`  
3. **Nommage** → Préfixe `test_` obligatoire
4. **Documentation** → Docstring expliquant l'objectif

Exemple :
```python
def test_new_feature(self):
    """Test de la nouvelle fonctionnalité X."""
    # Arrange
    setup_data = ...
    
    # Act  
    result = function_to_test(setup_data)
    
    # Assert
    self.assertEqual(result, expected_value)
```

## CI/CD

Ces tests sont exécutés automatiquement lors :
- Des pull requests
- Des déploiements
- Du pipeline CI/CD GitHub Actions

## Métriques de réussite

- **Couverture** : >80% du code applicatif
- **Temps d'exécution** : <30 secondes
- **Fiabilité** : 100% de réussite requise pour le déploiement