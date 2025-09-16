import unittest
import tempfile
import os
import io
import json
from unittest.mock import patch, MagicMock
from PIL import Image
import numpy as np

# Import de l'application
import sys
import os
# Ajouter le répertoire parent (racine du projet) au path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Ajouter le dossier app au path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'app'))

# Mock du modèle avant l'import de l'app
with patch('keras.saving.load_model') as mock_load_model:
    # Créer un mock du modèle
    mock_model = MagicMock()
    mock_model.input_shape = (None, 224, 224, 3)
    mock_model.predict.return_value = np.array([[0.1, 0.7, 0.15, 0.05]])  # Probabilités factices
    mock_load_model.return_value = mock_model
    
    from app_correct import app, preprocess_from_pil, allowed_file, to_data_url

class TestImageClassificationApp(unittest.TestCase):
    
    def setUp(self):
        """Configuration avant chaque test."""
        self.app = app.test_client()
        self.app.testing = True
        
    def create_test_image(self, size=(600, 600), format='JPEG'):
        """Crée une image de test."""
        img = Image.new('RGB', size, color='red')
        img_buffer = io.BytesIO()
        img.save(img_buffer, format=format)
        img_buffer.seek(0)
        return img_buffer
    
    def test_allowed_file_valid_extensions(self):
        """Test de validation des extensions de fichier valides."""
        self.assertTrue(allowed_file("test.jpg"))
        self.assertTrue(allowed_file("test.jpeg"))
        self.assertTrue(allowed_file("test.png"))
        self.assertTrue(allowed_file("test.webp"))
        self.assertTrue(allowed_file("test.JPG"))  # Insensible à la casse
    
    def test_allowed_file_invalid_extensions(self):
        """Test de validation des extensions de fichier invalides."""
        self.assertFalse(allowed_file("test.txt"))
        self.assertFalse(allowed_file("test.pdf"))
        self.assertFalse(allowed_file("test"))
        self.assertFalse(allowed_file(""))
    
    def test_preprocess_from_pil_shape_correction(self):
        """Test critique: vérification que le preprocessing redimensionne correctement."""
        # Créer une image de test avec une taille différente de 224x224
        test_img = Image.new('RGB', (600, 600), color='blue')
        
        # Appliquer le preprocessing
        processed = preprocess_from_pil(test_img)
        
        # Vérifier que l'image est redimensionnée aux bonnes dimensions
        expected_shape = (1, 224, 224, 3)  # (batch, height, width, channels)
        self.assertEqual(processed.shape, expected_shape, 
                        f"L'image doit être redimensionnée à {expected_shape}, obtenu: {processed.shape}")
        
        # Vérifier que les valeurs sont normalisées entre 0 et 1
        self.assertTrue(np.all(processed >= 0) and np.all(processed <= 1),
                       "Les valeurs doivent être normalisées entre 0 et 1")
        
        # Vérifier le type de données
        self.assertEqual(processed.dtype, np.float32, 
                        "Le type de données doit être float32")
    
    def test_preprocess_different_input_sizes(self):
        """Test que le preprocessing fonctionne avec différentes tailles d'entrée."""
        test_sizes = [(100, 100), (300, 200), (800, 600), (1024, 768)]
        
        for width, height in test_sizes:
            with self.subTest(size=(width, height)):
                test_img = Image.new('RGB', (width, height), color='green')
                processed = preprocess_from_pil(test_img)
                
                # Toutes les images doivent être redimensionnées à 224x224
                expected_shape = (1, 224, 224, 3)
                self.assertEqual(processed.shape, expected_shape,
                               f"Taille {width}x{height} mal redimensionnée")
    
    def test_to_data_url(self):
        """Test de conversion d'image en data URL."""
        test_img = Image.new('RGB', (10, 10), color='red')
        data_url = to_data_url(test_img, fmt='PNG')
        
        self.assertTrue(data_url.startswith('data:image/png;base64,'),
                       "La data URL doit commencer par le bon préfixe")
    
    def test_index_route(self):
        """Test de la route principale."""
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn('Classification images satellite'.encode('utf-8'), response.data)
    
    def test_predict_without_file(self):
        """Test de prédiction sans fichier."""
        response = self.app.post('/predict', data={})
        # Doit rediriger vers la page principale
        self.assertEqual(response.status_code, 302)
    
    def test_predict_with_invalid_file(self):
        """Test de prédiction avec un fichier invalide."""
        response = self.app.post('/predict', data={
            'file': (io.BytesIO(b'not an image'), 'test.txt')
        })
        # Doit rediriger vers la page principale
        self.assertEqual(response.status_code, 302)
    
    def test_predict_with_valid_image(self):
        """Test de prédiction avec une image valide."""
        # Créer une image de test
        test_image = self.create_test_image(size=(600, 600))
        
        response = self.app.post('/predict', data={
            'file': (test_image, 'test.jpg')
        }, content_type='multipart/form-data')
        
        # La prédiction doit réussir
        self.assertEqual(response.status_code, 200)
        self.assertIn('Classe prédite'.encode('utf-8'), response.data)
        
        # Vérifier la présence des classes dans la réponse
        for class_name in ['desert', 'forest', 'meadow', 'mountain']:
            self.assertIn(class_name.encode('utf-8'), response.data)
    
    def test_feedback_post_route(self):
        """Test de soumission de feedback."""
        feedback_data = {
            'image_data': 'data:image/jpeg;base64,fake_data',
            'predicted_label': 'desert',
            'predicted_confidence': 0.85,
            'user_label': 'mountain'
        }
        
        response = self.app.post('/feedback',
                               data=json.dumps(feedback_data),
                               content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        response_data = json.loads(response.data)
        self.assertEqual(response_data['status'], 'success')
    
    def test_regression_large_image_processing(self):
        """Test de régression: vérification que les grandes images sont traitées correctement.
        
        Ce test reproduit le bug original où une image 600x600 causait une erreur
        car elle n'était pas redimensionnée.
        """
        # Image de test avec la taille problématique du bug original
        test_image = self.create_test_image(size=(600, 600))
        
        # Cette requête devait échouer avant le correctif
        response = self.app.post('/predict', data={
            'file': (test_image, 'large_image.jpg')
        }, content_type='multipart/form-data')
        
        # Maintenant elle doit réussir
        self.assertEqual(response.status_code, 200, 
                        "Les images 600x600 doivent être traitées sans erreur")
        self.assertNotIn('ValueError'.encode('utf-8'), response.data)
        self.assertNotIn('incompatible with the layer'.encode('utf-8'), response.data)

class TestModelIntegration(unittest.TestCase):
    """Tests d'intégration avec le modèle."""
    
    @patch('app_correct.model')
    def test_model_input_requirements(self, mock_model):
        """Test que le modèle accepte bien les entrées preprocessées."""
        # Configurer le mock
        mock_model.predict.return_value = np.array([[0.2, 0.5, 0.2, 0.1]])
        
        # Créer une image de test
        test_img = Image.new('RGB', (800, 600), color='blue')
        
        # Appliquer le preprocessing
        processed = preprocess_from_pil(test_img)
        
        # Vérifier que le preprocessing donne la bonne forme
        self.assertEqual(processed.shape, (1, 224, 224, 3))
        
        # Simuler une prédiction
        predictions = mock_model.predict(processed, verbose=0)
        self.assertEqual(len(predictions[0]), 4, 
                       "Le modèle doit retourner 4 probabilités (une par classe)")
        self.assertAlmostEqual(np.sum(predictions[0]), 1.0, places=1,
                             msg="Les probabilités doivent approximativement sommer à 1")

if __name__ == '__main__':
    # Créer le dossier logs si nécessaire
    os.makedirs('logs', exist_ok=True)
    
    # Exécuter les tests
    unittest.main(verbosity=2)