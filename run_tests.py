#!/usr/bin/env python3
"""
Script pour exécuter les tests depuis la racine du projet
Usage: python run_tests.py
Note: Ce script suppose que l'application s'appelle app_correct.py
"""

import os
import sys
import subprocess

def main():
    # Se placer dans le répertoire du script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Ajouter le répertoire app au PYTHONPATH
    app_dir = os.path.join(script_dir, 'app')
    env = os.environ.copy()
    env['PYTHONPATH'] = app_dir + ':' + env.get('PYTHONPATH', '')
    
    # Lancer les tests
    try:
        print("🧪 Exécution des tests...")
        result = subprocess.run([
            sys.executable, 'tests/test_app.py'
        ], env=env, cwd=script_dir)
        
        if result.returncode == 0:
            print("✅ Tous les tests sont passés avec succès!")
        else:
            print("❌ Certains tests ont échoué.")
            sys.exit(1)
            
    except FileNotFoundError:
        print("❌ Impossible de trouver le fichier de tests.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Erreur lors de l'exécution des tests: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()