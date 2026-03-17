# Outils d'Analyse du Monitoring

Outils d'analyse et de diagnostic pour le système de monitoring Flask-MonitoringDashboard.

## debug_monitoring.py

Script d'analyse de la base de données SQLite du monitoring Flask-MonitoringDashboard.

**Usage :**
```bash
python monitoring/debug_monitoring.py
```

**Fonctionnalités :**
- Liste toutes les tables et leur contenu
- Analyse des requêtes par endpoint
- Activité récente (10 dernières requêtes)
- Statistiques de performance
- Identification des requêtes les plus lentes

**Exemple de sortie :**
```
📋 Tables disponibles (18):
  - fmd_Request: 95 entrées
  - fmd_Endpoint: 6 entrées

📊 ANALYSE DES REQUÊTES
Endpoint             Requêtes   Moy (ms)     Max (ms)
predict              41         4312748.6    6906270.7
submit_feedback      41         47939.8      426253.6
```

## analyze_performance.py

Script d'analyse approfondie des performances avec diagnostic automatique.

**Usage :**
```bash
python scripts/analyze_performance.py
```

**Fonctionnalités :**
- Diagnostic automatique des endpoints lents
- Détection du rate limiting
- Suggestions d'optimisation ciblées
- Actions prioritaires recommandées

**Diagnostic automatique :**
- 🚨 TRÈS LENT : >5000ms
- ⚠️ LENT : >2000ms
- 🟡 MOYEN : >500ms
- ✅ RAPIDE : <500ms

**Exemple de sortie :**
```
🔍 DIAGNOSTIC DE PERFORMANCE
predict              41         4312749ms    🚨 TRÈS LENT
submit_feedback      41         47940ms      🚨 TRÈS LENT

🚦 ANALYSE DU RATE LIMITING
Intervalle moyen entre prédictions: 35.0s
⚠️ Rate limiting détecté
```

## Prérequis

- Application Flask lancée avec monitoring activé
- Base de données SQLite `app/monitoring_dashboard.db` existante
- Python 3.8+

## Utilisation recommandée

1. **Après tests de charge** (notebook monitoring_test.ipynb)
2. **Diagnostic de problèmes** de performance
3. **Optimisation** avant déploiement en production
4. **Monitoring continu** en développement

## Scripts complémentaires

Ces scripts complètent l'écosystème de monitoring :
- `notebooks/monitoring_test.ipynb` : Génération de trafic de test
- `app/app_correct.py` : Application avec monitoring intégré
- Dashboard : `http://localhost:5000/dashboard` pour interface graphique