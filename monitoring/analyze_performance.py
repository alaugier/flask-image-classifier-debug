#!/usr/bin/env python3
"""
Analyseur de performances pour identifier les goulots d'étranglement.
Usage: python scripts/analyze_performance.py
"""

import sqlite3
import os
from datetime import datetime
import statistics

def analyze_performance_issues(conn):
    """Analyse détaillée des problèmes de performance."""
    cursor = conn.cursor()
    
    print("🔍 DIAGNOSTIC DE PERFORMANCE")
    print("=" * 50)
    
    # Analyse des endpoint les plus lents
    query = """
    SELECT e.name as endpoint,
           COUNT(r.id) as requests,
           AVG(r.duration) as avg_duration,
           MAX(r.duration) as max_duration,
           MIN(r.duration) as min_duration
    FROM fmd_Request r
    JOIN fmd_Endpoint e ON r.endpoint_id = e.id
    GROUP BY e.name
    HAVING COUNT(r.id) > 1
    ORDER BY avg_duration DESC;
    """
    
    cursor.execute(query)
    results = cursor.fetchall()
    
    print("📊 Performances par endpoint:")
    print(f"{'Endpoint':<20} {'Requêtes':<10} {'Moyenne':<12} {'Max':<12} {'Diagnostic'}")
    print("-" * 80)
    
    for row in results:
        avg_ms = row['avg_duration'] * 1000
        max_ms = row['max_duration'] * 1000
        
        # Diagnostic automatique
        if avg_ms > 5000:
            diagnostic = "🚨 TRÈS LENT"
        elif avg_ms > 2000:
            diagnostic = "⚠️ LENT"  
        elif avg_ms > 500:
            diagnostic = "🟡 MOYEN"
        else:
            diagnostic = "✅ RAPIDE"
            
        print(f"{row['endpoint']:<20} {row['requests']:<10} "
              f"{avg_ms:<12.0f}ms {max_ms:<12.0f}ms {diagnostic}")

def suggest_optimizations():
    """Suggestions d'optimisation basées sur l'analyse."""
    print("\n💡 SUGGESTIONS D'OPTIMISATION")
    print("=" * 40)
    
    suggestions = [
        "1. PRÉDICTION LENTE (4-7 secondes):",
        "   - Vérifier si le modèle utilise le GPU",
        "   - Optimiser le preprocessing (vectorisation)",
        "   - Considérer un modèle plus léger",
        "   - Implémenter un cache pour images similaires",
        "",
        "2. FEEDBACK LENT (20-47 secondes):",
        "   - Vérifier la connexion MongoDB Atlas",
        "   - Optimiser les index MongoDB", 
        "   - Réduire la taille des données stockées",
        "   - Passer en asynchrone pour l'écriture DB",
        "",
        "3. RATE LIMITING:",
        "   - Le décorateur @rate_limit(10) cause des délais artificiels",
        "   - Considérer l'augmenter ou le supprimer en dev",
        "",
        "4. MONITORING:",
        "   - Flask-MonitoringDashboard ajoute une overhead",
        "   - Réduire MONITOR_LEVEL en production",
        "",
        "5. CODE OPTIMIZATIONS:",
        "   - Lazy loading du modèle",
        "   - Connection pooling pour MongoDB",
        "   - Compression des images avant stockage"
    ]
    
    for suggestion in suggestions:
        print(suggestion)

def analyze_rate_limiting_impact(conn):
    """Analyse l'impact du rate limiting."""
    cursor = conn.cursor()
    
    print("\n🚦 ANALYSE DU RATE LIMITING")
    print("=" * 35)
    
    # Chercher les patterns de délais fixes (indicateur de rate limiting)
    query = """
    SELECT e.name,
           r.duration,
           r.time_requested,
           LAG(r.time_requested) OVER (PARTITION BY e.name ORDER BY r.time_requested) as prev_request
    FROM fmd_Request r
    JOIN fmd_Endpoint e ON r.endpoint_id = e.id
    WHERE e.name = 'predict'
    ORDER BY r.time_requested DESC
    LIMIT 10;
    """
    
    cursor.execute(query)
    results = cursor.fetchall()
    
    intervals = []
    for i, row in enumerate(results[1:], 1):  # Skip first row (no previous)
        if row['prev_request']:
            curr_time = datetime.fromisoformat(row['time_requested'].replace('Z', '+00:00'))
            prev_time = datetime.fromisoformat(row['prev_request'].replace('Z', '+00:00'))
            interval = (curr_time - prev_time).total_seconds()
            intervals.append(interval)
    
    if intervals:
        avg_interval = statistics.mean(intervals)
        print(f"Intervalle moyen entre prédictions: {avg_interval:.1f}s")
        
        # Le rate limit à 10/minute = 6s minimum entre requêtes
        if avg_interval >= 5.5:  # Avec marge d'erreur
            print("⚠️ Rate limiting détecté - considérez l'ajuster pour les tests")
        else:
            print("✅ Pas de rate limiting apparent")

def main():
    """Fonction principale d'analyse."""
    db_path = os.path.join('app', 'monitoring_dashboard.db')
    if not os.path.exists(db_path):
        print(f"❌ Base de données non trouvée: {db_path}")
        return
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    try:
        analyze_performance_issues(conn)
        analyze_rate_limiting_impact(conn)
        suggest_optimizations()
        
        print("\n🎯 ACTIONS PRIORITAIRES:")
        print("1. Retirer @rate_limit pour les tests de performance")
        print("2. Vérifier la latence réseau vers MongoDB Atlas")
        print("3. Profiler le temps de prédiction du modèle")
        print("4. Réduire MONITOR_LEVEL=1 en production")
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    main()