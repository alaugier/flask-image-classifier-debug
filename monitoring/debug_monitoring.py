#!/usr/bin/env python3
"""
Script de debug pour analyser la base de données de monitoring Flask.
Usage: python scripts/debug_monitoring.py
"""

import sqlite3
import os
from datetime import datetime

def connect_monitoring_db():
    """Connexion à la base de monitoring."""
    db_path = os.path.join('app', 'monitoring_dashboard.db')
    if not os.path.exists(db_path):
        print(f"❌ Base de données non trouvée: {db_path}")
        return None
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # Pour accéder aux colonnes par nom
    return conn

def show_tables(conn):
    """Affiche toutes les tables de la base."""
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    print(f"📋 Tables disponibles ({len(tables)}):")
    for table in sorted(tables):
        cursor.execute(f"SELECT COUNT(*) FROM {table};")
        count = cursor.fetchone()[0]
        print(f"  - {table}: {count} entrées")
    return tables

def analyze_requests(conn):
    """Analyse détaillée des requêtes."""
    cursor = conn.cursor()
    
    print("\n📊 ANALYSE DES REQUÊTES")
    print("=" * 50)
    
    # Requêtes par endpoint
    query = """
    SELECT e.name as endpoint, 
           COUNT(r.id) as total_requests,
           AVG(r.duration) as avg_duration,
           MAX(r.duration) as max_duration,
           MIN(r.duration) as min_duration
    FROM fmd_Request r
    JOIN fmd_Endpoint e ON r.endpoint_id = e.id
    GROUP BY e.name
    ORDER BY total_requests DESC;
    """
    
    cursor.execute(query)
    results = cursor.fetchall()
    
    if results:
        print(f"{'Endpoint':<20} {'Requêtes':<10} {'Moy (ms)':<12} {'Max (ms)':<12} {'Min (ms)':<12}")
        print("-" * 75)
        for row in results:
            print(f"{row['endpoint']:<20} {row['total_requests']:<10} "
                  f"{row['avg_duration']*1000:<12.1f} {row['max_duration']*1000:<12.1f} "
                  f"{row['min_duration']*1000:<12.1f}")
    else:
        print("Aucune donnée de requête trouvée")

def analyze_recent_activity(conn):
    """Analyse de l'activité récente."""
    cursor = conn.cursor()
    
    print("\n🕐 ACTIVITÉ RÉCENTE (10 dernières requêtes)")
    print("=" * 60)
    
    query = """
    SELECT e.name as endpoint, 
           r.duration,
           r.time_requested,
           r.status_code,
           r.ip
    FROM fmd_Request r
    JOIN fmd_Endpoint e ON r.endpoint_id = e.id
    ORDER BY r.time_requested DESC
    LIMIT 10;
    """
    
    cursor.execute(query)
    results = cursor.fetchall()
    
    if results:
        for row in results:
            timestamp = datetime.fromisoformat(row['time_requested'].replace('Z', '+00:00'))
            duration_ms = row['duration'] * 1000
            print(f"{timestamp.strftime('%H:%M:%S')} | {row['endpoint']:<15} | "
                  f"{duration_ms:>6.1f}ms | {row['status_code']} | {row['ip']}")
    else:
        print("Aucune activité récente trouvée")

def analyze_performance(conn):
    """Analyse des performances."""
    cursor = conn.cursor()
    
    print("\n⚡ ANALYSE DE PERFORMANCE")
    print("=" * 40)
    
    # Requêtes les plus lentes
    query = """
    SELECT e.name as endpoint,
           r.duration,
           r.time_requested
    FROM fmd_Request r
    JOIN fmd_Endpoint e ON r.endpoint_id = e.id
    ORDER BY r.duration DESC
    LIMIT 5;
    """
    
    cursor.execute(query)
    slow_requests = cursor.fetchall()
    
    if slow_requests:
        print("🐌 5 requêtes les plus lentes:")
        for req in slow_requests:
            duration_ms = req['duration'] * 1000
            timestamp = datetime.fromisoformat(req['time_requested'].replace('Z', '+00:00'))
            print(f"  {req['endpoint']}: {duration_ms:.1f}ms ({timestamp.strftime('%H:%M:%S')})")
    
    # Statistiques générales
    cursor.execute("SELECT COUNT(*) as total, AVG(duration) as avg_duration FROM fmd_Request;")
    stats = cursor.fetchone()
    
    if stats and stats['total'] > 0:
        print(f"\n📈 Statistiques générales:")
        print(f"  Total requêtes: {stats['total']}")
        print(f"  Durée moyenne: {stats['avg_duration']*1000:.1f}ms")

def main():
    """Fonction principale."""
    print("🔍 Debug du monitoring Flask-MonitoringDashboard")
    print("=" * 55)
    
    conn = connect_monitoring_db()
    if not conn:
        return
    
    try:
        tables = show_tables(conn)
        
        if 'fmd_Request' in tables and 'fmd_Endpoint' in tables:
            analyze_requests(conn)
            analyze_recent_activity(conn)
            analyze_performance(conn)
        else:
            print("⚠️ Tables de monitoring non trouvées - lancez l'application d'abord")
            
    except Exception as e:
        print(f"❌ Erreur lors de l'analyse: {e}")
    finally:
        conn.close()
        
    print(f"\n💡 Pour des requêtes personnalisées:")
    print(f"   sqlite3 app/monitoring_dashboard.db")

if __name__ == "__main__":
    main()