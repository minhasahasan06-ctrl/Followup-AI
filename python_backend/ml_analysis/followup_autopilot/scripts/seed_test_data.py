"""
Test Data Seeding Script for Followup Autopilot

Creates a test patient with sample:
- Daily features (30 days)
- Patient signals (various categories)
- Trigger events
- Follow-up tasks
- Notifications

Usage:
    python seed_test_data.py [--patient-id <id>] [--days 30]

HIPAA Compliance:
- Uses synthetic data only
- No real PHI
- Audit logged

Wellness Positioning:
- All data reflects wellness monitoring use case
"""

import os
import sys
import logging
import argparse
from datetime import datetime, date, timedelta, timezone
from uuid import uuid4
from typing import Optional
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

logger = logging.getLogger(__name__)


def generate_daily_features(patient_id: str, days: int = 30) -> list:
    """Generate synthetic daily feature records"""
    features = []
    today = date.today()
    
    for i in range(days):
        feature_date = today - timedelta(days=i)
        
        base_pain = random.uniform(2, 4)
        base_fatigue = random.uniform(2, 4)
        base_mood = random.uniform(6, 8)
        
        if i < 7 and random.random() > 0.7:
            base_pain += random.uniform(1, 3)
            base_fatigue += random.uniform(1, 2)
            base_mood -= random.uniform(1, 2)
        
        features.append({
            'id': str(uuid4()),
            'patient_id': patient_id,
            'date': feature_date,
            'avg_pain': min(10, max(0, base_pain + random.uniform(-1, 1))),
            'avg_fatigue': min(10, max(0, base_fatigue + random.uniform(-1, 1))),
            'avg_mood': min(10, max(0, base_mood + random.uniform(-1, 1))),
            'checkins_count': random.randint(1, 5),
            'steps': random.randint(2000, 12000),
            'resting_hr': random.uniform(58, 78),
            'sleep_hours': random.uniform(5.5, 8.5),
            'weight': random.uniform(150, 180),
            'env_risk_score': random.uniform(10, 60),
            'pollen_index': random.uniform(0, 8),
            'aqi': random.uniform(20, 80),
            'temp_c': random.uniform(18, 28),
            'med_adherence_7d': random.uniform(0.7, 1.0),
            'mh_score': random.uniform(0.1, 0.4),
            'video_resp_risk': random.uniform(0.05, 0.3),
            'audio_emotion_score': random.uniform(0.1, 0.4),
            'pain_severity_score': random.uniform(0.1, 0.4),
            'engagement_rate_14d': random.uniform(0.6, 1.0),
        })
    
    return features


def generate_patient_signals(patient_id: str, days: int = 30) -> list:
    """Generate synthetic signal records"""
    signals = []
    now = datetime.now(timezone.utc)
    
    categories = ['device', 'symptom', 'video', 'audio', 'pain', 'mental', 'environment', 'meds']
    
    for i in range(days):
        signal_date = now - timedelta(days=i)
        
        for category in random.sample(categories, random.randint(2, 5)):
            signal_time = signal_date.replace(
                hour=random.randint(6, 22),
                minute=random.randint(0, 59)
            )
            
            if category == 'device':
                payload = {
                    'heart_rate': random.randint(55, 85),
                    'spo2': random.randint(95, 100),
                    'steps': random.randint(500, 3000),
                    'source': 'wearable_sync'
                }
                ml_score = None
            elif category == 'symptom':
                payload = {
                    'pain_level': random.randint(0, 6),
                    'fatigue_level': random.randint(0, 6),
                    'symptoms': random.sample(['headache', 'nausea', 'dizziness', 'fatigue'], random.randint(0, 2))
                }
                ml_score = (payload['pain_level'] + payload['fatigue_level']) / 20
            elif category == 'video':
                payload = {
                    'session_id': str(uuid4()),
                    'respiratory_risk': random.uniform(0.05, 0.35),
                    'completed_segments': random.randint(5, 7)
                }
                ml_score = payload['respiratory_risk']
            elif category == 'audio':
                payload = {
                    'session_id': str(uuid4()),
                    'emotion_score': random.uniform(0.1, 0.4),
                    'stress_level': random.uniform(0.1, 0.5)
                }
                ml_score = payload['emotion_score']
            elif category == 'pain':
                payload = {
                    'vas_score': random.randint(0, 6),
                    'joint': random.choice(['knee', 'back', 'shoulder', 'hip']),
                    'duration': random.choice(['hours', 'days', 'weeks'])
                }
                ml_score = payload['vas_score'] / 10
            elif category == 'mental':
                payload = {
                    'questionnaire_type': random.choice(['PHQ-9', 'GAD-7', 'PSS-10']),
                    'total_score': random.randint(0, 12),
                    'max_score': 27
                }
                ml_score = payload['total_score'] / payload['max_score']
            elif category == 'environment':
                payload = {
                    'aqi': random.randint(20, 80),
                    'pollen_index': random.uniform(0, 8),
                    'temperature': random.uniform(18, 28)
                }
                ml_score = payload['aqi'] / 300
            else:
                payload = {
                    'medication': random.choice(['Med A', 'Med B', 'Med C']),
                    'action': random.choice(['taken', 'taken', 'taken', 'missed', 'late']),
                }
                ml_score = 1.0 if payload['action'] == 'taken' else 0.0
            
            signals.append({
                'id': str(uuid4()),
                'patient_id': patient_id,
                'category': category,
                'source': f'{category}_source',
                'raw_payload': payload,
                'ml_score': ml_score,
                'signal_time': signal_time,
            })
    
    return signals


def generate_trigger_events(patient_id: str, count: int = 8) -> list:
    """Generate synthetic trigger events"""
    events = []
    now = datetime.now(timezone.utc)
    
    trigger_types = [
        ('missed_meds_pattern', 'warning'),
        ('env_risk_high_with_symptoms', 'alert'),
        ('mh_score_spike', 'alert'),
        ('anomaly_day', 'warning'),
        ('pain_spike', 'warning'),
    ]
    
    for i in range(count):
        trigger_name, severity = random.choice(trigger_types)
        created_at = now - timedelta(days=random.randint(0, 14), hours=random.randint(0, 12))
        
        events.append({
            'id': str(uuid4()),
            'patient_id': patient_id,
            'name': trigger_name,
            'severity': severity,
            'context': {'triggered_by': 'autopilot', 'day': i},
            'created_at': created_at,
        })
    
    return events


def generate_tasks(patient_id: str, count: int = 5) -> list:
    """Generate synthetic follow-up tasks"""
    tasks = []
    now = datetime.now(timezone.utc)
    
    task_types = [
        ('symptom_check', 'medium', 'symptoms'),
        ('med_adherence_check', 'medium', 'medications'),
        ('mh_check', 'high', 'mental_health'),
        ('pain_check', 'medium', 'paintrack'),
        ('resp_symptom_check', 'high', 'symptoms'),
    ]
    
    for i in range(count):
        task_type, priority, ui_tab = random.choice(task_types)
        due_hours = random.randint(-12, 48)
        due_at = now + timedelta(hours=due_hours)
        
        status = 'pending'
        if due_hours < -6:
            status = random.choice(['completed', 'pending'])
        
        tasks.append({
            'id': str(uuid4()),
            'patient_id': patient_id,
            'task_type': task_type,
            'task_description': f'Complete your {task_type.replace("_", " ")}',
            'priority': priority,
            'status': status,
            'due_at': due_at,
            'trigger_name': random.choice(['missed_meds_pattern', 'anomaly_day', 'routine']),
            'ui_tab_target': ui_tab,
            'metadata': {},
            'created_at': now - timedelta(hours=random.randint(1, 24)),
        })
    
    return tasks


def generate_notifications(patient_id: str, count: int = 4) -> list:
    """Generate synthetic notifications"""
    notifications = []
    now = datetime.now(timezone.utc)
    
    notification_templates = [
        ('Wellness Check Reminder', 'Time for your daily wellness check-in.', 'medium'),
        ('Medication Reminder', 'Remember to take your scheduled medication.', 'high'),
        ('Mental Health Check', 'A quick mental health check would be helpful.', 'medium'),
        ('Activity Goal', 'Great progress on your step goal today!', 'low'),
    ]
    
    for i in range(count):
        title, body, priority = random.choice(notification_templates)
        created_at = now - timedelta(hours=random.randint(1, 48))
        
        notifications.append({
            'id': str(uuid4()),
            'patient_id': patient_id,
            'channel': random.choice(['in_app', 'push']),
            'title': title,
            'body': body,
            'priority': priority,
            'status': 'pending',
            'is_read': random.random() > 0.6,
            'created_at': created_at,
        })
    
    return notifications


def seed_database(patient_id: str, days: int = 30, db_url: Optional[str] = None):
    """Seed the database with test data"""
    
    if not db_url:
        db_url = os.environ.get('DATABASE_URL')
    
    if not db_url:
        logger.error("No DATABASE_URL provided")
        print("Error: DATABASE_URL environment variable not set")
        return False
    
    try:
        import psycopg2
        import psycopg2.extras
        from psycopg2.extras import Json
        
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()
        
        print(f"Seeding data for patient: {patient_id}")
        
        features = generate_daily_features(patient_id, days)
        print(f"  Generated {len(features)} daily feature records")
        
        for f in features:
            cur.execute("""
                INSERT INTO autopilot_daily_features 
                (id, patient_id, date, avg_pain, avg_fatigue, avg_mood, checkins_count,
                 steps, resting_hr, sleep_hours, weight, env_risk_score, pollen_index,
                 aqi, temp_c, med_adherence_7d, mh_score, video_resp_risk, 
                 audio_emotion_score, pain_severity_score, engagement_rate_14d)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (patient_id, date) DO UPDATE SET
                    avg_pain = EXCLUDED.avg_pain,
                    avg_fatigue = EXCLUDED.avg_fatigue,
                    avg_mood = EXCLUDED.avg_mood,
                    engagement_rate_14d = EXCLUDED.engagement_rate_14d
            """, (
                f['id'], f['patient_id'], f['date'], f['avg_pain'], f['avg_fatigue'],
                f['avg_mood'], f['checkins_count'], f['steps'], f['resting_hr'],
                f['sleep_hours'], f['weight'], f['env_risk_score'], f['pollen_index'],
                f['aqi'], f['temp_c'], f['med_adherence_7d'], f['mh_score'],
                f['video_resp_risk'], f['audio_emotion_score'], f['pain_severity_score'],
                f['engagement_rate_14d']
            ))
        
        signals = generate_patient_signals(patient_id, days)
        print(f"  Generated {len(signals)} signal records")
        
        for s in signals:
            cur.execute("""
                INSERT INTO autopilot_patient_signals 
                (id, patient_id, category, source, raw_payload, ml_score, signal_time)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
            """, (
                s['id'], s['patient_id'], s['category'], s['source'],
                Json(s['raw_payload']), s['ml_score'], s['signal_time']
            ))
        
        events = generate_trigger_events(patient_id)
        print(f"  Generated {len(events)} trigger events")
        
        for e in events:
            cur.execute("""
                INSERT INTO autopilot_trigger_events 
                (id, patient_id, name, severity, context, created_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
            """, (
                e['id'], e['patient_id'], e['name'], e['severity'],
                Json(e['context']), e['created_at']
            ))
        
        tasks = generate_tasks(patient_id)
        print(f"  Generated {len(tasks)} tasks")
        
        for t in tasks:
            cur.execute("""
                INSERT INTO autopilot_followup_tasks 
                (id, patient_id, task_type, task_description, priority, status,
                 due_at, trigger_name, ui_tab_target, metadata, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
            """, (
                t['id'], t['patient_id'], t['task_type'], t['task_description'],
                t['priority'], t['status'], t['due_at'], t['trigger_name'],
                t['ui_tab_target'], Json(t['metadata']), t['created_at']
            ))
        
        notifications = generate_notifications(patient_id)
        print(f"  Generated {len(notifications)} notifications")
        
        for n in notifications:
            cur.execute("""
                INSERT INTO autopilot_notifications 
                (id, patient_id, channel, title, body, priority, status, is_read, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
            """, (
                n['id'], n['patient_id'], n['channel'], n['title'], n['body'],
                n['priority'], n['status'], n['is_read'], n['created_at']
            ))
        
        cur.execute("""
            INSERT INTO autopilot_patient_states 
            (patient_id, risk_score, risk_state, risk_components, last_updated, 
             last_checkin_at, next_followup_at, model_version, inference_confidence)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (patient_id) DO UPDATE SET
                risk_score = EXCLUDED.risk_score,
                risk_state = EXCLUDED.risk_state,
                risk_components = EXCLUDED.risk_components,
                last_updated = EXCLUDED.last_updated
        """, (
            patient_id,
            random.uniform(2, 6),
            random.choice(['Stable', 'AtRisk']),
            Json({'pain': random.uniform(10, 30), 'mental_health': random.uniform(10, 25), 
                  'adherence': random.uniform(5, 20), 'environment': random.uniform(5, 15)}),
            datetime.now(timezone.utc),
            datetime.now(timezone.utc) - timedelta(hours=random.randint(1, 12)),
            datetime.now(timezone.utc) + timedelta(hours=random.randint(12, 48)),
            '1.0.0',
            random.uniform(0.6, 0.9)
        ))
        
        conn.commit()
        cur.close()
        conn.close()
        
        print("\nSeed data created successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Failed to seed database: {e}")
        print(f"Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Seed test data for Followup Autopilot')
    parser.add_argument('--patient-id', type=str, default='test-patient-001',
                       help='Patient ID to seed data for')
    parser.add_argument('--days', type=int, default=30,
                       help='Number of days of data to generate')
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    success = seed_database(args.patient_id, args.days)
    
    if success:
        print(f"\nTest patient ID: {args.patient_id}")
        print("Use this ID to test the Autopilot API endpoints")
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
