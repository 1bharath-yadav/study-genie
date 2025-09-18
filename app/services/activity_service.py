from typing import Any, Dict, Optional
import json
from .db import create_supabase_client, safe_extract_single


async def insert_activity(activity: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Insert a student activity row and return the created row.
    Expects activity dict containing student_id, activity_type, payload, optional score/time_spent and related ids.
    """
    client = create_supabase_client()
    if not client:
        return None

    payload = activity.get('payload') or {}
    # Accept stringified JSON payloads and coerce to dict
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except Exception:
            payload = {}

    insert_obj = {
        'student_id': activity.get('student_id'),
        'activity_type': activity.get('activity_type'),
        'related_subject_id': activity.get('related_subject_id'),
        'related_chapter_id': activity.get('related_chapter_id'),
        'related_concept_id': activity.get('related_concept_id'),
        'payload': payload,
        'score': activity.get('score'),
        'time_spent_seconds': activity.get('time_spent_seconds'),
    }

    resp = client.table('student_activity').insert(insert_obj).execute()
    return safe_extract_single(resp)


async def get_activity_summary(student_id: str, days: int = 30) -> Dict[str, Any]:
    """Return simple aggregates over recent activity for analytics dashboard."""
    client = create_supabase_client()
    if not client:
        return {}

    try:
        r = client.table('student_activity').select('activity_type, score, time_spent_seconds, created_at').gte(
            'created_at', f"now() - interval '{days} days'"
        ).eq('student_id', student_id).execute()
        data = r.data or []

        total = len(data)
        distinct_types = len(set(d.get('activity_type') for d in data if d.get('activity_type')))

        scores = []
        for d in data:
            s = d.get('score')
            try:
                if s is not None:
                    scores.append(float(s))
            except Exception:
                continue

        times = []
        for d in data:
            try:
                times.append(int(d.get('time_spent_seconds') or 0))
            except Exception:
                times.append(0)

        avg_score = sum(scores) / len(scores) if scores else 0
        total_time = sum(times)
        return {'total_activities': total, 'distinct_activity_types': distinct_types, 'avg_score': avg_score, 'total_time_spent': total_time}
    except Exception:
        return {}


async def get_weakness_analysis(student_id: str, days: int = 30, top_n: int = 10) -> Dict[str, Any]:
    """Compute weak concepts/subjects based on recent activity.

    Strategy:
    - Fetch recent activity rows for the student
    - Prefer explicit related_concept_id / related_subject_id columns; fall back to payload fields
    - For each concept/subject compute count, avg_score and failure rate (when score or payload indicates correctness)
    - Return top_n weakest concepts and subjects with simple recommendations
    """
    client = create_supabase_client()
    if not client:
        return {}

    try:
        r = client.table('student_activity').select('related_concept_id, related_subject_id, payload, score, activity_type, created_at').gte(
            'created_at', f"now() - interval '{days} days'"
        ).eq('student_id', student_id).execute()
        rows = r.data or []

        concept_stats: Dict[str, Dict[str, Any]] = {}
        subject_stats: Dict[str, Dict[str, Any]] = {}

        for d in rows:
            # Normalize payload to a dict if it's stringified JSON
            raw_payload = d.get('payload')
            if isinstance(raw_payload, str):
                try:
                    payload = json.loads(raw_payload)
                except Exception:
                    payload = {}
            elif isinstance(raw_payload, dict):
                payload = raw_payload
            else:
                payload = {}

            # Determine ids (prefer explicit related_* columns)
            c_id = d.get('related_concept_id') or payload.get('concept_id')
            s_id = d.get('related_subject_id') or payload.get('subject_id')

            # score may be present
            score = d.get('score')
            try:
                score = float(score) if score is not None else None
            except Exception:
                score = None

            # Try to infer correctness from payload flags
            is_correct = None
            for k in ('correct', 'is_correct', 'was_correct', 'result', 'isRight'):
                if isinstance(payload.get(k), bool):
                    is_correct = payload.get(k)
                    break
            # also support numeric result where 1==correct
            if is_correct is None and isinstance(payload.get('correct'), (int, float)):
                is_correct = bool(payload.get('correct'))

            if c_id:
                key = str(c_id)
                st = concept_stats.setdefault(key, {'count': 0, 'scores': [], 'fails': 0})
                st['count'] += 1
                if score is not None:
                    st['scores'].append(score)
                if is_correct is False:
                    st['fails'] += 1

            if s_id:
                key = str(s_id)
                st = subject_stats.setdefault(key, {'count': 0, 'scores': [], 'fails': 0})
                st['count'] += 1
                if score is not None:
                    st['scores'].append(score)
                if is_correct is False:
                    st['fails'] += 1

        def summarize(stats: Dict[str, Dict[str, Any]], id_label: str):
            out = []
            for id_, v in stats.items():
                avg_score = (sum(v['scores']) / len(v['scores'])) if v['scores'] else None
                fail_rate = (v['fails'] / v['count']) if v['count'] > 0 else 0
                weakness_score = (1 - (avg_score or 0) / 100) * (v['count']) + fail_rate * 10
                out.append({id_label: id_, 'count': v['count'], 'avg_score': avg_score, 'fail_rate': round(fail_rate, 3), 'weakness_score': weakness_score})
            out.sort(key=lambda x: x['weakness_score'], reverse=True)
            return out

        concepts = summarize(concept_stats, 'concept_id')[:top_n]
        subjects = summarize(subject_stats, 'subject_id')[:top_n]

        # Build simple recommendations per weak concept
        recommendations = []
        for c in concepts:
            recommendations.append({
                'concept_id': c['concept_id'],
                'reason': f"Low avg score ({(c['avg_score'] or 0):.1f}) over {c['count']} activities",
                'action': 'Recommend targeted revision and flashcard practice.'
            })

        return {
            'concept_weaknesses': concepts,
            'subject_weaknesses': subjects,
            'recommendations': recommendations,
        }
    except Exception:
        return {}


async def get_weekly_trends(student_id: str, weeks: int = 4) -> Dict[str, Any]:
    """Return weekly aggregates for the past `weeks` weeks: total activities, avg_score, total_time_spent per-week."""
    from datetime import datetime, timedelta

    client = create_supabase_client()
    if not client:
        return {}

    try:
        days = max(7, weeks * 7)
        r = client.table('student_activity').select('score, time_spent_seconds, created_at').gte(
            'created_at', f"now() - interval '{days} days'"
        ).eq('student_id', student_id).execute()
        rows = r.data or []

        # Build week buckets keyed by ISO year-week (YYYY-WW)
        buckets: Dict[str, Dict[str, Any]] = {}
        for d in rows:
            created = d.get('created_at')
            try:
                dt = datetime.fromisoformat(created.replace('Z', '+00:00')) if isinstance(created, str) else datetime.utcnow()
            except Exception:
                dt = datetime.utcnow()
            year, week, _ = dt.isocalendar()
            key = f"{year}-{week:02d}"
            b = buckets.setdefault(key, {'total_activities': 0, 'scores': [], 'total_time_spent': 0, 'week_start': (dt - timedelta(days=dt.weekday())).date().isoformat()})
            b['total_activities'] += 1
            s = d.get('score')
            try:
                if s is not None:
                    b['scores'].append(float(s))
            except Exception:
                pass
            try:
                b['total_time_spent'] += int(d.get('time_spent_seconds') or 0)
            except Exception:
                pass

        # Convert to sorted list for last `weeks` weeks
        items = []
        # ensure we include empty weeks
        today = datetime.utcnow().date()
        for i in range(weeks - 1, -1, -1):
            wk_start = today - timedelta(days=today.weekday()) - timedelta(weeks=i)
            year, week, _ = wk_start.isocalendar()
            key = f"{year}-{week:02d}"
            b = buckets.get(key, {'total_activities': 0, 'scores': [], 'total_time_spent': 0, 'week_start': wk_start.isoformat()})
            avg_score = (sum(b['scores']) / len(b['scores'])) if b.get('scores') else None
            items.append({'week': key, 'week_start': b.get('week_start'), 'total_activities': b.get('total_activities', 0), 'avg_score': avg_score, 'total_time_spent': b.get('total_time_spent', 0)})

        return {'weeks': items}
    except Exception:
        return {}


async def get_monthly_trends(student_id: str, months: int = 3) -> Dict[str, Any]:
    """Return monthly aggregates for the past `months` months: total activities, avg_score, total_time_spent per-month."""
    from datetime import datetime

    client = create_supabase_client()
    if not client:
        return {}
    try:
        # approximate days
        days = max(28, months * 30)
        r = client.table('student_activity').select('score, time_spent_seconds, created_at').gte(
            'created_at', f"now() - interval '{days} days'"
        ).eq('student_id', student_id).execute()
        rows = r.data or []

        buckets: Dict[str, Dict[str, Any]] = {}
        for d in rows:
            created = d.get('created_at')
            try:
                dt = datetime.fromisoformat(created.replace('Z', '+00:00')) if isinstance(created, str) else datetime.utcnow()
            except Exception:
                dt = datetime.utcnow()
            key = f"{dt.year}-{dt.month:02d}"
            b = buckets.setdefault(key, {'total_activities': 0, 'scores': [], 'total_time_spent': 0, 'month_start': dt.replace(day=1).date().isoformat()})
            b['total_activities'] += 1
            s = d.get('score')
            try:
                if s is not None:
                    b['scores'].append(float(s))
            except Exception:
                pass
            try:
                b['total_time_spent'] += int(d.get('time_spent_seconds') or 0)
            except Exception:
                pass

        # Build sorted list for last `months` months
        items = []
        today = datetime.utcnow().date()
        # iterate months backwards
        for i in range(months - 1, -1, -1):
            # compute first day of the month i months ago
            year = today.year
            month = today.month - i
            while month <= 0:
                month += 12
                year -= 1
            key = f"{year}-{month:02d}"
            b = buckets.get(key, {'total_activities': 0, 'scores': [], 'total_time_spent': 0, 'month_start': f"{year}-{month:02d}-01"})
            avg_score = (sum(b['scores']) / len(b['scores'])) if b.get('scores') else None
            items.append({'month': key, 'month_start': b.get('month_start'), 'total_activities': b.get('total_activities', 0), 'avg_score': avg_score, 'total_time_spent': b.get('total_time_spent', 0)})

        return {'months': items}
    except Exception:
        return {}
