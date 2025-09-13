"""
Supabase integration service for StudyGenie - Pure Functional Style (Integrated with student_service)
"""
from asyncio.log import logger
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json
from app.db.db_client import get_supabase_client
from app.services.student_service import (
    get_student_by_id,
    update_student_data,
    get_existing_student,
    create_new_student,
    StudentData,
)

client = get_supabase_client()


# Student management (using imported functions)
async def create_or_get_student(student_id: str, username: str, email: str, full_name: str) -> str:
    existing = await get_existing_student(student_id)
    if existing:
        return existing['student_id']
    
    student_data_dict = {
        'student_id': student_id, 'username': username, 'email': email,
        'full_name': full_name, 'learning_preferences': {},
        'created_at': datetime.now().isoformat(), 'updated_at': datetime.now().isoformat()
    }
    created = await create_new_student(student_data_dict)
    logger.info(f"Created student: {student_id}")
    return created['student_id']

async def update_student_service(student_id: str, full_name: Optional[str] = None, learning_preferences: Optional[Dict[str, Any]] = None) -> Optional[StudentData]:
    update_dict = {}
    if full_name is not None:
        update_dict["name"] = full_name
    if learning_preferences is not None:
        update_dict["learning_preferences"] = json.dumps(learning_preferences)
    if not update_dict:
        return get_student_by_id(student_id)
    return update_student_data(student_id, StudentData(**update_dict))


async def update_concept_progress(student_id: str, concept_id: int, correct_answers: int, total_questions: int):
    accuracy = (correct_answers / total_questions * 100) if total_questions > 0 else 0
    status = "mastered" if accuracy >= 90 else "in_progress" if accuracy >= 70 else "needs_review"
    
    progress_data = {
        'student_id': student_id, 'concept_id': concept_id, 'status': status,
        'mastery_score': accuracy, 'attempts_count': 1, 'correct_answers': correct_answers,
        'total_questions': total_questions, 'last_practiced': datetime.now().isoformat(),
        'updated_at': datetime.now().isoformat()
    }
    
    existing = client.table("student_concept_progress").select("*").eq("student_id", student_id).eq("concept_id", concept_id).execute()
    if existing.data:
        rec = existing.data[0]
        progress_data.update({
            'attempts_count': rec['attempts_count'] + 1,
            'correct_answers': rec['correct_answers'] + correct_answers,
            'total_questions': rec['total_questions'] + total_questions
        })
        total_acc = (progress_data['correct_answers'] / progress_data['total_questions']) * 100
        progress_data['mastery_score'] = (rec['mastery_score'] + total_acc) / 2
        progress_data['first_learned'] = rec.get('first_learned', datetime.now().isoformat())
    else:
        progress_data['first_learned'] = datetime.now().isoformat()
    
    await update_concept_progress(progress_data)
    logger.info(f"Updated progress {student_id}, {concept_id}")

async def record_learning_activity(student_id: str, concept_id: int, activity_type: str, activity_data: Dict[str, Any], score: Optional[float] = None, time_spent: Optional[int] = None):
    activity_record = {
        'student_id': student_id, 'concept_id': concept_id, 'activity_type': activity_type,
        'activity_data': activity_data, 'score': score, 'time_spent': time_spent,
        'completed_at': datetime.now().isoformat()
    }
    await record_learning_activity(activity_record)
    logger.info(f"Recorded activity {student_id}")

async def create_or_get_subject(subject_name: str, description: str = "") -> int:
    try:
        return await create_or_get_subject(subject_name, description)
    except Exception as e:
        logger.error(f"Subject {subject_name}: {e}")
        raise

async def create_or_get_chapter(subject_id: int, chapter_name: str, description: str = "") -> int:
    try:
        return await create_or_get_chapter(subject_id, chapter_name, description)
    except Exception as e:
        logger.error(f"Chapter {chapter_name}: {e}")
        raise

async def create_or_get_concept(chapter_id: int, concept_name: str, difficulty_level: str = "Medium", description: str = "") -> int:
    try:
        return await create_or_get_concept(chapter_id, concept_name, difficulty_level, description)
    except Exception as e:
        logger.error(f"Concept {concept_name}: {e}")
        raise

async def get_recommendations(student_id: str, active_only: bool = True) -> List[Dict[str, Any]]:
    try:
        recs = await get_recommendations(student_id, active_only)
        generated = await generate_personalized_recommendations(student_id)
        if generated:
            await save_recommendations(student_id, generated)
            return generated
        return recs
    except Exception as e:
        logger.error(f"Recommendations {student_id}: {e}")
        try:
            return await generate_personalized_recommendations(student_id)
        except:
            return []

async def save_recommendations(student_id: str, recommendations: List[Dict[str, Any]]) -> bool:
    try:
        return await save_recommendations(student_id, recommendations)
    except Exception as e:
        logger.error(f"Save recs {student_id}: {e}")
        return False

async def generate_personalized_recommendations(student_id: str) -> List[Dict[str, Any]]:
    try:
        progress_data = await get_student_progress(student_id)
        activities_resp = client.table("learning_activities").select("""
            *, concepts:concept_id (concept_name, chapters:chapter_id (chapter_name, subjects:subject_id (subject_name)))
        """).eq("student_id", student_id).order("completed_at", desc=True).execute()
        activities_data = activities_resp.data or []

        if not progress_data and not activities_data:
            return [
                {'recommendation_type': 'onboarding', 'concept_id': None, 'title': 'Welcome to StudyGenie!', 'description': 'Start by uploading materials', 'priority_score': 10, 'is_active': True, 'is_completed': False},
                {'recommendation_type': 'first_quiz', 'concept_id': None, 'title': 'Take Your First Quiz', 'description': 'Complete a quiz to understand your level', 'priority_score': 8, 'is_active': True, 'is_completed': False},
                {'recommendation_type': 'explore_features', 'concept_id': None, 'title': 'Explore Learning Features', 'description': 'Try flashcards and quizzes', 'priority_score': 6, 'is_active': True, 'is_completed': False}
            ]

        recommendations = []
        if activities_data and not progress_data:
            recent = activities_data[:10]
            low_perf = [act for act in recent if (act.get('activity_data', {}).get('correct_answers', 0) / max(1, act.get('activity_data', {}).get('total_questions', 1))) * 100 < 70]
            for c in low_perf[:5]:
                acc = (c.get('activity_data', {}).get('correct_answers', 0) / max(1, c.get('activity_data', {}).get('total_questions', 1))) * 100
                recommendations.append({
                    'recommendation_type': 'concept_review', 'concept_id': c.get('concept_id'), 'title': f"Improve {c.get('concepts', {}).get('concept_name', 'Concept')}",
                    'description': f"Accuracy {acc:.1f}%. Practice more.", 'priority_score': 10 if acc < 50 else 7, 'is_active': True, 'is_completed': False
                })
            if len(recommendations) < 3:
                recommendations.append({'recommendation_type': 'continue_learning', 'concept_id': None, 'title': 'Continue Learning', 'description': 'Practice quizzes and flashcards', 'priority_score': 6, 'is_active': True, 'is_completed': False})
            return recommendations

        weak_concepts = [p for p in progress_data if p.get('status') == 'needs_review' or p.get('mastery_score', 0) < 70]
        for c in weak_concepts[:5]:
            pri = 10 if c.get('mastery_score', 0) < 50 else 7
            name = c.get('concepts', {}).get('concept_name', 'Concept')
            recommendations.append({
                'recommendation_type': 'concept_review', 'concept_id': c.get('concept_id'), 'title': f"Review {name}",
                'description': 'Improve understanding', 'priority_score': pri, 'is_active': True, 'is_completed': False
            })

        strong_concepts = [p for p in progress_data if p.get('status') == 'mastered' and (datetime.now() - datetime.fromisoformat(p.get('last_practiced', '').replace('Z', '+00:00'))).days > 7]
        for c in strong_concepts[:3]:
            name = c.get('concepts', {}).get('concept_name', 'Concept')
            recommendations.append({
                'recommendation_type': 'maintenance_practice', 'concept_id': c.get('concept_id'), 'title': f"Practice {name}",
                'description': 'Keep skills sharp', 'priority_score': 4, 'is_active': True, 'is_completed': False
            })

        return recommendations
    except Exception as e:
        logger.error(f"Generate recs {student_id}: {e}")
        return [{'recommendation_type': 'system_error', 'concept_id': None, 'title': 'Continue Learning', 'description': 'Practice materials', 'priority_score': 5, 'is_active': True, 'is_completed': False}]

async def process_llm_response(student_id: str, subject_name: str, chapter_name: str, concept_name: str, llm_response: Dict[str, Any], user_query: str) -> Dict[str, Any]:
    try:
        metadata = llm_response.get('metadata', {})
        diff = metadata.get('difficulty_level', 'Medium')
        est_time = metadata.get('estimated_study_time', '')
        objectives = llm_response.get('learning_objectives', [])
        summary = llm_response.get('summary', '')

        desc = f"Auto-created from LLM. {summary}"
        if objectives:
            desc += f"\nObjectives: {'; '.join(objectives)}"
        if est_time:
            desc += f"\nTime: {est_time}"

        sub_id = await create_or_get_subject(subject_name, "Auto-created")
        chap_id = await create_or_get_chapter(sub_id, chapter_name, "Auto-created")
        conc_id = await create_or_get_concept(chap_id, concept_name, diff, desc)

        act_data = {
            'llm_response': llm_response, 'user_query': user_query, 'timestamp': datetime.now().isoformat(),
            'response_type': 'structured_content', 'metadata': metadata, 'difficulty_level': diff,
            'estimated_study_time': est_time, 'learning_objectives_count': len(objectives),
            'flashcards_count': len(llm_response.get('flashcards', [])), 'quiz_questions_count': len(llm_response.get('quiz', []))
        }
        await record_learning_activity(student_id, conc_id, "content_study", act_data)

        enhanced = llm_response.copy()
        track_meta = {
            'student_id': student_id, 'subject_id': sub_id, 'chapter_id': chap_id, 'concept_id': conc_id,
            'subject_name': subject_name, 'chapter_name': chapter_name, 'concept_name': concept_name,
            'difficulty_level': diff, 'estimated_study_time': est_time, 'learning_objectives': objectives,
            'content_summary': summary, 'created_at': datetime.now()
        }
        return {"enhanced_response": enhanced, "tracking_metadata": track_meta, "created_entities": {"subject_created": True, "chapter_created": True, "concept_created": True}}
    except Exception as e:
        logger.error(f"Process LLM: {e}")
        raise

# Ensure Student
async def ensure_student_exists(student_id: str):
    existing = get_student_by_id(student_id)
    if not existing:
        await create_or_get_student(student_id, student_id, student_id, f"User {student_id}")
        logger.info(f"Created {student_id}")
        
async def save_concept_progress(student_id: str, subject_name: str, concept_name: str, mastery_level: float, correct_answers: int, total_questions: int, time_spent: int, difficulty_level: str, activity_type: str):
    await ensure_student_exists(student_id)
    sub_id = await create_or_get_subject(subject_name)
    chap_id = await create_or_get_chapter(sub_id, f"{subject_name} Concepts")
    conc_id = await create_or_get_concept(chap_id, concept_name, difficulty_level)
    
    await update_concept_progress(student_id, conc_id, correct_answers, total_questions)
    act_data = {'activity_type': activity_type, 'correct_answers': correct_answers, 'total_questions': total_questions, 'mastery_level': mastery_level, 'difficulty_level': difficulty_level}
    await record_learning_activity(student_id, conc_id, activity_type, act_data, mastery_level, time_spent)

async def record_quiz_attempt(student_id: str, concept_id: int, quiz_data: Dict[str, Any], score: float, time_spent: int) -> Dict[str, Any]:
    questions = quiz_data.get('questions', [])
    answers = quiz_data.get('answers', [])
    correct = sum(1 for i, ans in enumerate(answers) if i < len(questions) and ans == questions[i].get('correct_answer'))
    total = len(questions)
    
    await update_concept_progress(student_id, concept_id, correct, total, time_spent)
    act_data = {'quiz_data': quiz_data, 'answers': answers, 'score': score, 'correct_answers': correct, 'total_questions': total}
    await record_learning_activity(student_id, concept_id, "quiz_attempt", act_data, score, time_spent)
    
    return {"quiz_recorded": True, "correct_answers": correct, "total_questions": total, "accuracy": (correct / total * 100) if total > 0 else 0, "score": score}

async def get_student_progress(student_id: str) -> Dict[str, Any]:
    try:
        prog_data = await get_student_progress(student_id)
        subjects = {}
        total_concepts, mastered_concepts = 0, 0
        
        for p in prog_data:
            conc = p.get('concepts', {})
            chap = conc.get('chapters', {})
            sub = chap.get('subjects', {})
            s_name, c_name, conc_name = sub.get('subject_name', 'Unknown'), chap.get('chapter_name', 'Unknown'), conc.get('concept_name', 'Unknown')
            
            if s_name not in subjects:
                subjects[s_name] = {'chapters': {}, 'total_concepts': 0, 'mastered_concepts': 0}
            
            if c_name not in subjects[s_name]['chapters']:
                subjects[s_name]['chapters'][c_name] = {'concepts': [], 'mastery_rate': 0}
            
            conc_prog = {
                'concept_name': conc_name, 'status': p.get('status', 'not_started'), 'mastery_score': p.get('mastery_score', 0),
                'attempts_count': p.get('attempts_count', 0), 'last_practiced': p.get('last_practiced'), 'first_learned': p.get('first_learned')
            }
            subjects[s_name]['chapters'][c_name]['concepts'].append(conc_prog)
            subjects[s_name]['total_concepts'] += 1
            total_concepts += 1
            if p.get('status') == 'mastered':
                subjects[s_name]['mastered_concepts'] += 1
                mastered_concepts += 1
        
        for sub in subjects.values():
            for chap in sub['chapters'].values():
                m_count = sum(1 for c in chap['concepts'] if c['status'] == 'mastered')
                chap['mastery_rate'] = (m_count / len(chap['concepts']) * 100) if chap['concepts'] else 0
        
        overall = (mastered_concepts / total_concepts * 100) if total_concepts > 0 else 0
        return {'student_id': student_id, 'subjects': subjects, 'overall_stats': {'total_concepts': total_concepts, 'mastered_concepts': mastered_concepts, 'overall_mastery_rate': overall}}
    except Exception as e:
        logger.error(f"Get progress {student_id}: {e}")
        return {'student_id': student_id, 'subjects': {}, 'overall_stats': {}}

async def get_learning_analytics(student_id: str, days: int = 30) -> Dict[str, Any]:
    try:
        prog_resp = client.table("student_concept_progress").select("""
            *, concepts:concept_id (concept_name, chapters:chapter_id (chapter_name, subjects:subject_id (subject_name)))
        """).eq("student_id", student_id).execute()
        prog_data = prog_resp.data or []
        
        act_resp = client.table("learning_activities").select("""
            *, concepts:concept_id (concept_name, chapters:chapter_id (chapter_name, subjects:subject_id (subject_name)))
        """).eq("student_id", student_id).gte("completed_at", (datetime.now() - timedelta(days=days)).isoformat()).execute()
        act_data = act_resp.data or []
        
        total_act = len(act_data)
        total_time = sum(a.get('time_spent', 0) for a in act_data)
        quiz_act = [a for a in act_data if a.get('activity_type') in ['quiz_attempt', 'quiz']]
        flash_act = [a for a in act_data if a.get('activity_type') in ['flashcard_practice', 'flashcard']]
        
        t_q = sum(a.get('activity_data', {}).get('total_questions', 0) for a in quiz_act) + sum(a.get('activity_data', {}).get('total_questions', 0) for a in flash_act)
        t_c = sum(a.get('activity_data', {}).get('correct_answers', 0) for a in quiz_act) + sum(a.get('activity_data', {}).get('correct_answers', 0) for a in flash_act)
        quiz_acc = (t_c / t_q * 100) if t_q > 0 else 0
        conc_learned = len([p for p in prog_data if p.get('mastery_score', 0) >= 80])
        
        dates = sorted(set(datetime.fromisoformat(a['completed_at']).date() for a in act_data if a.get('completed_at')))
        streak = 0
        if dates:
            curr = datetime.now().date()
            streak = 1 if curr in dates else (1 if (curr - timedelta(1)) in dates else 0)
            check = curr - timedelta(streak + 1)
            while check in dates:
                streak += 1
                check -= timedelta(1)
        
        subjects_analytics = {}
        for p in prog_data:
            info = p.get('concepts', {})
            if not info:
                continue
            chap_info = info.get('chapters', {})
            if not chap_info:
                continue
            sub_info = chap_info.get('subjects', {})
            if not sub_info:
                continue
            s_name, c_name, conc_name = sub_info.get('subject_name', 'Unknown'), chap_info.get('chapter_name', 'Unknown'), info.get('concept_name', 'Unknown')
            mastery = p.get('mastery_score', 0)
            
            if s_name not in subjects_analytics:
                subjects_analytics[s_name] = {'subject_name': s_name, 'total_concepts': 0, 'mastered_concepts': 0, 'time_spent': 0, 'activities_count': 0, 'quiz_accuracy': 0, 'chapters': {}, 'concepts': []}
            
            subjects_analytics[s_name]['total_concepts'] += 1
            if mastery >= 80:
                subjects_analytics[s_name]['mastered_concepts'] += 1
            subjects_analytics[s_name]['concepts'].append({
                'concept_name': conc_name, 'chapter_name': c_name, 'mastery_score': mastery,
                'total_attempts': p.get('total_attempts', 0), 'correct_answers': p.get('correct_answers', 0),
                'total_questions': p.get('total_questions', 0), 'last_updated': p.get('last_updated')
            })
            
            if c_name not in subjects_analytics[s_name]['chapters']:
                subjects_analytics[s_name]['chapters'][c_name] = {'chapter_name': c_name, 'concepts_count': 0, 'mastered_count': 0, 'average_mastery': 0, 'concepts': []}
            
            subjects_analytics[s_name]['chapters'][c_name]['concepts'].append({
                'concept_name': conc_name, 'mastery_score': mastery, 'attempts_count': p.get('total_attempts', 0),
                'correct_answers': p.get('correct_answers', 0), 'total_questions': p.get('total_questions', 0),
                'last_practiced': p.get('last_updated'), 'status': 'mastered' if mastery >= 80 else 'needs_review' if mastery < 60 else 'in_progress'
            })
            subjects_analytics[s_name]['chapters'][c_name]['concepts_count'] += 1
            if mastery >= 80:
                subjects_analytics[s_name]['chapters'][c_name]['mastered_count'] += 1
        
        for a in act_data:
            info = a.get('concepts', {})
            if not info:
                continue
            chap_info = info.get('chapters', {})
            if not chap_info:
                continue
            sub_info = chap_info.get('subjects', {})
            if not sub_info:
                continue
            s_name = sub_info.get('subject_name', 'Unknown')
            if s_name in subjects_analytics:
                subjects_analytics[s_name]['time_spent'] += a.get('time_spent', 0)
                subjects_analytics[s_name]['activities_count'] += 1
        
        for s_name, s_data in subjects_analytics.items():
            s_quiz = [a for a in act_data if a.get('activity_type') == 'quiz' and a.get('concepts', {}).get('chapters', {}).get('subjects', {}).get('subject_name') == s_name]
            s_tq = sum(a.get('activity_data', {}).get('total_questions', 0) for a in s_quiz)
            s_tc = sum(a.get('activity_data', {}).get('correct_answers', 0) for a in s_quiz)
            s_data['quiz_accuracy'] = (s_tc / s_tq * 100) if s_tq > 0 else 0
            m_c, t_c = s_data.get('mastered_concepts', 0), s_data.get('total_concepts', 0)
            s_data['mastery_percentage'] = (m_c / t_c * 100) if t_c > 0 else 0
            for c_name, c_data in s_data['chapters'].items():
                m_c, c_c = c_data.get('mastered_count', 0), c_data.get('concepts_count', 0)
                c_data['mastery_rate'] = (m_c / c_c * 100) if c_c > 0 else 0
                c_concs = c_data.get('concepts', [])
                c_data['average_mastery'] = sum(con.get('mastery_score', 0) for con in c_concs) / len(c_concs) if c_concs else 0
        
        return {
            'student_id': student_id, 'period_days': days, 'study_streak': streak, 'concepts_learned': conc_learned,
            'time_spent': total_time, 'quiz_accuracy': quiz_acc, 'total_activities': total_act, 'quiz_count': len(quiz_act),
            'flashcard_count': len(flash_act), 'total_questions': t_q, 'correct_answers': t_c, 'concepts_mastered': conc_learned,
            'session_count': total_act, 'average_score': quiz_acc / 100 if quiz_acc > 0 else 0, 'activity_timeline': act_data[-10:],
            'subjects_analytics': subjects_analytics, 'subjects_summary': [
                {'subject_name': s_name, 'mastery_percentage': s_data['mastery_percentage'], 'time_spent': s_data['time_spent'],
                 'concepts_total': s_data['total_concepts'], 'concepts_mastered': s_data['mastered_concepts'],
                 'quiz_accuracy': s_data['quiz_accuracy'], 'activities_count': s_data['activities_count']}
                for s_name, s_data in subjects_analytics.items()
            ]
        }
    except Exception as e:
        logger.error(f"Analytics {student_id}: {e}")
        return {'student_id': student_id, 'period_days': days, 'study_streak': 0, 'concepts_learned': 0, 'time_spent': 0, 'quiz_accuracy': 0, 'total_activities': 0, 'total_questions': 0, 'correct_answers': 0, 'concepts_mastered': 0, 'session_count': 0, 'average_score': 0, 'activity_timeline': []}

# Service dict
def get_learning_progress_service():
    return {
        'create_or_get_student': create_or_get_student,  
        'update_student': update_student_service,
        'update_learning_preferences': lambda sid, prefs: update_student_service(sid, learning_preferences=prefs),
        'get_or_create_subject': create_or_get_subject,
        'get_or_create_chapter': create_or_get_chapter,
        'get_or_create_concept': create_or_get_concept,
        'process_llm_response': process_llm_response,
        'update_concept_progress': update_concept_progress,
        'ensure_student_exists': ensure_student_exists,
        'save_concept_progress': save_concept_progress,
        'save_recommendations': save_recommendations,
        'record_quiz_attempt': record_quiz_attempt,
        'get_student_progress': get_student_progress,
        'get_recommendations': get_recommendations,
        'get_learning_analytics': get_learning_analytics    }