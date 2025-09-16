
async def get_llm_response(uploaded_files_paths: List[Path], userprompt: str, temp_dir: str, user_api_key: str, user_id: str, provider_name: str, model_name: str, session_id: str | None = None, session_name: str | None = None) -> Dict[str, Any]:
    """
    Complete RAG pipeline for generating structured learning content
    """
    try:
        # Validate that user has provided API key
        if not user_api_key:
            return {
                "status": "error",
                "error": "API key is required. Please add your Gemini API key in settings to continue.",
                "error_type": "missing_api_key"
            }

        logger.info("Starting enhanced RAG pipeline...")

        # Step 1: Load documents from files
        logger.info("Loading documents from uploaded files...")
        documents = await load_documents_from_files([str(p) for p in uploaded_files_paths], temp_dir,user_api_key,provider_name,model_name)
        #  if no of (tokens words) is less than 1300 skip embedding send direct to llm
        # Auto-detect content type from user prompt early so both paths can use it
        content_type = detect_content_type(userprompt, "all")

        # If the uploaded documents are small (few words/tokens), skip embedding/retrieval
        # and send the combined text directly to the LLM. This avoids unnecessary embedding
        # calls for short inputs. Heuristic: use word count as a proxy for tokens.
        combined_text = "\n\n".join(documents)
        word_count = len(combined_text.split())
        if word_count < 1300:
            logger.info("Document small (words=%d) — skipping embedding/retrieval and sending direct to LLM", word_count)
            formatted_context = combined_text

            # Generate structured response directly without vector store
            response = await generate_structured_response(formatted_context, userprompt, provider_name, model_name, user_api_key, content_type)

            # Persist the LLM response into chat_history via service helper
            out_session_id = None
            try:
                from app.services.learning_history_service import upsert_chat_history

                # Build a guaranteed session name from LLM metadata: Subject | Chapter | Concept
                # This ensures the session_name is never optional and is consistent across writes.
                composed_session_name = None
                try:
                    if isinstance(response, dict):
                        meta = response.get('metadata') or {}
                        subject_name = (meta.get('subject_name') or meta.get('subject') or '').strip()
                        chapter_name = (meta.get('chapter_name') or meta.get('chapter') or '').strip()
                        concept_name = (meta.get('concept_name') or meta.get('concept') or '').strip()

                        composed_session_name = f"{subject_name}-{chapter_name}-{concept_name}"
                except Exception:
                    composed_session_name = None

                # prefer an explicit composed name, then caller-provided session_name, then fallback
                final_session_name = composed_session_name or session_name or f"Uncategorized | {datetime.now().date()}"

                # upsert_chat_history will write conversational entries into chat_history
                out_session_id = upsert_chat_history(user_id, session_id, final_session_name, userprompt, response)

                # Also, since this is a small-doc path where the full structured response is available,
                # persist the full structured learning content into study_material_history for the session.
                if isinstance(response, dict) and out_session_id:
                    try:
                        from app.db.db_client import get_supabase_client as _get_supabase_client
                        client = _get_supabase_client()
                        existing = client.table('chat_history').select('*').eq('session_id', out_session_id).execute()
                        if existing and getattr(existing, 'data', None) and len(existing.data) > 0:
                            rec = existing.data[0]
                            materials = rec.get('study_material_history') or []
                            materials.append(response)
                            now_iso_small = datetime.now().isoformat()
                            client.table('chat_history').update({'study_material_history': materials, 'updated_at': now_iso_small}).eq('session_id', out_session_id).execute()
                    except Exception:
                        logger.debug('Failed to persist study_material_history for small-doc path')
            except Exception as e:
                logger.debug('Skipping persistence to chat_history due to error: %s', e)

            # Also persist metadata (subject/chapter/concept) for small-doc path
            try:
                if isinstance(response, dict):
                    from app.db.db_client import get_supabase_client as _get_supabase_client

                    meta = response.get('metadata') or {}
                    subject_name = meta.get('subject_name')
                    chapter_name = meta.get('chapter_name')
                    concept_name = meta.get('concept_name')
                    difficulty = meta.get('difficulty_level') or meta.get('difficulty')

                    now_iso_local = datetime.now().isoformat()

                    if subject_name:
                        client = _get_supabase_client()

                        existing = client.table('subjects').select('*').eq('student_id', user_id).eq('llm_suggested_subject_name', subject_name).execute()
                        subject_id_val = None
                        if existing and getattr(existing, 'data', None):
                            rows = existing.data if isinstance(existing.data, list) else [existing.data]
                            if len(rows) > 0:
                                subject_id_val = rows[0].get('subject_id')
                        if not subject_id_val:
                            ins = {
                                'student_id': user_id,
                                'llm_suggested_subject_name': subject_name,
                                'created_at': now_iso_local,
                                'updated_at': now_iso_local
                            }
                            resp = client.table('subjects').insert(ins).execute()
                            if resp and getattr(resp, 'data', None):
                                inserted = resp.data[0] if isinstance(resp.data, list) else resp.data
                                subject_id_val = inserted.get('subject_id')

                        chapter_id_val = None
                        if subject_id_val and chapter_name:
                            existing = client.table('chapters').select('*').eq('student_id', user_id).eq('subject_id', subject_id_val).eq('llm_suggested_chapter_name', chapter_name).execute()
                            if existing and getattr(existing, 'data', None):
                                rows = existing.data if isinstance(existing.data, list) else [existing.data]
                                if len(rows) > 0:
                                    chapter_id_val = rows[0].get('chapter_id')
                            if not chapter_id_val:
                                ins = {
                                    'student_id': user_id,
                                    'subject_id': subject_id_val,
                                    'llm_suggested_chapter_name': chapter_name,
                                    'chapter_order': 0,
                                    'description': None,
                                    'created_at': now_iso_local,
                                    'updated_at': now_iso_local
                                }
                                resp = client.table('chapters').insert(ins).execute()
                                if resp and getattr(resp, 'data', None):
                                    inserted = resp.data[0] if isinstance(resp.data, list) else resp.data
                                    chapter_id_val = inserted.get('chapter_id')

                        if chapter_id_val and concept_name:
                            existing = client.table('concepts').select('*').eq('student_id', user_id).eq('chapter_id', chapter_id_val).eq('llm_suggested_concept_name', concept_name).execute()
                            if existing and getattr(existing, 'data', None):
                                pass
                            else:
                                ins = {
                                    'student_id': user_id,
                                    'chapter_id': chapter_id_val,
                                    'llm_suggested_concept_name': concept_name,
                                    'concept_order': 0,
                                    'description': None,
                                    'difficulty_level': difficulty or 'Medium',
                                    'created_at': now_iso_local,
                                    'updated_at': now_iso_local
                                }
                                resp = client.table('concepts').insert(ins).execute()
            except Exception as e:
                logger.error('Failed to persist small-doc LLM metadata: %s', e)

            return {
                'session_id': out_session_id,
                'llm_response': response
            } if isinstance(response, dict) else (response or {})
        else:
            # Step 2: Chunk documents for processing
            logger.info("Chunking documents for optimal processing...")
            chunks = await perform_document_chunking(documents)

        # Step 3: Determine embedding preference (per-user) and setup vector store
        logger.info("Determining embedding model preference for user and configuring retrieval...")
        from app.llm.providers import get_user_model_preferences
        from app.services.api_key_service import get_api_key_for_provider

        embedding_provider = provider_name
        embedding_model = model_name
        # Try to find an explicit embedding preference
        try:
            prefs = get_user_model_preferences(user_id)
            emb_pref = None
            for p in prefs:
                if p and p.get("use_for_embedding"):
                    emb_pref = p
                    break
            if emb_pref:
                mid = emb_pref.get("model_id")
                prov = emb_pref.get("provider_name") or None
                if mid and "-" in mid:
                    parts = mid.split("-", 1)
                    embedding_provider = prov or parts[0]
                    embedding_model = parts[1]
                else:
                    # fallback to provided provider_name if available
                    embedding_provider = prov or embedding_provider
                    embedding_model = mid or embedding_model
        except Exception:
            logger.debug("Failed to read embedding preferences; falling back to chat model")

        # Fetch embedding API key for the embedding provider (may be None)
        try:
            embedding_api_key = await get_api_key_for_provider(user_id, embedding_provider)
        except Exception:
            embedding_api_key = None

        logger.info("Using embedding provider=%s model=%s (key present=%s)", embedding_provider, embedding_model, bool(embedding_api_key))
        vector_store, hybrid_retriever = await setup_vector_store_and_retriever(chunks, embedding_provider, embedding_model, embedding_api_key)

        # Step 4: Retrieve relevant content using hybrid search
        logger.info(f"Retrieving relevant content for query: '{userprompt}'")
        retrieved_docs = await hybrid_retriever.ainvoke(userprompt)

        # Step 5: Enhance with surrounding context
        enhanced_docs = await enhance_retrieved_context(retrieved_docs, chunks)

        # Step 6: Format context for LLM processing
        formatted_context = format_context_for_llm(enhanced_docs)

        # Step 7: Generate structured response with user's API key
        logger.info("Generating structured learning content...")

        # Auto-detect content type from user prompt
        content_type = "all"  # default
        userprompt_lower = userprompt.lower()

        # Check for multiple content types
        has_flashcards = "flashcard" in userprompt_lower
        has_quiz = "quiz" in userprompt_lower
        has_match = "match" in userprompt_lower or "matching" in userprompt_lower
        has_summary = "summary" in userprompt_lower

        # If multiple content types are requested, use "all"
        content_types_count = sum([has_flashcards, has_quiz, has_match])

        if content_types_count > 1:
            content_type = "all"
        elif has_flashcards:
            content_type = "flashcards"
        elif has_quiz:
            content_type = "quiz"
        elif has_match:
            content_type = "match_the_following"
        elif has_summary:
            content_type = "summary"

        logger.info(
            f"Detected content type: {content_type} (flashcards:{has_flashcards}, quiz:{has_quiz}, match:{has_match})")
        logger.info(f"User prompt was: '{userprompt}'")
        logger.info(f"Content types count: {content_types_count}")
        response = await generate_structured_response(formatted_context, userprompt, provider_name, model_name, user_api_key, content_type)
        # Ensure we always return a dict
    # Persist the LLM response into chat_history for session/history tracking
    # Table schema: chat_history (id PK), session_id UUID, student_id, session_name,
    # llm_response_history (JSONB), study_material_history (JSONB), created_at, updated_at
        created_session_id = None
        try:
            from app.db.db_client import get_supabase_client
            import uuid

            client = get_supabase_client()

            # Determine session_name from response metadata when available when not provided
            try:
                meta = response.get('metadata') if isinstance(response, dict) else None
                subj = meta.get('subject_name') if meta else None
                conc = meta.get('concept_name') if meta else None
                if not session_name:
                    if subj and conc:
                        session_name = f"{subj} - {conc}"
                    elif subj:
                        session_name = subj
                    elif conc:
                        session_name = conc
            except Exception:
                pass

            now_iso = datetime.now().isoformat()

            # Append a user+assistant pair to chat_history so sessions have conversational memory
            try:
                user_entry = {'role': 'user', 'content': userprompt, 'timestamp': now_iso}
                assistant_content = None
                if isinstance(response, dict):
                    assistant_content = response.get('summary')
                    meta = response.get('metadata')
                    if not assistant_content and isinstance(meta, dict):
                        assistant_content = meta.get('summary')
                    if not assistant_content:
                        try:
                            assistant_content = json.dumps(response)
                        except Exception:
                            assistant_content = str(response)
                else:
                    assistant_content = response
                assistant_entry = {'role': 'assistant', 'content': assistant_content, 'timestamp': now_iso}

                if session_id:
                    existing = client.table('chat_history').select('*').eq('session_id', session_id).execute()
                    if existing and getattr(existing, 'data', None) and len(existing.data) > 0:
                        rec = existing.data[0]
                        history = rec.get('llm_response_history') or []
                        history.append(user_entry)
                        history.append(assistant_entry)
                        update_obj = {
                            'llm_response_history': history,
                            'session_name': session_name or rec.get('session_name'),
                            'updated_at': now_iso
                        }
                        # if we have a structured response, also append to study_material_history
                        if isinstance(response, dict):
                            materials = rec.get('study_material_history') or []
                            materials.append(response)
                            update_obj['study_material_history'] = materials

                        client.table('chat_history').update(update_obj).eq('session_id', session_id).execute()
                        created_session_id = session_id
                        logger.info('Appended to chat_history session_id=%s student=%s', session_id, user_id)
                    else:
                        row = {
                            'session_id': session_id,
                            'student_id': user_id,
                            'session_name': session_name,
                            'llm_response_history': [user_entry, assistant_entry],
                            'study_material_history': [response] if isinstance(response, dict) else [],
                            'created_at': now_iso,
                            'updated_at': now_iso
                        }
                        client.table('chat_history').insert(row).execute()
                        created_session_id = session_id
                        logger.info('Inserted new chat_history row with session_id=%s', session_id)
                else:
                    new_session_id = str(uuid.uuid4())
                    row = {
                        'session_id': new_session_id,
                        'student_id': user_id,
                        'session_name': session_name,
                        'llm_response_history': [user_entry, assistant_entry],
                        'study_material_history': [response] if isinstance(response, dict) else [],
                        'created_at': now_iso,
                        'updated_at': now_iso
                    }
                    client.table('chat_history').insert(row).execute()
                    created_session_id = new_session_id
                    logger.info('Inserted new chat_history row for student=%s session_id=%s', user_id, new_session_id)
            except Exception as e:
                logger.error('Failed to upsert chat_history for session: %s', e)
        except Exception as e:
            logger.debug('Skipping persistence to chat_history due to error: %s', e)

        # Persist extracted metadata (subject -> chapter -> concept) into DB so frontend can render
        try:
            # Only persist when response is a dict and contains metadata
            if isinstance(response, dict):
                # local imports to avoid top-level dependency in module import time
                from app.db.db_client import get_supabase_client as _get_supabase_client

                meta = response.get('metadata') or {}
                subject_name = meta.get('subject_name')
                chapter_name = meta.get('chapter_name')
                concept_name = meta.get('concept_name')
                difficulty = meta.get('difficulty_level') or meta.get('difficulty')

                now_iso_local = datetime.now().isoformat()

                if subject_name:
                    client = _get_supabase_client()

                    # Try to find existing subject for this student
                    existing = client.table('subjects').select('*').eq('student_id', user_id).eq('llm_suggested_subject_name', subject_name).execute()
                    subject_id_val = None
                    if existing and getattr(existing, 'data', None):
                        rows = existing.data if isinstance(existing.data, list) else [existing.data]
                        if len(rows) > 0:
                            subject_id_val = rows[0].get('subject_id')
                            logger.info('Found existing subject id=%s for student=%s name=%s', subject_id_val, user_id, subject_name)
                    if not subject_id_val:
                        ins = {
                            'student_id': user_id,
                            'llm_suggested_subject_name': subject_name,
                            'created_at': now_iso_local,
                            'updated_at': now_iso_local
                        }
                        resp = client.table('subjects').insert(ins).execute()
                        if resp and getattr(resp, 'data', None):
                            inserted = resp.data[0] if isinstance(resp.data, list) else resp.data
                            subject_id_val = inserted.get('subject_id')
                            logger.info('Inserted subject id=%s for student=%s name=%s', subject_id_val, user_id, subject_name)

                    # Chapters
                    chapter_id_val = None
                    if subject_id_val and chapter_name:
                        existing = client.table('chapters').select('*').eq('student_id', user_id).eq('subject_id', subject_id_val).eq('llm_suggested_chapter_name', chapter_name).execute()
                        if existing and getattr(existing, 'data', None):
                            rows = existing.data if isinstance(existing.data, list) else [existing.data]
                            if len(rows) > 0:
                                chapter_id_val = rows[0].get('chapter_id')
                                logger.info('Found existing chapter id=%s for subject_id=%s name=%s', chapter_id_val, subject_id_val, chapter_name)
                        if not chapter_id_val:
                            ins = {
                                'student_id': user_id,
                                'subject_id': subject_id_val,
                                'llm_suggested_chapter_name': chapter_name,
                                'chapter_order': 0,
                                'description': None,
                                'created_at': now_iso_local,
                                'updated_at': now_iso_local
                            }
                            resp = client.table('chapters').insert(ins).execute()
                            if resp and getattr(resp, 'data', None):
                                inserted = resp.data[0] if isinstance(resp.data, list) else resp.data
                                chapter_id_val = inserted.get('chapter_id')
                                logger.info('Inserted chapter id=%s for subject_id=%s name=%s', chapter_id_val, subject_id_val, chapter_name)

                    # Concepts
                    if chapter_id_val and concept_name:
                        existing = client.table('concepts').select('*').eq('student_id', user_id).eq('chapter_id', chapter_id_val).eq('llm_suggested_concept_name', concept_name).execute()
                        if existing and getattr(existing, 'data', None):
                            rows = existing.data if isinstance(existing.data, list) else [existing.data]
                            if len(rows) > 0:
                                logger.info('Found existing concept for chapter_id=%s name=%s', chapter_id_val, concept_name)
                        else:
                            ins = {
                                'student_id': user_id,
                                'chapter_id': chapter_id_val,
                                'llm_suggested_concept_name': concept_name,
                                'concept_order': 0,
                                'description': None,
                                'difficulty_level': difficulty or 'Medium',
                                'created_at': now_iso_local,
                                'updated_at': now_iso_local
                            }
                            resp = client.table('concepts').insert(ins).execute()
                            if resp and getattr(resp, 'data', None):
                                logger.info('Inserted concept for chapter_id=%s name=%s', chapter_id_val, concept_name)
        except Exception as e:
            logger.error('Failed to persist LLM metadata (subjects/chapters/concepts): %s', e)

        # Return the generated response along with the session_id used/created so clients can persist it
        out_session_id = created_session_id or session_id

        return {
            'session_id': out_session_id,
            'llm_response': response
        } if isinstance(response, dict) else (response or {})
    except ValueError as ve:
        logger.error("Validation error in get_llm_response: %s", ve)
        return {
            "status": "error",
            "error": str(ve),
            "error_type": "validation_error"
        }
    except Exception as e:
        logger.exception("Error in get_llm_response: %s", e)
        return {
            "status": "error",
            "error": str(e),
            "error_type": "processing_error"
        }
