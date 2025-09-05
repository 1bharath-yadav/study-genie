"""
Supabase client configuration and utilities for StudyGenie
"""
import os
import logging
from typing import Optional, Dict, Any, List
from supabase import create_client, Client
from postgrest import APIError
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("supabase_client")


class SupabaseClient:
    """Supabase client wrapper for StudyGenie"""

    def __init__(self):
        self.url: str = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
        self.key: str = os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY")
        self.service_role_key: str = os.getenv(
            "SUPABASE_SERVICE_ROLE_KEY")  # For admin operations

        if not self.url or not self.key:
            raise ValueError(
                "Supabase URL and anon key must be provided in environment variables")

        # Create client instances
        self.client: Client = create_client(self.url, self.key)
        self.admin_client: Optional[Client] = None

        # Create admin client if service role key is available
        if self.service_role_key:
            self.admin_client = create_client(self.url, self.service_role_key)

        logger.info("Supabase client initialized successfully")

    async def test_connection(self) -> bool:
        """Test the Supabase connection"""
        try:
            # Try to fetch from a system table
            response = self.client.table(
                "students").select("*").limit(1).execute()
            logger.info("Supabase connection test successful")
            return True
        except Exception as e:
            logger.error(f"Supabase connection test failed: {e}")
            return False

    # Student operations
    async def create_student(self, student_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new student record"""
        try:
            response = self.client.table(
                "students").insert(student_data).execute()
            if response.data:
                logger.info(
                    f"Student created successfully: {student_data.get('student_id')}")
                return response.data[0]
            else:
                raise Exception("No data returned from insert operation")
        except APIError as e:
            logger.error(f"Error creating student: {e}")
            raise

    async def get_student(self, student_id: str) -> Optional[Dict[str, Any]]:
        """Get student by ID"""
        try:
            response = self.client.table("students").select(
                "*").eq("student_id", student_id).execute()
            if response.data:
                return response.data[0]
            return None
        except APIError as e:
            logger.error(f"Error fetching student {student_id}: {e}")
            raise

    async def update_student(self, student_id: str, update_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update student record"""
        try:
            response = self.client.table("students").update(
                update_data).eq("student_id", student_id).execute()
            if response.data:
                return response.data[0]
            else:
                raise Exception("No data returned from update operation")
        except APIError as e:
            logger.error(f"Error updating student {student_id}: {e}")
            raise

    async def delete_student(self, student_id: str) -> bool:
        """Delete student record"""
        try:
            response = self.client.table("students").delete().eq(
                "student_id", student_id).execute()
            return True
        except APIError as e:
            logger.error(f"Error deleting student {student_id}: {e}")
            raise

    # Progress tracking operations
    async def get_student_progress(self, student_id: str) -> List[Dict[str, Any]]:
        """Get student's learning progress"""
        try:
            response = self.client.table("student_concept_progress")\
                .select("*, concepts(concept_name, chapters(chapter_name, subjects(subject_name)))")\
                .eq("student_id", student_id)\
                .execute()
            return response.data
        except APIError as e:
            logger.error(
                f"Error fetching progress for student {student_id}: {e}")
            raise

    async def update_concept_progress(self, progress_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update concept progress"""
        try:
            # Use upsert to handle both insert and update
            response = self.client.table("student_concept_progress")\
                .upsert(progress_data, on_conflict="student_id,concept_id")\
                .execute()
            if response.data:
                return response.data[0]
            else:
                raise Exception("No data returned from upsert operation")
        except APIError as e:
            logger.error(f"Error updating concept progress: {e}")
            raise

    # API Key management
    async def store_api_key(self, student_id: str, api_key: str, service: str = "gemini") -> bool:
        """Store encrypted API key for student"""
        try:
            from app.core.encryption import encrypt_api_key

            encrypted_key = encrypt_api_key(api_key)

            response = self.client.table("student_api_keys")\
                .upsert({
                    "student_id": student_id,
                    "encrypted_api_key": encrypted_key,
                    "service": service,
                    "updated_at": "now()"
                }, on_conflict="student_id")\
                .execute()
            logger.info(
                f"API key stored successfully for student {student_id}")
            return True
        except APIError as e:
            logger.error(
                f"Error storing API key for student {student_id}: {e}")
            raise
        except Exception as e:
            logger.error(
                f"Error encrypting/storing API key for student {student_id}: {e}")
            raise

    async def get_api_key(self, student_id: str) -> Optional[str]:
        """Get decrypted API key for student"""
        try:
            response = self.client.table("student_api_keys")\
                .select("encrypted_api_key, service")\
                .eq("student_id", student_id)\
                .execute()
            if response.data:
                from app.core.encryption import decrypt_api_key
                encrypted_key = response.data[0]["encrypted_api_key"]
                decrypted_key = decrypt_api_key(encrypted_key)
                # Clean the API key - remove any whitespace or newlines
                clean_key = decrypted_key.strip()
                # Debug - log length and format info
                print(
                    f"Debug - Retrieved API key for {student_id}: length={len(clean_key)}, starts_with={clean_key[:10]}..., ends_with=...{clean_key[-5:]}")
                print(
                    f"Debug - Key format check: has_newlines={bool(chr(10) in clean_key or chr(13) in clean_key)}, has_spaces={bool(' ' in clean_key)}")
                return clean_key
            return None
        except APIError as e:
            logger.error(
                f"Error fetching API key for student {student_id}: {e}")
            raise
        except Exception as e:
            logger.error(
                f"Error decrypting API key for student {student_id}: {e}")
            raise

    async def get_api_key_info(self, student_id: str) -> Optional[Dict[str, Any]]:
        """Get API key info (without decrypted key) for student"""
        try:
            response = self.client.table("student_api_keys")\
                .select("service, created_at, updated_at")\
                .eq("student_id", student_id)\
                .execute()
            if response.data:
                return response.data[0]
            return None
        except APIError as e:
            logger.error(
                f"Error fetching API key info for student {student_id}: {e}")
            raise

    async def delete_api_key(self, student_id: str) -> bool:
        """Delete API key for student"""
        try:
            response = self.client.table("student_api_keys")\
                .delete()\
                .eq("student_id", student_id)\
                .execute()
            logger.info(
                f"API key deleted successfully for student {student_id}")
            return True
        except APIError as e:
            logger.error(
                f"Error deleting API key for student {student_id}: {e}")
            raise

    # Learning activities
    async def record_learning_activity(self, activity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Record a learning activity"""
        try:
            response = self.client.table(
                "learning_activities").insert(activity_data).execute()
            if response.data:
                return response.data[0]
            else:
                raise Exception("No data returned from insert operation")
        except APIError as e:
            logger.error(f"Error recording learning activity: {e}")
            raise

    # Recommendations
    async def get_recommendations(self, student_id: str, active_only: bool = True) -> List[Dict[str, Any]]:
        """Get recommendations for student"""
        try:
            logger.info(
                f"Querying recommendations table for student: {student_id}")

            query = self.client.table("recommendations")\
                .select("*, concepts(concept_name, chapters(chapter_name, subjects(subject_name)))")\
                .eq("student_id", student_id)

            if active_only:
                query = query.eq("is_active", True).eq("is_completed", False)

            response = query.order("priority_score", desc=True).execute()

            # Handle the case where response.data is None
            recommendations = response.data if response.data else []
            logger.info(
                f"Found {len(recommendations)} recommendations for student {student_id}")

            return recommendations

        except APIError as e:
            logger.error(
                f"Supabase API error fetching recommendations for student {student_id}: {e}")
            # Return empty list instead of raising exception
            return []
        except Exception as e:
            logger.error(
                f"Unexpected error fetching recommendations for student {student_id}: {e}")
            return []

    async def save_recommendations(self, student_id: str, recommendations: List[Dict[str, Any]]) -> bool:
        """Save recommendations for student"""
        try:
            logger.info(
                f"Saving {len(recommendations)} recommendations for student {student_id}")

            # First ensure the student exists and resolve the correct student_id
            resolved_student_id = await self._resolve_student_id(student_id)
            if not resolved_student_id:
                logger.error(f"Could not resolve student_id for {student_id}")
                return False

            # First deactivate old recommendations
            try:
                self.client.table("recommendations")\
                    .update({"is_active": False})\
                    .eq("student_id", resolved_student_id)\
                    .execute()
                logger.info(
                    f"Deactivated old recommendations for student {resolved_student_id}")
            except Exception as e:
                logger.warning(
                    f"Could not deactivate old recommendations for {resolved_student_id}: {e}")

            # Insert new recommendations with resolved student_id
            if recommendations:
                for rec in recommendations:
                    rec["student_id"] = resolved_student_id
                    # Ensure required fields have default values
                    rec.setdefault("is_active", True)
                    rec.setdefault("is_completed", False)
                    rec.setdefault("priority_score", 5)

                try:
                    response = self.client.table("recommendations").insert(
                        recommendations).execute()
                    logger.info(
                        f"Successfully inserted {len(recommendations)} recommendations for student {resolved_student_id}")
                except Exception as e:
                    logger.error(
                        f"Failed to insert recommendations for {resolved_student_id}: {e}")
                    return False

            return True

        except APIError as e:
            logger.error(
                f"Supabase API error saving recommendations for student {student_id}: {e}")
            return False
        except Exception as e:
            logger.error(
                f"Unexpected error saving recommendations for student {student_id}: {e}")
            return False

    async def ensure_student_exists(self, student_id: str) -> bool:
        """Ensure student exists in database, create if not found"""
        try:
            # Check if student exists
            student_check = self.client.table("students")\
                .select("student_id")\
                .eq("student_id", student_id)\
                .execute()

            if not student_check.data:
                logger.info(f"Student {student_id} not found, creating...")
                # Extract email and username from student_id
                email = student_id
                username = email.split('@')[0] if '@' in email else student_id
                full_name = username.title()

                student_data = {
                    "student_id": student_id,
                    "email": email,
                    "username": username,
                    "full_name": full_name,
                    "learning_preferences": {}
                }

                self.client.table("students").insert(student_data).execute()
                logger.info(
                    f"Successfully created student record for {student_id}")
            else:
                logger.info(f"Student {student_id} already exists")

            return True

        except Exception as e:
            logger.error(
                f"Failed to ensure student exists for {student_id}: {e}")
            return False

    async def _resolve_student_id(self, student_id: str) -> Optional[str]:
        """Resolve student_id to the correct format used in the database"""
        try:
            # If it's an email, try to find the student by email first
            if '@' in student_id:
                email_check = self.client.table("students")\
                    .select("student_id")\
                    .eq("email", student_id)\
                    .execute()

                if email_check.data:
                    return email_check.data[0]['student_id']

                # If not found by email, try to find by student_id
                id_check = self.client.table("students")\
                    .select("student_id")\
                    .eq("student_id", student_id)\
                    .execute()

                if id_check.data:
                    return id_check.data[0]['student_id']

                # Create the student if not found
                if await self.ensure_student_exists(student_id):
                    return student_id
            else:
                # It's a numeric ID, check if it exists
                id_check = self.client.table("students")\
                    .select("student_id")\
                    .eq("student_id", student_id)\
                    .execute()

                if id_check.data:
                    return id_check.data[0]['student_id']

                # Create the student if not found
                if await self.ensure_student_exists(student_id):
                    return student_id

            return None

        except Exception as e:
            logger.error(f"Error resolving student_id for {student_id}: {e}")
            return None

    # Subject management
    async def create_or_get_subject(self, subject_name: str, description: Optional[str] = None) -> int:
        """Create a new subject or get existing one"""
        try:
            # Try to get existing subject first
            response = self.client.table("subjects")\
                .select("subject_id")\
                .eq("subject_name", subject_name)\
                .execute()

            if response.data:
                return response.data[0]["subject_id"]

            # Create new subject
            response = self.client.table("subjects")\
                .insert({"subject_name": subject_name, "description": description})\
                .execute()

            if response.data:
                return response.data[0]["subject_id"]
            else:
                raise Exception("No data returned from subject creation")
        except APIError as e:
            logger.error(f"Error creating/getting subject {subject_name}: {e}")
            raise

    async def create_or_get_chapter(self, subject_id: int, chapter_name: str, description: Optional[str] = None) -> int:
        """Create a new chapter or get existing one"""
        try:
            # Try to get existing chapter first
            response = self.client.table("chapters")\
                .select("chapter_id")\
                .eq("subject_id", subject_id)\
                .eq("chapter_name", chapter_name)\
                .execute()

            if response.data:
                return response.data[0]["chapter_id"]

            # Get next chapter order
            order_response = self.client.table("chapters")\
                .select("chapter_order")\
                .eq("subject_id", subject_id)\
                .order("chapter_order", desc=True)\
                .limit(1)\
                .execute()

            next_order = 1
            if order_response.data:
                next_order = order_response.data[0]["chapter_order"] + 1

            # Create new chapter
            response = self.client.table("chapters")\
                .insert({
                    "subject_id": subject_id,
                    "chapter_name": chapter_name,
                    "chapter_order": next_order,
                    "description": description
                })\
                .execute()

            if response.data:
                return response.data[0]["chapter_id"]
            else:
                raise Exception("No data returned from chapter creation")
        except APIError as e:
            logger.error(f"Error creating/getting chapter {chapter_name}: {e}")
            raise

    async def create_or_get_concept(self, chapter_id: int, concept_name: str,
                                    difficulty_level: str = "Medium", description: Optional[str] = None) -> int:
        """Create a new concept or get existing one"""
        try:
            # Try to get existing concept first
            response = self.client.table("concepts")\
                .select("concept_id")\
                .eq("chapter_id", chapter_id)\
                .eq("concept_name", concept_name)\
                .execute()

            if response.data:
                return response.data[0]["concept_id"]

            # Get next concept order
            order_response = self.client.table("concepts")\
                .select("concept_order")\
                .eq("chapter_id", chapter_id)\
                .order("concept_order", desc=True)\
                .limit(1)\
                .execute()

            next_order = 1
            if order_response.data:
                next_order = order_response.data[0]["concept_order"] + 1

            # Create new concept
            response = self.client.table("concepts")\
                .insert({
                    "chapter_id": chapter_id,
                    "concept_name": concept_name,
                    "concept_order": next_order,
                    "difficulty_level": difficulty_level,
                    "description": description
                })\
                .execute()

            if response.data:
                return response.data[0]["concept_id"]
            else:
                raise Exception("No data returned from concept creation")
        except APIError as e:
            logger.error(f"Error creating/getting concept {concept_name}: {e}")
            raise


# Global Supabase client instance
supabase_client: Optional[SupabaseClient] = None


def get_supabase_client() -> SupabaseClient:
    """Get or create Supabase client instance"""
    global supabase_client
    if supabase_client is None:
        supabase_client = SupabaseClient()
    return supabase_client


async def initialize_supabase() -> SupabaseClient:
    """Initialize Supabase client and test connection"""
    client = get_supabase_client()

    # Test connection
    is_connected = await client.test_connection()
    if not is_connected:
        raise Exception("Failed to connect to Supabase")

    logger.info("Supabase initialized successfully")
    return client
