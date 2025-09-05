"""
Database initialization script for StudyGenie
This script can be run independently to set up the database tables
"""
from models import initialize_database, cleanup_database, migrate_existing_data, health_check
import asyncio
import logging
import sys
from pathlib import Path

# Add the app directory to the Python path
app_dir = Path(__file__).parent.parent
sys.path.insert(0, str(app_dir))


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def init_database():
    """Initialize the database with all tables and indexes"""
    logger.info("Starting database initialization...")

    try:
        # Initialize database
        db_manager = await initialize_database()
        logger.info("✅ Database connection and tables created successfully")

        # Run migrations if needed
        logger.info("Running database migrations...")
        await migrate_existing_data(db_manager)
        logger.info("✅ Database migrations completed")

        # Run health check
        logger.info("Running health check...")
        health_status = await health_check(db_manager)
        logger.info(f"✅ Health check completed: {health_status['status']}")

        # Display table statistics
        if health_status.get('table_stats'):
            logger.info("📊 Table Statistics:")
            for table, count in health_status['table_stats'].items():
                logger.info(f"  - {table}: {count} records")

        # Close connections
        await db_manager.close_pool()
        logger.info("✅ Database initialization completed successfully")

    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        raise


async def cleanup_old_data():
    """Clean up old data from the database"""
    logger.info("Starting database cleanup...")

    try:
        db_manager = await initialize_database()
        await cleanup_database(db_manager)
        await db_manager.close_pool()
        logger.info("✅ Database cleanup completed successfully")

    except Exception as e:
        logger.error(f"❌ Database cleanup failed: {e}")
        raise


async def reset_database():
    """Reset the database (drop and recreate all tables)"""
    logger.warning("🚨 CAUTION: This will drop all existing tables and data!")

    response = input(
        "Are you sure you want to reset the database? Type 'yes' to confirm: ")
    if response.lower() != 'yes':
        logger.info("Database reset cancelled")
        return

    logger.info("Starting database reset...")

    try:
        from models import DatabaseManager
        import os
        import asyncpg

        # Initialize connection
        DATABASE_URL = os.getenv("DATABASE_URL")
        if not DATABASE_URL:
            raise ValueError("DATABASE_URL environment variable not set")

        # Create temporary connection to drop tables
        conn = await asyncpg.connect(DATABASE_URL)

        # Drop all tables in reverse order to handle foreign keys
        tables_to_drop = [
            'student_recommendations',
            'study_sessions',
            'recommendations',
            'learning_activities',
            'student_weaknesses',
            'student_concept_progress',
            'student_subjects',
            'concepts',
            'chapters',
            'subjects',
            'students'
        ]

        logger.info("Dropping existing tables...")
        for table in tables_to_drop:
            try:
                await conn.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
                logger.info(f"  - Dropped table: {table}")
            except Exception as e:
                logger.warning(f"  - Warning dropping table {table}: {e}")

        await conn.close()
        logger.info("✅ All tables dropped successfully")

        # Now recreate everything
        await init_database()

    except Exception as e:
        logger.error(f"❌ Database reset failed: {e}")
        raise


async def check_database_status():
    """Check the current database status"""
    logger.info("Checking database status...")

    try:
        db_manager = await initialize_database()
        health_status = await health_check(db_manager)

        print("\n" + "="*50)
        print("📊 DATABASE STATUS REPORT")
        print("="*50)
        print(f"Status: {health_status['status'].upper()}")
        print(f"Connection: {health_status.get('connection', 'unknown')}")
        print(
            f"Database Size: {health_status.get('database_size', 'unknown')}")
        print(f"Timestamp: {health_status.get('timestamp', 'unknown')}")

        if health_status.get('table_stats'):
            print("\n📋 TABLE STATISTICS:")
            print("-" * 30)
            total_records = 0
            for table, count in health_status['table_stats'].items():
                print(f"{table:<25}: {count:>8} records")
                total_records += count
            print("-" * 30)
            print(f"{'TOTAL':<25}: {total_records:>8} records")

        print("\n" + "="*50)

        await db_manager.close_pool()

    except Exception as e:
        logger.error(f"❌ Database status check failed: {e}")
        raise


def main():
    """Main CLI interface"""
    import argparse

    parser = argparse.ArgumentParser(
        description="StudyGenie Database Management")
    parser.add_argument(
        'action',
        choices=['init', 'status', 'cleanup', 'reset'],
        help='Action to perform'
    )

    args = parser.parse_args()

    if args.action == 'init':
        asyncio.run(init_database())
    elif args.action == 'status':
        asyncio.run(check_database_status())
    elif args.action == 'cleanup':
        asyncio.run(cleanup_old_data())
    elif args.action == 'reset':
        asyncio.run(reset_database())


if __name__ == "__main__":
    main()
