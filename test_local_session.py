from pathlib import Path

from dotenv import load_dotenv
from snowflake_session import create_local_session

load_dotenv(
    Path(__file__).resolve().parent / ".env",
    override=True,
)

session = create_local_session()

try:
    result = session.sql(
        """
        SELECT
            CURRENT_USER() AS current_user,
            CURRENT_ROLE() AS current_role,
            CURRENT_WAREHOUSE() AS current_warehouse,
            CURRENT_DATABASE() AS current_database,
            CURRENT_SCHEMA() AS current_schema
        """
    ).collect()

    print(result)
finally:
    session.close()