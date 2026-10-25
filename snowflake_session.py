import os

from snowflake.snowpark import Session


def create_local_session() -> Session:
    required_vars = [
        "SNOWFLAKE_ACCOUNT",
        "SNOWFLAKE_USER",
        "SNOWFLAKE_PAT",
        "SNOWFLAKE_WAREHOUSE",
        "SNOWFLAKE_DATABASE",
        "SNOWFLAKE_SCHEMA",
    ]

    missing = [
        name
        for name in required_vars
        if not os.getenv(name)
    ]

    if missing:
        raise RuntimeError(
            "Missing required environment variables: "
            + ", ".join(missing)
        )

    connection_parameters = {
        "account": os.environ["SNOWFLAKE_ACCOUNT"].strip(),
        "user": os.environ["SNOWFLAKE_USER"].strip(),
        "password": os.environ["SNOWFLAKE_PAT"].strip(),
        "role": os.getenv(
            "SNOWFLAKE_ROLE",
            "DATA_SCIENTIST",
        ).strip(),
        "warehouse": os.environ["SNOWFLAKE_WAREHOUSE"].strip(),
        "database": os.environ["SNOWFLAKE_DATABASE"].strip(),
        "schema": os.environ["SNOWFLAKE_SCHEMA"].strip(),
    }

    return Session.builder.configs(
        connection_parameters
    ).create()