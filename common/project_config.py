"""Load public defaults, then local private settings and environment overrides."""

import configparser
import os
from pathlib import Path


def load_project_config(project_root: Path | None = None) -> configparser.ConfigParser:
    project_root = project_root or Path(__file__).resolve().parent.parent
    config = configparser.ConfigParser(interpolation=None)
    paths = [project_root / "config" / "env_config.ini",
             project_root / "config" / "env_config.local.ini"]
    if not config.read(paths, encoding="utf-8"):
        raise FileNotFoundError("Missing config/env_config.ini; see config/README.md")
    if not config.has_section("environment"):
        config.add_section("environment")
    env = os.environ.get("VLA_EXECUTION_ENV", config.get("environment", "execution_env", fallback="fat")).strip().lower()
    if env not in {"dev", "fat", "prod"}:
        raise ValueError("VLA_EXECUTION_ENV must be dev, fat or prod")
    config.set("environment", "execution_env", env)
    section = f"{env}-vla"
    if not config.has_section(section):
        config.add_section(section)
    for key in ("api_url", "token_url", "username", "password", "s3_endpoint",
                "s3_console_endpoint", "s3_access_key", "s3_secret_key", "s3_bucket",
                "db_host", "db_port", "db_user", "db_password", "db_name"):
        value = os.environ.get(f"VLA_{key.upper()}")
        if value is not None:
            config.set(section, key, value)
    return config
