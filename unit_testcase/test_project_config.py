from common.project_config import load_project_config


def test_local_config_and_environment_override_public_template(tmp_path, monkeypatch):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "env_config.ini").write_text(
        "[environment]\nexecution_env = fat\n[fat-vla]\npassword =\n",
        encoding="utf-8",
    )
    (config_dir / "env_config.local.ini").write_text(
        "[fat-vla]\npassword = local%password\ndb_name = test-db\n",
        encoding="utf-8",
    )
    config = load_project_config(tmp_path)
    assert config.get("fat-vla", "password") == "local%password"
    monkeypatch.setenv("VLA_PASSWORD", "environment%password")
    config = load_project_config(tmp_path)
    assert config.get("fat-vla", "password") == "environment%password"
    assert config.get("fat-vla", "db_name") == "test-db"


def test_environment_switch_uses_matching_connection_settings(tmp_path, monkeypatch):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "env_config.ini").write_text(
        "[environment]\nexecution_env = fat\n[fat-vla]\ndb_host = fat.example.com\n"
        "[dev-vla]\ndb_host = dev.example.com\n", encoding="utf-8",
    )
    monkeypatch.setenv("VLA_EXECUTION_ENV", "dev")
    monkeypatch.setenv("VLA_DB_USER", "test-user")
    config = load_project_config(tmp_path)
    assert config.get("environment", "execution_env") == "dev"
    assert config.get("dev-vla", "db_host") == "dev.example.com"
    assert config.get("dev-vla", "db_user") == "test-user"
