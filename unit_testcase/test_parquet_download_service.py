from pathlib import Path
from unittest.mock import Mock, patch

import requests

from common.parquet_download_service import (
    download_converted_parquet,
    load_s3_login_config,
    login_s3_console,
)


def _write_config(project_root: Path, console_endpoint: str = "") -> None:
    config_dir = project_root / "config"
    config_dir.mkdir()
    console_line = f"s3_console_endpoint = {console_endpoint}\n" if console_endpoint else ""
    (config_dir / "env_config.ini").write_text(
        "[environment]\nexecution_env = fat\n\n"
        "[fat-vla]\n"
        "s3_endpoint = http://minio.internal:9000\n"
        f"{console_line}"
        "s3_access_key = access\n"
        "s3_secret_key = secret\n",
        encoding="utf-8",
    )


def test_load_s3_login_config_derives_console_endpoint(tmp_path: Path) -> None:
    _write_config(tmp_path)

    assert load_s3_login_config(tmp_path) == ("http://minio.internal:9001", "access", "secret")


def test_login_s3_console_accepts_json_token(tmp_path: Path) -> None:
    _write_config(tmp_path)
    session = requests.Session()
    response = Mock(status_code=200)
    response.json.return_value = {"token": "test-token"}
    session.post = Mock(return_value=response)

    login_s3_console(session, tmp_path)

    assert session.cookies.get("token") == "test-token"
    assert session.post.call_args.kwargs["json"] == {"accessKey": "access", "secretKey": "secret"}


def test_login_s3_console_uses_token_set_by_cookie(tmp_path: Path) -> None:
    _write_config(tmp_path)
    session = requests.Session()
    response = Mock(status_code=200)

    def post(*args, **kwargs):
        session.cookies.set("token", "cookie-token")
        return response

    session.post = Mock(side_effect=post)
    login_s3_console(session, tmp_path)

    assert session.cookies.get("token") == "cookie-token"
    response.json.assert_not_called()


def test_download_automatically_logs_in_when_no_cookie_file(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    response = Mock(status_code=200, content=b"parquet-content")
    session = Mock()
    session.get.return_value = response

    with patch("common.parquet_download_service.requests.Session", return_value=session), patch(
        "common.parquet_download_service.login_s3_console"
    ) as login:
        result = download_converted_parquet("TASK-1", "v2", output_dir=output_dir)

    login.assert_called_once()
    session.get.assert_called_once()
    assert result.read_bytes() == b"parquet-content"


def test_download_retries_after_unauthorized_response(tmp_path: Path) -> None:
    unauthorized = Mock(status_code=401)
    response = Mock(status_code=200, content=b"parquet-content")
    session = Mock()
    session.get.side_effect = [unauthorized, response]

    with patch("common.parquet_download_service.requests.Session", return_value=session), patch(
        "common.parquet_download_service.login_s3_console"
    ) as login:
        download_converted_parquet("TASK-1", "v2", output_dir=tmp_path / "output")

    assert login.call_count == 2
    assert session.get.call_count == 2
