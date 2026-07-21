"""S3 parquet 下载公共能力。"""

from __future__ import annotations

import argparse
import configparser
from pathlib import Path

import requests

S3_DOWNLOAD_URL_TEMPLATE = (
    "http://172.16.166.105:9001/api/v1/buckets/vla-label-studio/objects/download"
    "?prefix={task_no}%2Ffinal-data%2F{folder}%2Fdata%2Fchunk-000%2Fepisode_000000.parquet"
)


def normalize_folder(folder: str) -> str:
    folder_normalized = str(folder).strip().lower().replace(" ", "")
    if folder_normalized in {"v2", "2", "lerobot_v2"}:
        return "lerobot_v2"
    if folder_normalized in {"v3", "3", "lerobot_v3"}:
        return "lerobot_v3"
    return folder_normalized


def load_task_no_from_env_config(project_root: Path | None = None) -> str:
    project_root = project_root or Path(__file__).resolve().parent.parent
    config_path = project_root / "config" / "env_config.ini"

    config = configparser.ConfigParser()
    if not config.read(config_path, encoding="utf-8"):
        raise FileNotFoundError(f"未找到配置文件: {config_path}")

    execution_env = config.get("environment", "execution_env", fallback="").strip().lower()
    if execution_env not in {"dev", "fat", "prod"}:
        raise ValueError(f"execution_env 配置错误: {execution_env}，仅支持 dev、fat 或 prod")

    section = f"{execution_env}-vla"
    if not config.has_section(section):
        raise ValueError(f"配置文件缺少节: [{section}]")

    task_no = config.get(section, "Task_no", fallback="").strip()
    if not task_no:
        raise ValueError(f"配置文件节 [{section}] 中缺少 Task_no")

    return task_no


def load_cookie(cookie_file_path: Path) -> str:
    if not cookie_file_path.exists():
        raise FileNotFoundError(f"未找到 cookie 文件: {cookie_file_path}")

    cookie_value = cookie_file_path.read_text(encoding="utf-8").strip()
    if cookie_value.lower().startswith("cookie:"):
        cookie_value = cookie_value.split(":", 1)[1].strip()

    if not cookie_value:
        raise ValueError(f"cookie 文件为空: {cookie_file_path}")

    return cookie_value


def build_parquet_download_url(task_no: str, folder: str) -> str:
    """按固定模板替换 task_no 和 folder，不改动 URL 编码格式。"""
    normalized_folder = normalize_folder(folder)
    return S3_DOWNLOAD_URL_TEMPLATE.format(task_no=task_no, folder=normalized_folder)


def download_converted_parquet(
    task_no: str,
    folder: str,
    output_dir: Path | None = None,
    cookie_file_path: Path | None = None,
    timeout: int = 120,
) -> Path:
    """清空 parquet_file 后下载指定任务对应的 parquet。"""
    project_root = Path(__file__).resolve().parent.parent
    resolved_output_dir = output_dir or (project_root / "parquet_file")
    resolved_cookie_file = cookie_file_path or (project_root / "config" / "cookie.txt")

    normalized_folder = normalize_folder(folder)
    download_url = build_parquet_download_url(task_no, normalized_folder)
    print(f"[下载准备] task_no={task_no}, folder={normalized_folder}")
    print(f"[下载准备] 目标下载地址: {download_url}")

    cookie_value = load_cookie(resolved_cookie_file)

    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    for old_file in resolved_output_dir.iterdir():
        if old_file.is_file():
            old_file.unlink()

    response = requests.get(
        download_url,
        headers={"Cookie": cookie_value},
        timeout=timeout,
    )
    if response.status_code != 200:
        response_text = response.text[:300]
        raise RuntimeError(f"下载 parquet 失败: status={response.status_code}, body={response_text}")
    response.raise_for_status()

    file_name = f"{task_no}_{normalized_folder}.parquet"
    target_file = resolved_output_dir / file_name
    target_file.write_bytes(response.content)
    print(f"[下载完成] parquet 已保存: {target_file}")
    return target_file


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="下载转换后的 parquet 文件")
    parser.add_argument("--task-no", dest="task_no", default=None, help="任务编号；默认从 env_config.ini 读取")
    parser.add_argument("--folder", dest="folder", required=True, help="转换目标目录名，例如 lerobot_v2")
    parser.add_argument("--output-dir", dest="output_dir", default=None, help="输出目录，默认项目根目录/parquet_file")
    parser.add_argument("--cookie-file", dest="cookie_file", default=None, help="cookie 文件路径，默认项目根目录/config/cookie.txt")
    parser.add_argument("--timeout", dest="timeout", type=int, default=120, help="请求超时时间，默认120秒")
    return parser


def main(
    folder: str,
    task_no: str | None = None,
    output_dir: Path | None = None,
    cookie_file_path: Path | None = None,
    timeout: int = 120,
) -> Path:
    """脚本主入口：folder 必填，task_no 为空时自动读取配置文件。"""
    resolved_task_no = task_no or load_task_no_from_env_config()
    downloaded_file = download_converted_parquet(
        task_no=resolved_task_no,
        folder=folder,
        output_dir=output_dir,
        cookie_file_path=cookie_file_path,
        timeout=timeout,
    )
    print(f"下载完成: {downloaded_file}")
    return downloaded_file


def main_from_cli() -> None:
    """命令行入口，兼容 --task-no/--folder 等参数。"""
    parser = _build_cli_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else None
    cookie_file = Path(args.cookie_file) if args.cookie_file else None
    main(
        folder=args.folder,
        task_no=args.task_no,
        output_dir=output_dir,
        cookie_file_path=cookie_file,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    folder = "lerobot_v2"
    main(folder=folder)


