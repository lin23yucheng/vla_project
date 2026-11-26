"""Audit locally copied raw MCAP episodes."""

import json
import os
from pathlib import Path

import allure

from common.mcap_stability import audit_mcap_directory
from common.mcap_stability import render_mcap_report


DEFAULT_MCAP_ROOT = Path(r"C:\Users\admin\Desktop\mcap文件")


def test_raw_mcap_stability():
    mcap_root = Path(os.environ.get("VLA_MCAP_ROOT", str(DEFAULT_MCAP_ROOT)))
    result = audit_mcap_directory(mcap_root)
    summary_report = render_mcap_report(result)
    raw_report = json.dumps(result, ensure_ascii=False, indent=2)
    allure.attach(
        summary_report,
        name="原始 MCAP 审计摘要",
        attachment_type=allure.attachment_type.TEXT,
    )
    allure.attach(
        raw_report,
        name="原始 MCAP 审计明细",
        attachment_type=allure.attachment_type.JSON,
    )
    print(summary_report)

    assert result["is_valid"], "原始 MCAP 审计发现错误，详见 Allure 附件和控制台报告"
