"""Audit locally copied raw MCAP episodes."""

import json
import os
from pathlib import Path

import allure

from common.mcap_stability import audit_mcap_directory


DEFAULT_MCAP_ROOT = Path(r"C:\Users\admin\Desktop\mcap文件")


def test_raw_mcap_stability():
    mcap_root = Path(os.environ.get("VLA_MCAP_ROOT", str(DEFAULT_MCAP_ROOT)))
    result = audit_mcap_directory(mcap_root)
    report = json.dumps(result, ensure_ascii=False, indent=2)
    allure.attach(report, name="原始 MCAP 审计结果", attachment_type=allure.attachment_type.JSON)
    print(f"MCAP 目录: {result['root']}")
    print(f"机器人配置: {result['robot_config']}")
    print(f"文件数: {result['file_count']}，通过: {result['valid_file_count']}")
    for file_result in result["files"]:
        print(f"\nEpisode {file_result.get('episode')}：{file_result['file']}")
        for warning in file_result.get("warnings", []):
            print(
                f"  提示: {warning}"
            )
        for topic in file_result.get("camera_topics", []):
            camera = file_result["topics"][topic]
            timing = camera["log_time"]
            longest_gap = timing["interval_max_ns"]
            longest_gap_ms = (
                longest_gap / 1_000_000 if longest_gap is not None else float("nan")
            )
            rate_hz = timing["observed_rate_hz"]
            rate_text = f"{rate_hz:.3f}" if rate_hz is not None else "N/A"
            print(
                f"  {camera['role'] or topic} | {topic} | "
                f"帧数={camera['message_count']} | "
                f"尺寸={camera['image_dimensions']} | "
                f"观测帧率={rate_text}fps | "
                f"平均间隔={timing['interval_mean_ns'] / 1_000_000:.3f}ms | "
                f"P95={timing['interval_p95_ns'] / 1_000_000:.3f}ms | "
                f"最大间隔={longest_gap_ms:.3f}ms | "
                f"解码失败={camera['image_decode_failures']}"
            )
            for window in camera["sixty_second_trend"]:
                p95 = window["interval_p95_ns"]
                p95_ms = p95 / 1_000_000 if p95 is not None else float("nan")
                print(
                    f"    {window['start_offset_seconds']:.0f}-"
                    f"{window['end_offset_seconds']:.0f}s: "
                    f"帧数={window['message_count']}, P95={p95_ms:.3f}ms"
                )
        relationship = file_result.get("four_camera_header_time_relationship")
        if relationship:
            print(
                "  四路 header.stamp 最近帧时间差（同一时钟前提）: "
                f"P95={relationship['p95_ns'] / 1_000_000:.3f}ms, "
                f"最大={relationship['maximum_ns'] / 1_000_000:.3f}ms"
            )
        for error in file_result.get("errors", []):
            print(f"  异常: {error}")

    assert result["is_valid"], "原始 MCAP 审计发现错误，详见 Allure 附件和控制台报告"
