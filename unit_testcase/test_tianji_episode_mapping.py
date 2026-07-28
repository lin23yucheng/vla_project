from testcase.test_tianji_v2_verify import (
    TASK_ANNOTATION_CONFIG,
    TestTianjiV2Verify as TianjiVerifier,
)


def test_l2_and_l3_segments_resolve_parent_l1_by_sequence():
    verifier = object.__new__(TianjiVerifier)
    verifier.created_segments_by_key = {}

    l1_segments = next(
        layer["segments"]
        for layer in TASK_ANNOTATION_CONFIG["layers"]
        if layer["layerId"] == "l1"
    )
    expected = {
        segment["current_sequence"]: (
            segment["startTimeNs"], segment["endTimeNs"]
        )
        for segment in l1_segments
    }

    assert verifier._resolve_parent_l1_range_ns("l2_01") == expected[4]
    assert verifier._resolve_parent_l1_range_ns("l2_04") == expected[6]
    assert verifier._resolve_parent_l1_range_ns("l3_03") == expected[6]
