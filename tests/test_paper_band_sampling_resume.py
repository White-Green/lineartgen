import json

import pytest
from train_paper_band_generalization import sampling_resume


def test_sampling_resume_keeps_checkpoint_prefix_and_counts_only_current_segment(tmp_path):
    rows = [
        dict(
            step=step,
            depth=(step % 6) + 1,
            teacher_counts=dict(native_corpus=3, original_geometry=1),
        )
        for step in range(1, 11)
    ]
    path = tmp_path / "sampling.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows) + '{"step":')
    counts, teachers = sampling_resume(tmp_path, 8, 6)
    assert sum(counts.values()) == 2
    assert teachers == dict(native_corpus=6, original_geometry=2)
    assert [json.loads(x)["step"] for x in path.read_text().splitlines()] == list(range(1, 9))
    assert len(list((tmp_path / "sampling-resume-history").glob("*.json"))) == 1
    assert sampling_resume(tmp_path, 8, 6) == (counts, teachers)


def test_sampling_resume_rejects_a_missing_committed_update(tmp_path):
    path = tmp_path / "sampling.jsonl"
    path.write_text(json.dumps(dict(step=2, depth=1, teacher_counts=dict(native_corpus=4))) + "\n")
    with pytest.raises(AssertionError):
        sampling_resume(tmp_path, 2, 0)
