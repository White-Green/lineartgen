import importlib.util
import json
import sys
from pathlib import Path


def test_resume_archives_only_uncommitted_log_records(tmp_path, monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "lineart_journal_test", scripts / "train_lineart_loss_study.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    log = tmp_path / "training.jsonl"
    committed = [{"step": 225}, {"step": 250}]
    log.write_text(
        "".join(json.dumps(row) + "\n" for row in [*committed, {"step": 275}]) + '{"step":'
    )
    (tmp_path / "progress.json").write_text(json.dumps({"step": 275}))

    module.restore_journal(tmp_path, 250)

    assert [json.loads(line) for line in log.read_text().splitlines()] == committed
    assert json.loads((tmp_path / "progress.json").read_text()) == {"step": 250}
    archives = list((tmp_path / "resume-history").glob("*.json"))
    assert len(archives) == 1
    archived = json.loads(archives[0].read_text())
    assert archived["restored_step"] == 250
    assert archived["rolled_back_log_lines"] == [json.dumps({"step": 275}), '{"step":']
    module.restore_journal(tmp_path, 250)
    assert len(list((tmp_path / "resume-history").glob("*.json"))) == 1
