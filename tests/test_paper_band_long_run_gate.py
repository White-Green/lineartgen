import pytest
import train_paper_band_long_run as trainer
from campaign_protocol import sha, write


def test_completed_failed_fit_is_valid_context_but_not_an_adoption(monkeypatch, tmp_path):
    gate = tmp_path / "context"
    gate.mkdir()
    final = dict(
        stage="complete_fitting_diagnostic",
        final_step=7200,
        final_images_reviewed=2,
        full_state_restore_exact=True,
        adoption=False,
        methods_advance={"constant": False, "settle": False},
    )
    write(gate / "final-audit.json", final)
    monkeypatch.setattr(trainer, "check", lambda _: dict(fit_gate=str(gate)))
    write(
        tmp_path / "fit-gate-audit.json",
        dict(allowed=True, final_audit_sha256=sha(gate / "final-audit.json")),
    )
    trainer.check_advancement(tmp_path)
    final["final_images_reviewed"] = 0
    write(gate / "final-audit.json", final)
    write(
        tmp_path / "fit-gate-audit.json",
        dict(allowed=True, final_audit_sha256=sha(gate / "final-audit.json")),
    )
    with pytest.raises(AssertionError):
        trainer.check_advancement(tmp_path)


def test_context_digest_cannot_change_after_launch(monkeypatch, tmp_path):
    gate = tmp_path / "context"
    gate.mkdir()
    write(gate / "final-audit.json", {})
    monkeypatch.setattr(trainer, "check", lambda _: dict(fit_gate=str(gate)))
    write(tmp_path / "fit-gate-audit.json", dict(allowed=True, final_audit_sha256="changed"))
    with pytest.raises(AssertionError):
        trainer.check_advancement(tmp_path)
