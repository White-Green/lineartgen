"""Apply the same 2px peak stroke style to the old wide-line guard cases."""

from style_family_data import styled_candidates


def allstroke_candidates(data, record):
    return styled_candidates(
        data, record | dict(kind="straight") if record["kind"] == "wide" else record
    )
