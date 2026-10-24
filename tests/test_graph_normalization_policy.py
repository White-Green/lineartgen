import json

import numpy as np
import pytest
from lineartgen_runtime.graph import GraphLineartModel


@pytest.mark.parametrize(
    "policy", [None, "local_peak_full", "protected_peak_full", "painted_peak_full"]
)
def test_runtime_uses_explicit_policy_before_area_pyramid(policy):
    model = object.__new__(GraphLineartModel)
    model.metadata = {} if policy is None else {"guide_normalization": policy}
    seen = {}

    def stage(guide, noise, insert, level):
        seen[level] = guide.copy()
        return np.zeros((*guide.shape[:-1], 9), np.float32)

    model.stage = stage
    source = np.ones((1, 128, 128, 1), np.float32)
    source[:, 64, :64] = -1
    source[:, 64, 64:] = 0.2  # 40% ink on the editable side
    noise = np.zeros_like(source)
    noise[:, :, 64:] = 0.5
    model.heads(source, noise, 3)
    fine = 0.2 if policy in ("protected_peak_full", "painted_peak_full") else 0.68
    assert seen[0][0, 64, 64, 0] == pytest.approx(fine)
    # The subpixel stroke gets area averaged, not renormalized at each depth.
    assert seen[1][0, 32, 32, 0] == pytest.approx((fine + 1) / 2)
    assert seen[2][0, 16, 16, 0] == pytest.approx((fine + 3) / 4)


@pytest.mark.parametrize(
    "version,policy,message",
    [
        (4, None, "requires an explicit guide_normalization"),
        (4, "unknown", "Unsupported graph guide_normalization"),
        (3, "protected_peak_full", "requires graph manifest v4"),
        (3, "painted_peak_full", "requires graph manifest v4"),
        (2, "unknown", "Unsupported graph guide_normalization"),
    ],
)
def test_invalid_normalization_manifest_fails_before_onnx_loading(
    tmp_path, version, policy, message
):
    metadata = dict(
        format_version=version, representation="attraction_graph", levels=6, max_stroke_width=None
    )
    if policy is not None:
        metadata["guide_normalization"] = policy
    path = tmp_path / "graph-model.json"
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match=message):
        GraphLineartModel(path)
