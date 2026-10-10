# W weighting mechanism: thin gaps remain a weak point

Measured while W training is underway, before its final outputs. This is an analytic pixel-weight diagnostic on four fixed96px synthetic targets, not model quality evidence. The imported numerical implementation and data are bound by the saved plan/script hashes under `runs/local-tone-balance-20261010/weighting-mechanism`.

| Target query / same-phase reference | Global ratio | W local8 ratio | Pixel-gradient magnitude, global→W |
| --- | ---: | ---: | ---: |
| 1px ink line / 10px ink band | 1 | 3.1623 | .00063453→.00224647 |
| 4px gap in a1px ink line / distant paper | 1 | 1.0233 | .00020208→.00016395 |
| 2px paper gap between two1px lines / distant paper | 1 | 1.0646 | .00019672→.00017120 |
| 1px paper slit inside a32px black area / distant paper | 1 | 4.1231 | .00018050→.00069250 |

The derivative magnitudes are for one wrong pixel at strength.5 with only the supervised interval-L1 term; signs always point toward its teacher. The W formula strongly emphasizes thin ink and a narrow paper slit in a mostly black neighborhood. However, an intended gap in a thin line remains surrounded mostly by paper in a square window. It receives little extra weight relative to distant paper. Per-image normalization then reduces its absolute gradient by18.9% (thin-line gap) and13.0% (parallel gap) in these fixtures.

Thus the unit tests for a narrow white slit do not establish equally strong emphasis for all line-separation errors. This is a specific limitation of the proposed local minority statistic, and a possible route to ink/bridge tradeoffs. W's4k plan and adoption guards remain unchanged. Do not call it a topology loss or infer a learned failure before evaluation.

If W fails, a later hypothesis could preserve the existing total ink/paper weight budgets while redistributing weight within each phase, or use directional neighborhoods to detect gaps along a thin stroke. Neither alternative is implemented, scheduled or validated here. A directional method would also need rotation and antialiasing checks. Do not silently combine them with the frozen W comparison.
