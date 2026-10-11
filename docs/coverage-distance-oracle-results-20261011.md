# BG: coverage survives the distance extension; no learned quality claim

All14 teacher-only coverage round trips have exactly zero maximum error, including
partial-mask reconstruction. The50 checks passed. Nine tests passed(.72s), all28
field/output blocks were audited and three sheets reviewed. No neural updates.
Numerical commit26185dc.

Thresholded-mask distance loses fractional coverage: AD geometry coverage-overlap
scores range .81776–.94846, despite perfect access to the teacher. The alpha.2
line disappears entirely, and alpha.5 becomes black. This is an unsuitable target
representation if these shades are supposed to represent pixel coverage.

The alternative preserves fractional y as d=y/2 and extends saturated ink/paper
with clipped pixel-center distances. Rendering clip(2*d,-1,1) is exact by
construction. All14 cases, including the256-tone tile, reconstruct exactly;
this explicitly shows invertibility is NOT a guarantee of lineart appearance.
The field is not a continuous SDF or an Eikonal target. It supplies an additional
spatial target outside the antialias band; learning benefit remains untested.

- [Geometry native comparisons](../runs/coverage-distance-oracle-20261011/evaluation/geometry-native.png)
- [Extra native comparisons](../runs/coverage-distance-oracle-20261011/evaluation/extras-native.png)
- [Distance extensions](../runs/coverage-distance-oracle-20261011/evaluation/fields-native.png)
- [Final audit](../runs/coverage-distance-oracle-20261011/final-audit.json)

Distance-based differentiable rasterization is explored by [Mihai and Hare](https://arxiv.org/abs/2103.16194).
[DeepLSD](https://arxiv.org/abs/2212.07766) learns line-attraction fields for a
different straight-line detection task. Our construction is a separate simple
target extension; neither paper proves it will improve curved line restoration.
No code/model was imported. Production and runtime unchanged.
