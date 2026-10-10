# U: strong corruption mixture, 2026-10-10

**Completed; not adopted.** Stronger input augmentation restores more opacity and removes measured cuts, but thickens strokes and fills intended gaps. The fixed raw4k endpoint fails58/229 established development checks. All14 final sheets were inspected. This is a matched data comparison against R, not independent validation: the new ranges deliberately cover the existing strong development corruption.

The [fixed plan](strong-corruption-mixture-plan-20261010.md) changes only `sketch.strong_probability=.5`. Both arms start from E raw4k with fresh optimizer/EMA; R's historical controls are reproduced exactly. Same130,605 parameters, native teacher stream, geometry/coverage fractions, interval L1 and auxiliaries, recursive1–6 training, stage multiplier0 with recursive gradients, and sqrt(batch) learning rate. Both4k schedules contain16,000 teachers. Strong enabled geometry exposures (opacity<=.45 and sigma>=1.5, before editing masks) rise53→482 for blur and53→414 for strokes; these are not thin-only counts.

Whole-image strength.5 aggregate results:

| Metric | R | U strong50 |
| --- | ---: | ---: |
| Corpus blur F1 | .936629 | .934268 |
| Corpus blur ink deficit | .070756 | .037420 |
| Corpus blur halo | .008512 | .010005 |
| Corpus strokes F1 | .843711 | .836889 |
| Corpus strokes halo | .095144 | .101251 |
| Geometry blur F1 | .824615 | .871429 |
| Geometry blur ink deficit | .108294 | .025083 |
| Geometry blur halo | .025011 | .033841 |
| Geometry blur width MAE, px | .589909 | .735485 |
| Geometry blur cuts | .083333 | 0 |
| Geometry blur bridges | .125000 | .333333 |
| Geometry strokes F1 | .682142 | .659740 |
| Geometry strokes halo | .161461 | .186224 |
| Geometry strokes width MAE, px | 2.339986 | 2.770459 |
| Geometry strokes bridges | .395833 | .437500 |

The58 failures comprise18 grouped F1,19 halo-versus-reference,11 bridge,7 width,1 primary geometry F1,1 absolute corpus strokes halo, and1 blank guard. Both edited black interiors retain mean/minimum alpha1. Blank maximum opacity is.135919, exceeding.05; the ordinary display sheets look white because the residual is localized, not a broad gray field.

Local raw4k blur.5 F1 improves .799440→.841701 TRAIN and.785048→.839859 PROBE. Deficit falls .206846→.033227 and.251725→.060791, but halo rises .023949→.029971 and.018760→.030578. Local failures decrease20 at1k→13 at4k. These are diagnostic panels, not the adoption screen. Native sheets show darker curves/corners/junctions but rounded ends, overthick repeated strokes, merged close parallels and filled gaps; dense tire hatching loses separation. The criterion is clean line geometry, not just stronger opacity.

44 tests passed. Frozen numerical commit `3bebe4c29b2ec2a038d6ca08f224db920844278d`. The data audit reproduces all166 R clean batch hashes and all6 masks; Gaussian draws remain the same with maximum residual roundoff4.768e-7. Default augmentation agrees bitwise with frozen R. Actual startup6/1k/4k complete state restores are exact, four raw/EMA exports match, all4,000 sampling records and counts agree, and reused E/R references reproduce every metric. Evaluation contains128 primary,912 local broad and4,698 broader outputs; finite/protected pixels are exact. Hash-bound reviews and `final-audit.json` are in `runs/strong-corruption-mixture-20261010`.

All rough inputs are synthetic; corpus means existing clean images with synthetic corruption. No sealed draws, independent training seed, production checkpoint or plugin change. U does not justify more severe augmentation as a standalone fix. The next controlled hypothesis is the narrow final decoder and main head, which Q's encoder/detail widening did not expand.
