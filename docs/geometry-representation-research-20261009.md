# Geometry supervision: research notes, not a queued experiment

The active preclip-opacity comparison isolates one missing gradient route. A successful opacity term would still not directly specify tangent continuity, line width or deliberate gaps. No additional training change is selected from these notes before its fixed final evaluation.

Distance-based representations are a possible later architecture study, distinct from the already tested `boundary_distance_loss`, which applies soft distance matching to raster outputs. [Differentiable Drawing and Sketching](https://arxiv.org/abs/2103.16194) constructs differentiable rasterization from primitive distance transforms. It supports investigating a geometry-constrained output head, but does not establish that a small recursive denoiser will improve on this dataset.

[DeepLSD](https://openaccess.thecvf.com/content/CVPR2023/papers/Pautrat_DeepLSD_Line_Segment_Detection_and_Refinement_With_Deep_Image_Gradients_CVPR_2023_paper.pdf) separately predicts distance and orientation fields for line-segment detection. This suggests a possible auxiliary representation rather than treating all grayscale values as the only geometry signal. Its straight-line detection task and pseudo-label supervision differ from antialiased curved drawing restoration; importing its loss directly is not justified.

An implementation here would first need an oracle reconstruction check: the chosen geometric targets and rasterizer must reproduce thin antialiased lines, corners, crossings, parallel curves, tapered widths and intentional gaps before a network is trained. Otherwise an attractive structural prior could impose the very geometry errors the current evaluation rejects. Blank images and protected boundaries need explicit treatment. A scalar distance field alone does not guarantee correct topology, and unweighted orientation loss at junctions is ambiguous.

The immediate decision still comes from the frozen preclip-opacity comparison. These sources are supporting context for a possible later hypothesis, not evidence of experimental success or a reason to relax current safeguards.
