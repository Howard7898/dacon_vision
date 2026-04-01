# Checkerboard Camera-Normalization Summary

## Conclusion
- `top view` checkerboard rotation normalization is worth using.
- full perspective / homography-style rectification is not robust enough for the default pipeline.
- `front view` horizon alignment is also not strong enough to enable by default.

## Evidence
- Existing preprocessing selection in `/Users/mgo/Downloads/open (7) 2/work_shared/preprocessing/preaugmentation_preproc_selection/preaug_preproc_selection_summary.json`
  - best geometry combo: `front_none__top_rot`
  - `front_none__top_none` dev logloss: `0.3332289582`
  - `front_none__top_rot` dev logloss: `0.3186997761`
  - `front_none__top_rot_persp` dev logloss: `0.3389490868`
- Geometry reliability in `/Users/mgo/Downloads/open (7) 2/work_shared/preprocessing/chapter3_geometry/chapter3_summary.json`
  - `top_rot` revert rate: `0.3338`
  - `top_rot_persp` revert rate: `0.9662`
  - `front_horizon` revert rate: `0.9038`
- Split-wise success from `/Users/mgo/Downloads/open (7) 2/work_shared/preprocessing/chapter3_geometry/chapter3_geometry_meta.csv`
  - `rot_ok`: train `1.000`, dev `1.000`, test `0.994`
  - `persp_ok`: train `0.018`, dev `0.230`, test `0.204`

## Pipeline decision
- Enable checkerboard-guided top rotation normalization by default.
- Keep full perspective rectification out of the mainline until a more reliable board-to-plane homography is available.
