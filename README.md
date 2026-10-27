# RoboDojo LoLA workspace

This workspace uses one Git repository at its root.

- `RoboDojo/` contains the benchmark source as ordinary tracked files.
- `RoboDojo/XPolicyLab/` contains the policy framework source as ordinary tracked
  files. Neither directory is a separate Git repository or a submodule.
- `sources.lock.json` records the original upstream revisions. Original license
  and attribution files are retained alongside the vendored sources.
- One upstream illustration,
  `RoboDojo/XPolicyLab/policy/Spirit_v15/spirit_v15/assets/rc_results.png`,
  is excluded: the acquired checkout contains only an unresolved Git LFS pointer,
  not the image. No policy source files are excluded for this reason.
- IsaacLab and CuRobo have not been downloaded. Their recorded revisions are not
  active submodules. Do not run `git submodule update` or the upstream install
  script's submodule initialization to install them in this workspace; dependency
  setup for this single-repository layout has not yet been implemented.
- LoLA source is currently referenced externally at
  `/home/v-wangxiaofa/lzl/lola_lerobot`; it is not included in this repository.

Datasets remain outside the repository at `/Data/songhuang/Data`. Data, weights,
runtime logs, caches, and generated inspection images are not versioned. Dataset
splits, train-only statistics, and small verification reports are versioned.

## Current integration scope

`numeric_dataset.py` loads episode-local numeric history and action windows.
`lola_joint.py` reuses original LoLA-v07 state/action encoders and the five-stream
DiT for 14-dimensional joint-space actions. Arm coordinates use train-only
mean/std normalization; grippers use sigmoid outputs and continuous-target MSE,
without binarization. Partial action chunks have masked action reconstruction
losses but no latent flow loss.

The action-core check uses real numeric rows, small randomly initialized original
modules, and synthetic post-bridge visual conditions. It does not validate the
image/VLM pipeline or constitute formal training.

Run the existing CPU tests in the verified environment:

```bash
PYTHONPATH=/home/v-wangxiaofa/lzl/lola_lerobot/src \
CUDA_VISIBLE_DEVICES='' \
/home/v-wangxiaofa/anaconda3/envs/lola_eval_local/bin/python \
  -m unittest -v test_lola_joint test_numeric_dataset
```

Commit and push from the workspace root; no recursive clone is needed:

```bash
git add <changed-files>
git commit -m "Describe the change"
git push
```
