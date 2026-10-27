"""CPU smoke with real numeric rows and explicitly synthetic post-bridge features."""

import json
import subprocess
from pathlib import Path

import torch

from lola_joint import LoLAJointCore, joint_config, load_train_statistics, numeric_batch
from numeric_dataset import NumericDataset, sha256


ROOT = Path(__file__).resolve().parent
SOURCE = Path("/home/v-wangxiaofa/lzl/lola_lerobot")


def main() -> None:
    torch.set_num_threads(4)
    torch.manual_seed(42)
    split = ROOT / "split-seed42.json"
    stats_path = ROOT / "train-numeric-stats-seed42.json"
    output = ROOT / "lola-joint-cpu-check.json"
    if output.exists():
        raise FileExistsError(output)
    expected_source = json.loads((ROOT / "sources.lock.json").read_text())[
        "repositories"
    ]["lola_v07"]["commit"]
    actual_source = subprocess.check_output(
        ["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True,
    ).strip()
    if actual_source != expected_source:
        raise ValueError("LoLA source commit differs from lock")
    from lerobot.policies.lola_v07 import modeling_lola_v07
    if Path(modeling_lola_v07.__file__).resolve() != SOURCE / "src/lerobot/policies/lola_v07/modeling_lola_v07.py":
        raise ValueError("Imported LoLA from an unexpected checkout")
    dataset = NumericDataset(Path("/Data/songhuang/Data/RoboDojo_lerobot_v30_video"), split, "train")
    stats = load_train_statistics(stats_path, split)
    if stats["source_sha256"] != dataset.source_sha256:
        raise ValueError("Statistics source identity mismatch")
    indices = [0, 31, int(dataset.starts[1]) - 1, int(dataset.starts[1])]
    batch = numeric_batch([dataset[i] for i in indices])
    model = LoLAJointCore(joint_config(small_test=True), stats)
    visual = torch.randn(4, 5, 64)
    global_condition = torch.randn(4, 64)
    mask = torch.ones(4, 5, dtype=torch.bool)
    noise = (torch.randn(4, 2, 32), torch.randn(4, 2, 16))
    result = model(batch, visual, global_condition, mask, torch.full((4,), .5), noise)
    if not torch.isfinite(result["loss"]):
        raise ValueError("Non-finite real-batch loss")
    result["loss"].backward()
    if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
        raise ValueError("Non-finite gradients")
    model.eval()
    prediction = model.sample(batch, visual, global_condition, mask, noise)
    if not torch.isfinite(prediction).all():
        raise ValueError("Non-finite sampled actions")
    grippers = prediction[..., [6, 13]]
    if not ((grippers >= 0) & (grippers <= 1)).all():
        raise ValueError("Sampled grippers outside [0, 1]")
    with output.open("x") as stream:
        json.dump({
            "status": "passed",
            "scope": "CPU FP32, small randomly initialized original LoLA modules, real train numeric rows, synthetic post-bridge visual/global conditions. No VLM, images, GPU, optimizer updates, pretrained policy, or formal training.",
            "lola_source_commit": actual_source,
            "statistics_sha256": sha256(stats_path),
            "adapter_sha256": sha256(ROOT / "lola_joint.py"),
            "check_script_sha256": sha256(Path(__file__)),
            "torch_version": torch.__version__,
            "sample_indices": indices,
            "episodes": [dataset[i]["episode_index"] for i in indices],
            "output_shape": list(prediction.shape),
            "gripper_min": float(grippers.min()), "gripper_max": float(grippers.max()),
            "losses": {key: value.detach().item() for key, value in result.items() if value.ndim == 0},
            "normalization": "Train-only mean/std on arm dimensions; identity on grippers; no clipping or epsilon.",
            "gripper_supervision": "sigmoid + continuous target MSE; no binary threshold",
            "masking": "Reconstruction uses valid steps; latent flow only full valid 8-step chunks.",
        }, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"PASS real batch forward/backward and 3-step sampling: {output}")


if __name__ == "__main__":
    main()
