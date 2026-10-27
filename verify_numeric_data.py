"""Verify real-data windows and train statistics with independent episode reductions."""

import argparse
import json
from pathlib import Path

import numpy as np

from numeric_dataset import NumericDataset, sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    stats = json.loads(args.stats.read_text())
    if stats["split_sha256"] != sha256(args.split):
        raise ValueError("Statistics belong to a different split")
    partitions = {}
    for partition in ("train", "validation"):
        dataset = NumericDataset(args.data_root, args.split, partition)
        if stats["source_sha256"] != dataset.source_sha256:
            raise ValueError("Statistics belong to different numeric data")
        checked = 0
        for offset, episode_id in enumerate(dataset.episode_ids):
            start, end = map(int, dataset.starts[offset:offset + 2])
            positions = {start, end - 1, min(start + 31, end - 1), max(start, end - 16)}
            for position in positions:
                sample = dataset[position]
                count_h = min(32, position - start + 1)
                count_a = min(16, end - position)
                assert sample["episode_index"] == episode_id
                assert sample["frame_index"] == position - start
                np.testing.assert_array_equal(sample["history.valid_mask"], np.arange(32) >= 32 - count_h)
                np.testing.assert_array_equal(sample["action.valid_mask"], np.arange(16) < count_a)
                np.testing.assert_array_equal(sample["history.state"][-count_h:],
                                              dataset.state[position - count_h + 1:position + 1])
                np.testing.assert_array_equal(sample["action"][:count_a],
                                              dataset.action[position:position + count_a])
                np.testing.assert_array_equal(sample["history.state"][:32 - count_h], 0)
                np.testing.assert_array_equal(sample["action"][count_a:], 0)
                checked += 1
        if partition == "train":
            for feature, values in (("observation.state", dataset.state), ("action", dataset.action)):
                count = 0
                mean = np.zeros(14, dtype=np.float64)
                m2 = np.zeros(14, dtype=np.float64)
                for start, end in zip(dataset.starts[:-1], dataset.starts[1:], strict=True):
                    block = values[start:end].astype(np.float64)
                    n = len(block)
                    block_mean = block.mean(axis=0)
                    delta = block_mean - mean
                    m2 += ((block - block_mean) ** 2).sum(axis=0) + delta ** 2 * count * n / (count + n)
                    mean += delta * n / (count + n)
                    count += n
                expected = stats["features"][feature]
                assert count == expected["count"]
                np.testing.assert_allclose(mean, expected["mean"], rtol=1e-10, atol=1e-12)
                np.testing.assert_allclose(np.sqrt(m2 / count), expected["std"], rtol=1e-10, atol=1e-12)
                np.testing.assert_array_equal(values.min(0), expected["min"])
                np.testing.assert_array_equal(values.max(0), expected["max"])
        partitions[partition] = {
            "episodes": len(dataset.episode_ids), "frames": len(dataset),
            "boundary_and_interior_windows_checked": checked,
        }
    with args.output.open("x") as stream:
        json.dump({
            "status": "passed", "partitions": partitions,
            "statistics_sha256": sha256(args.stats),
            "verification_script_sha256": sha256(Path(__file__)),
            "checks": [
                "Every episode: first/last and history/action full-window boundaries.",
                "Exact state/action copies, masks, zero padding, and episode identity.",
                "Train mean/std independently recomputed by merging episode moments.",
                "Statistics source hashes match both partition loads.",
                "No video reads or model execution.",
            ],
        }, stream, indent=2)
        stream.write("\n")
    print(json.dumps(partitions, indent=2))


if __name__ == "__main__":
    main()
