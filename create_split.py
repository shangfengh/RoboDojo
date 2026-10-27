"""Create the approved episode-level split without moving dataset files."""

import hashlib
import json
import random
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parent
DATA = Path("/Data/songhuang/Data/RoboDojo_lerobot_v30_video")


def main():
    task_path = DATA / "meta/tasks.parquet"
    episode_path = DATA / "meta/episodes/chunk-000/file-000.parquet"
    tasks = pq.ParquetFile(task_path).read().to_pylist()
    episodes = pq.ParquetFile(episode_path).read(
        columns=["episode_index", "tasks", "length"]
    ).to_pylist()
    groups = {}
    for task in tasks:
        instruction = task["__index_level_0__"]
        group = sorted(
            (e for e in episodes if e["tasks"] == [instruction]),
            key=lambda e: e["episode_index"],
        )
        if len(group) != 100:
            raise ValueError(f"Expected 100 episodes for task {task['task_index']}")
        groups[task["task_index"]] = group
    rng = random.Random(42)
    partitions = []
    train, validation = [], []
    for task_id, group in sorted(groups.items()):
        ids = [e["episode_index"] for e in group]
        rng.shuffle(ids)
        val_ids = sorted(ids[:10])
        train_ids = sorted(ids[10:])
        lengths = {e["episode_index"]: e["length"] for e in group}
        partitions.append({
            "task_index": task_id,
            "train_episode_ids": train_ids,
            "validation_episode_ids": val_ids,
            "train_frames": sum(lengths[i] for i in train_ids),
            "validation_frames": sum(lengths[i] for i in val_ids),
        })
        train.extend(train_ids)
        validation.extend(val_ids)
    all_ids = {e["episode_index"] for e in episodes}
    if len(train) != 3150 or len(validation) != 350:
        raise ValueError("Unexpected split counts")
    if set(train) & set(validation) or set(train) | set(validation) != all_ids:
        raise ValueError("Split overlap or incomplete coverage")
    if len(set(train)) != len(train) or len(set(validation)) != len(validation):
        raise ValueError("Duplicate episode assignment")
    report = {
        "schema_version": 1,
        "dataset": str(DATA),
        "dataset_revision": json.loads((ROOT / "data.sources.json").read_text())[
            "repository"
        ]["revision"],
        "metadata_sha256": {
            str(p.relative_to(DATA)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (task_path, episode_path)
        },
        "seed": 42,
        "algorithm": (
            "Python random.Random(42), tasks sorted by task_index, episodes sorted "
            "by episode_index, shuffle each task with the shared RNG, first 10 "
            "validation and remaining 90 train. Explicit IDs are authoritative."
        ),
        "train_episode_ids": sorted(train),
        "validation_episode_ids": sorted(validation),
        "train_frames": sum(p["train_frames"] for p in partitions),
        "validation_frames": sum(p["validation_frames"] for p in partitions),
        "tasks": partitions,
        "constraints": [
            "All 35 tasks retained, including the unmatched spelling task.",
            "No original data is copied, moved, modified, or excluded.",
            "Episode-disjoint only; shared-source and content duplicate leakage are not certified.",
            "Training normalization must use train episodes only; existing global stats are not train-only.",
            "Video boundary treatment is not specified by this split.",
        ],
    }
    output = ROOT / "split-seed42.json"
    with output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(f"Saved {output}")
    print(f"Train: {len(train)} episodes, {report['train_frames']} frames")
    print(f"Validation: {len(validation)} episodes, {report['validation_frames']} frames")


if __name__ == "__main__":
    main()
