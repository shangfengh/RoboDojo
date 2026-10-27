"""Inspect downloaded metadata without loading frame tables or decoding videos."""

import ast
import hashlib
import json
import runpy
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parent
DATA = Path("/Data/songhuang/Data/RoboDojo_lerobot_v30_video")
BENCHMARK = ROOT / "RoboDojo"


def read_rows(path):
    return pq.ParquetFile(path).read().to_pylist()


def summarize(values):
    return {
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "max": max(values),
    }


def main():
    info_path = DATA / "meta/info.json"
    tasks_path = DATA / "meta/tasks.parquet"
    episodes_path = DATA / "meta/episodes/chunk-000/file-000.parquet"
    stats_path = DATA / "meta/stats.json"
    info = json.loads(info_path.read_text())
    stats = json.loads(stats_path.read_text())
    tasks = read_rows(tasks_path)
    episodes = read_rows(episodes_path)
    assert len(tasks) == info["total_tasks"]
    assert len(episodes) == info["total_episodes"]
    assert sorted(e["episode_index"] for e in episodes) == list(range(len(episodes)))
    assert sum(e["length"] for e in episodes) == info["total_frames"]

    inventory = runpy.run_path(str(BENCHMARK / "scripts/internal/task_inventory.py"))[
        "build_inventory"
    ]()
    instructions = defaultdict(list)
    for task in inventory["tasks"]:
        tree = ast.parse((BENCHMARK / task["module"]).read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name != "gen_instruction":
                    continue
                for child in ast.walk(node):
                    if isinstance(child, ast.Constant) and isinstance(child.value, str):
                        instructions[child.value].append(task["name"])

    instruction_tasks = {t["__index_level_0__"]: t["task_index"] for t in tasks}
    assert len(instruction_tasks) == len(tasks)
    grouped = defaultdict(list)
    previous_end = 0
    video_span_counts = Counter()
    for episode in sorted(episodes, key=lambda e: e["episode_index"]):
        assert episode["dataset_from_index"] == previous_end
        assert episode["dataset_to_index"] - previous_end == episode["length"]
        previous_end = episode["dataset_to_index"]
        assert len(episode["tasks"]) == 1
        instruction = episode["tasks"][0]
        assert instruction in instruction_tasks
        grouped[instruction_tasks[instruction]].append(episode)
        for camera in info["features"]:
            if not camera.startswith("observation.images."):
                continue
            prefix = f"videos/{camera}"
            start = episode[f"{prefix}/from_timestamp"]
            end = episode[f"{prefix}/to_timestamp"]
            assert end > start >= 0
            video_frames = round((end - start) * info["fps"])
            video_span_counts[video_frames - episode["length"]] += 1
    assert previous_end == info["total_frames"]

    records = []
    mapped = set()
    dimensions = {t["name"]: t["dimension"] for t in inventory["tasks"]}
    for task in sorted(tasks, key=lambda t: t["task_index"]):
        task_id = task["task_index"]
        instruction = task["__index_level_0__"]
        matches = sorted(set(instructions[instruction]))
        base_matches = [name for name in matches if not name.endswith("_random")]
        mapped.update(base_matches)
        group = grouped[task_id]
        records.append({
            "task_index": task_id,
            "instruction": instruction,
            "exact_instruction_matches": matches,
            "base_task": base_matches[0] if len(base_matches) == 1 else None,
            "dimension": dimensions[base_matches[0]] if len(base_matches) == 1 else None,
            "episodes": len(group),
            "frames": sum(e["length"] for e in group),
            "episode_length_frames": summarize([e["length"] for e in group]),
            "episode_ids": [e["episode_index"] for e in group],
        })

    report = {
        "dataset": str(DATA),
        "revision": json.loads((ROOT / "data.sources.json").read_text())["repository"]["revision"],
        "metadata_sha256": {
            str(path.relative_to(DATA)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (info_path, tasks_path, episodes_path, stats_path)
        },
        "total_episodes": len(episodes),
        "total_frames": previous_end,
        "total_tasks": len(tasks),
        "fps": info["fps"],
        "declared_robot_type": info["robot_type"],
        "declared_splits": info["splits"],
        "episode_columns": pq.ParquetFile(episodes_path).schema_arrow.names,
        "episode_length_frames": summarize([e["length"] for e in episodes]),
        "tasks": records,
        "benchmark_counts": inventory["counts"],
        "unmatched_dataset_task_indices": [
            r["task_index"] for r in records if r["base_task"] is None
        ],
        "benchmark_base_tasks_without_exact_training_instruction": [
            t["name"] for t in inventory["tasks"]
            if t["runnable"] and t["variant"] == "standard" and t["name"] not in mapped
        ],
        "video_metadata_span_frames_minus_episode_length": dict(sorted(video_span_counts.items())),
        "declared_global_ranges": {
            feature: {key: stats[feature][key] for key in ("min", "max")}
            for feature in ("observation.state", "action")
        },
        "scope_and_split_constraints": [
            "Only metadata was read; frame/action tables and video content were not inspected.",
            "Task mapping uses exact string literals in gen_instruction; it does not prove collection provenance.",
            "Metadata declares only a train split. No train/validation split has been created.",
            "Episode metadata has no explicit original recording, collection seed, or embodiment identity column.",
            "Do not treat shared parquet or video containers as original recording identities.",
            "An episode-stratified split is possible, but does not by itself rule out shared-source or duplicate leakage.",
            "Global stats include all episodes; recompute normalization from the eventual training subset.",
            "Video timestamp span differences require decoder and frame-alignment checks before training.",
        ],
    }
    output = ROOT / "metadata-report.json"
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Saved {output}")
    print(f"Tasks={len(tasks)}, episodes={len(episodes)}, frames={previous_end}")
    print("Episodes per task:", dict(Counter(r["episodes"] for r in records)))
    print("Length frames:", report["episode_length_frames"])
    print("Unmatched dataset tasks:", report["unmatched_dataset_task_indices"])
    print("Benchmark tasks without exact training instruction:",
          report["benchmark_base_tasks_without_exact_training_instruction"])
    print("Video span deltas:", report["video_metadata_span_frames_minus_episode_length"])
    for record in records:
        print(record["task_index"], record["base_task"], record["dimension"],
              record["episodes"], record["frames"])


if __name__ == "__main__":
    main()
