"""CPU-only numeric samples; no video reads, tensor framework, or data rewriting.

Fit descriptive train statistics with:
    python numeric_dataset.py --data-root DATA --split SPLIT --output STATS

NumericDataset(DATA, SPLIT, "train" or "validation") returns raw joint values.
History is left-padded, actions are right-padded, and both have valid masks.
No state/action normalization or gripper transform is applied at this stage.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq


STATE_KEY = "observation.state"
ARM_INDICES = (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12)
GRIPPER_INDICES = (6, 13)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def numeric_array(array: pa.Array, dtype: str) -> npt.NDArray[Any]:
    expected = {"float32": pa.float32(), "int64": pa.int64()}[dtype]
    if array.null_count or array.type != expected:
        raise ValueError(f"Expected non-null {expected}, got {array.type}")
    native_dtype = np.dtype(dtype)
    return np.frombuffer(
        array.buffers()[1], dtype=native_dtype, count=len(array),
        offset=array.offset * native_dtype.itemsize,
    )


def vectors(column: pa.ChunkedArray) -> npt.NDArray[np.float32]:
    array = column.combine_chunks()
    if array.null_count or array.type != pa.list_(pa.float32(), 14):
        raise ValueError(f"Expected non-null float32[14], got {array.type}")
    values = array.values.slice(array.offset * 14, len(array) * 14)
    return numeric_array(values, "float32").reshape(-1, 14)


class NumericDataset:
    """One sample per numeric row, with left-padded history and right-padded targets.

    Masks use True for valid values. History includes the current state, never
    future states. Targets start at action[t]; the stored terminal action is
    valid and unchanged. Zeros in masked positions are padding, not observations.
    """

    def __init__(
        self, data_root: Path, split_path: Path, partition: str,
        history_length: int = 32, action_horizon: int = 16,
    ) -> None:
        if partition not in ("train", "validation"):
            raise ValueError(f"Unknown partition: {partition}")
        if history_length <= 0 or action_horizon <= 0:
            raise ValueError("Window lengths must be positive")
        self.history_length = history_length
        self.action_horizon = action_horizon
        self.partition = partition
        self.split_sha256 = sha256(split_path)
        split = json.loads(split_path.read_text())
        info_path = data_root / "meta/info.json"
        info = json.loads(info_path.read_text())
        if (info["features"][STATE_KEY]["shape"] != [14]
                or info["features"]["action"]["shape"] != [14]):
            raise ValueError("This loader supports only joint float32[14] data")
        self.fps = info["fps"]
        if self.fps <= 0:
            raise ValueError("Invalid frame rate")
        self.source_sha256 = {"meta/info.json": sha256(info_path)}
        required_meta = {"meta/tasks.parquet", "meta/episodes/chunk-000/file-000.parquet"}
        if not required_meta.issubset(split["metadata_sha256"]):
            raise ValueError("Split lacks required metadata identities")
        for relative, expected in split["metadata_sha256"].items():
            actual = sha256(data_root / relative)
            if actual != expected:
                raise ValueError(f"Split metadata identity mismatch: {relative}")
            self.source_sha256[relative] = actual
        episodes = pq.ParquetFile(
            data_root / "meta/episodes/chunk-000/file-000.parquet"
        ).read(columns=["episode_index", "length", "tasks"]).to_pylist()
        episode_meta = {e["episode_index"]: e for e in episodes}
        if len(episode_meta) != len(episodes) or len(episodes) != info["total_episodes"]:
            raise ValueError("Duplicate or inconsistent episode metadata")
        train = split["train_episode_ids"]
        validation = split["validation_episode_ids"]
        if (len(train) != len(set(train)) or len(validation) != len(set(validation))
                or set(train) & set(validation)
                or set(train) | set(validation) != set(episode_meta)):
            raise ValueError("Split must be unique, disjoint, and cover all episodes")
        self.episode_ids = sorted(train if partition == "train" else validation)
        if not self.episode_ids:
            raise ValueError("Empty partition")
        selected_ids = np.array(self.episode_ids, dtype=np.int64)
        tasks = pq.ParquetFile(data_root / "meta/tasks.parquet").read().to_pylist()
        task_ids = {t["__index_level_0__"]: t["task_index"] for t in tasks}
        if len(task_ids) != len(tasks):
            raise ValueError("Duplicate task instructions")
        self.metadata = episode_meta
        parts: dict[str, list[npt.NDArray[Any]]] = {
            key: [] for key in (STATE_KEY, "action", "episode_index", "frame_index", "timestamp", "task_index")
        }
        paths = sorted((data_root / "data").glob("*/*.parquet"))
        if not paths:
            raise ValueError("No numeric parquet files")
        total_rows = 0
        for path in paths:
            self.source_sha256[str(path.relative_to(data_root))] = sha256(path)
            table = pq.ParquetFile(path).read(columns=list(parts))
            total_rows += len(table)
            ids = numeric_array(table["episode_index"].combine_chunks(), "int64")
            if not np.isin(ids, list(episode_meta)).all():
                raise ValueError(f"Unknown episode in {path}")
            selected = np.isin(ids, selected_ids)
            for key in parts:
                if key in (STATE_KEY, "action"):
                    values = vectors(table[key])
                else:
                    dtype = "float32" if key == "timestamp" else "int64"
                    values = numeric_array(table[key].combine_chunks(), dtype)
                parts[key].append(values[selected])
        if total_rows != info["total_frames"]:
            raise ValueError("Numeric row count differs from info.json")
        arrays = {key: np.concatenate(value) for key, value in parts.items()}
        order = np.lexsort((arrays["frame_index"], arrays["episode_index"]))
        arrays = {key: value[order] for key, value in arrays.items()}
        self.state = arrays[STATE_KEY]
        self.action = arrays["action"]
        if not np.isfinite(self.state).all() or not np.isfinite(self.action).all():
            raise ValueError("Non-finite numeric samples")
        self.starts = np.cumsum(
            [0] + [episode_meta[i]["length"] for i in self.episode_ids], dtype=np.int64,
        )
        if self.starts[-1] != len(self.state) or len(self.state) != split[f"{partition}_frames"]:
            raise ValueError("Partition frame count mismatch")
        for index, episode_id in enumerate(self.episode_ids):
            start, end = self.starts[index:index + 2]
            meta = episode_meta[episode_id]
            if len(meta["tasks"]) != 1 or meta["tasks"][0] not in task_ids:
                raise ValueError(f"Unknown instruction for episode {episode_id}")
            if (not np.all(arrays["episode_index"][start:end] == episode_id)
                    or not np.array_equal(arrays["frame_index"][start:end], np.arange(end - start))
                    or not np.all(arrays["task_index"][start:end] == task_ids[meta["tasks"][0]])
                    or not np.allclose(
                        arrays["timestamp"][start:end], np.arange(end - start) / self.fps,
                        rtol=0, atol=1e-4)):
                raise ValueError(f"Invalid frame identity/order for episode {episode_id}")
        self.state.setflags(write=False)
        self.action.setflags(write=False)

    def __len__(self) -> int:
        return len(self.state)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if not isinstance(index, (int, np.integer)) or index < 0 or index >= len(self):
            raise IndexError(index)
        episode_offset = int(np.searchsorted(self.starts, index, side="right") - 1)
        start, end = self.starts[episode_offset:episode_offset + 2]
        episode_id = self.episode_ids[episode_offset]
        history_start = max(start, index + 1 - self.history_length)
        history_count = index + 1 - history_start
        action_count = min(self.action_horizon, end - index)
        history = np.zeros((self.history_length, 14), dtype=np.float32)
        actions = np.zeros((self.action_horizon, 14), dtype=np.float32)
        history_mask = np.zeros(self.history_length, dtype=bool)
        action_mask = np.zeros(self.action_horizon, dtype=bool)
        history[-history_count:] = self.state[history_start:index + 1]
        actions[:action_count] = self.action[index:index + action_count]
        history_mask[-history_count:] = True
        action_mask[:action_count] = True
        return {
            "episode_index": episode_id,
            "frame_index": int(index - start),
            "instruction": self.metadata[episode_id]["tasks"][0],
            "observation.state": self.state[index].copy(),
            "history.state": history,
            "history.valid_mask": history_mask,
            "action": actions,
            "action.valid_mask": action_mask,
        }

    def training_statistics(self) -> dict[str, Any]:
        if self.partition != "train":
            raise ValueError("Normalization statistics may only be fitted on train")
        features = {}
        for name, values in ((STATE_KEY, self.state), ("action", self.action)):
            features[name] = {
                "count": len(values),
                "mean": np.mean(values, axis=0, dtype=np.float64).tolist(),
                "std": np.std(values, axis=0, dtype=np.float64, ddof=0).tolist(),
                "min": values.min(axis=0).tolist(),
                "max": values.max(axis=0).tolist(),
            }
        return {
            "schema_version": 1,
            "partition": "train",
            "split_sha256": self.split_sha256,
            "source_sha256": self.source_sha256,
            "generator_sha256": sha256(Path(__file__)),
            "numpy_version": np.__version__,
            "episode_ids": self.episode_ids,
            "arm_indices": ARM_INDICES,
            "gripper_indices": GRIPPER_INDICES,
            "weighting": "One contribution per original train row, not per overlapping window.",
            "std_definition": "Population standard deviation (ddof=0), accumulated in float64.",
            "normalization_policy": "Raw statistics only; no clipping, epsilon, gripper binarization, or transform selected.",
            "features": features,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    dataset = NumericDataset(args.data_root, args.split, "train")
    report = dataset.training_statistics()
    report["window_contract"] = {
        "history_shape": [32, 14], "action_shape": [16, 14],
        "history_includes_current": True, "true_mask_means": "valid",
        "padding": "Zero with explicit mask; never crosses episode boundaries.",
        "sampling": "One sample per numeric row; no stride or task reweighting.",
        "video": "Not loaded; this is not yet a complete VLA training dataset.",
    }
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Saved train-only statistics: {args.output}; {len(dataset)} frames")


if __name__ == "__main__":
    main()
