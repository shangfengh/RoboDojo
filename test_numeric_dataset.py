import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from numeric_dataset import NumericDataset, numeric_array


class NumericDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "data/chunk-000").mkdir(parents=True)
        (self.root / "meta/episodes/chunk-000").mkdir(parents=True)
        info = {
            "fps": 25, "total_episodes": 3, "total_frames": 27,
            "features": {"observation.state": {"shape": [14]}, "action": {"shape": [14]}},
        }
        (self.root / "meta/info.json").write_text(json.dumps(info))
        self.lengths = [3, 20, 4]
        self.ids = np.repeat([0, 1, 2], self.lengths)
        self.frames = np.concatenate([np.arange(n) for n in self.lengths])
        self.state = np.repeat((self.ids * 100 + self.frames)[:, None], 14, axis=1).astype("float32")
        self.action = self.state + np.float32(.25)
        # The validation outlier must not influence train statistics.
        self.state[self.ids == 2] += 10000
        self.action[self.ids == 2] += 10000
        table = pa.table({
            "observation.state": pa.FixedSizeListArray.from_arrays(
                pa.array(self.state.flatten(), type=pa.float32()), 14),
            "action": pa.FixedSizeListArray.from_arrays(
                pa.array(self.action.flatten(), type=pa.float32()), 14),
            "episode_index": pa.array(self.ids, type=pa.int64()),
            "frame_index": pa.array(self.frames, type=pa.int64()),
            "timestamp": pa.array(self.frames / 25, type=pa.float32()),
            "task_index": pa.array(np.zeros(27, dtype="int64")),
        })
        # One episode spans two storage files.
        pq.write_table(table.slice(0, 10), self.root / "data/chunk-000/file-000.parquet")
        pq.write_table(table.slice(10), self.root / "data/chunk-000/file-001.parquet")
        task_path = self.root / "meta/tasks.parquet"
        episode_path = self.root / "meta/episodes/chunk-000/file-000.parquet"
        pq.write_table(pa.table({"task_index": [0], "__index_level_0__": ["task"]}), task_path)
        pq.write_table(pa.table({
            "episode_index": [0, 1, 2], "length": self.lengths,
            "tasks": [["task"], ["task"], ["task"]],
        }), episode_path)
        self.split = {
            "train_episode_ids": [0, 1], "validation_episode_ids": [2],
            "train_frames": 23, "validation_frames": 4,
            "metadata_sha256": {
                str(p.relative_to(self.root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (task_path, episode_path)
            },
        }
        self.split_path = self.root / "split.json"
        self.write_split()

    def write_split(self):
        self.split_path.write_text(json.dumps(self.split))

    def load(self, partition="train"):
        return NumericDataset(self.root, self.split_path, partition)

    def test_first_and_last_windows_do_not_cross_episode(self):
        data = self.load()
        first = data[0]
        self.assertEqual(first["history.state"].shape, (32, 14))
        self.assertEqual(first["action"].shape, (16, 14))
        self.assertEqual(first["history.valid_mask"].sum(), 1)
        self.assertEqual(first["action.valid_mask"].sum(), 3)
        np.testing.assert_array_equal(first["action"][:3], self.action[:3])
        last = data[2]
        self.assertEqual(last["action.valid_mask"].sum(), 1)
        np.testing.assert_array_equal(last["action"][0], self.action[2])
        self.assertFalse(np.array_equal(last["action"][0], last["observation.state"]))
        np.testing.assert_array_equal(last["action"][1:], 0)
        self.assertEqual(data[3]["episode_index"], 1)
        self.assertEqual(data[3]["history.valid_mask"].sum(), 1)
        np.testing.assert_array_equal(data[3]["history.state"][:-1], 0)

    def test_full_action_window_and_history_are_time_aligned(self):
        data = self.load()
        sample = data[7]
        self.assertEqual(sample["frame_index"], 4)
        self.assertEqual(sample["action.valid_mask"].sum(), 16)
        np.testing.assert_array_equal(sample["action"], self.action[7:23])
        np.testing.assert_array_equal(sample["history.state"][-5:], self.state[3:8])
        sample["observation.state"][:] = -100
        self.assertGreaterEqual(data[7]["observation.state"].min(), 100)

    def test_train_stats_exclude_heldout_rows(self):
        data = self.load()
        stats = data.training_statistics()["features"]
        selected = self.ids != 2
        for key, raw in (("observation.state", self.state), ("action", self.action)):
            self.assertEqual(stats[key]["count"], 23)
            np.testing.assert_allclose(stats[key]["mean"], raw[selected].mean(0, dtype="float64"))
            np.testing.assert_allclose(stats[key]["std"], raw[selected].std(0, dtype="float64"))
        with self.assertRaisesRegex(ValueError, "only.*train"):
            self.load("validation").training_statistics()

    def test_validation_partition_and_bad_indices(self):
        data = self.load("validation")
        self.assertEqual(len(data), 4)
        self.assertEqual(data[0]["episode_index"], 2)
        self.assertGreater(data[0]["observation.state"].min(), 10000)
        for bad in (-1, 4, 1.5):
            with self.assertRaises(IndexError):
                data[bad]

    def test_split_overlap_and_missing_coverage_rejected(self):
        for validation in ([0, 2], []):
            self.split["validation_episode_ids"] = validation
            self.write_split()
            with self.assertRaisesRegex(ValueError, "Split must"):
                self.load()

    def test_changed_metadata_rejected(self):
        with (self.root / "meta/tasks.parquet").open("ab") as stream:
            stream.write(b"x")
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            self.load()

    def test_numeric_slices_and_null_rejection(self):
        array = pa.array([5, 6, 7], type=pa.int64())
        np.testing.assert_array_equal(numeric_array(array.slice(1), "int64"), [6, 7])
        with self.assertRaises(ValueError):
            numeric_array(pa.array([1, None], type=pa.int64()), "int64")

    def test_history_truncation_and_invalid_options(self):
        data = NumericDataset(self.root, self.split_path, "train", history_length=4)
        sample = data[12]
        self.assertEqual(sample["history.valid_mask"].sum(), 4)
        np.testing.assert_array_equal(sample["history.state"], self.state[9:13])
        for partition, history, horizon in (("test", 32, 16), ("train", 0, 16), ("train", 32, 0)):
            with self.assertRaises(ValueError):
                NumericDataset(self.root, self.split_path, partition, history, horizon)

    def test_corrupted_frame_identity_rejected(self):
        path = self.root / "data/chunk-000/file-000.parquet"
        table = pq.ParquetFile(path).read()
        wrong_frames = self.frames[:10].copy()
        wrong_frames[1] = 0
        table = table.set_column(
            table.schema.get_field_index("frame_index"), "frame_index",
            pa.array(wrong_frames, type=pa.int64()),
        )
        pq.write_table(table, path)
        with self.assertRaisesRegex(ValueError, "Invalid frame identity"):
            self.load()


if __name__ == "__main__":
    unittest.main()
