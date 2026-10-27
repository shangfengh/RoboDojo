"""Check numeric tables and selected AV1 timestamps without changing the dataset."""

import argparse
import hashlib
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
DATA = Path("/Data/songhuang/Data/RoboDojo_lerobot_v30_video")
CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


def numeric_values(array, dtype):
    if array.null_count:
        raise ValueError("Null values in a numeric column")
    dtype = np.dtype(dtype)
    return np.frombuffer(array.buffers()[1], dtype=dtype, count=len(array),
                         offset=array.offset * dtype.itemsize)


def nearest_frame(path, timestamp):
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        container.seek(max(0, int(Fraction(str(timestamp)) / stream.time_base)),
                       stream=stream, backward=True)
        nearest = None
        error = float("inf")
        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError(f"Missing frame PTS: {path}")
            seconds = float(frame.pts * stream.time_base)
            distance = abs(seconds - timestamp)
            if distance < error:
                nearest = frame
                error = distance
            if seconds >= timestamp:
                break
        if nearest is None:
            raise ValueError(f"No decoded frame at {timestamp}: {path}")
        return nearest.to_image(), float(nearest.pts * stream.time_base), error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "sample-check")
    output = parser.parse_args().output
    output.mkdir(exist_ok=True)
    if (output / "report.json").exists():
        raise FileExistsError("Sample report already exists; preserve the existing evidence")
    info = json.loads((DATA / "meta/info.json").read_text())
    episodes = pq.ParquetFile(
        DATA / "meta/episodes/chunk-000/file-000.parquet"
    ).read(columns=[
        "episode_index", "length", "tasks",
        *[f"videos/observation.images.{camera}/{field}"
          for camera in CAMERAS
          for field in ("chunk_index", "file_index", "from_timestamp", "to_timestamp")],
    ]).to_pylist()
    paths = sorted((DATA / "data").glob("*/*.parquet"))
    table = pa.concat_tables([
        pq.ParquetFile(path).read(columns=[
            "observation.state", "action", "episode_index", "frame_index", "timestamp",
        ])
        for path in paths
    ]).combine_chunks()
    state = numeric_values(table["observation.state"].chunk(0).values, "float32").reshape(-1, 14)
    action = numeric_values(table["action"].chunk(0).values, "float32").reshape(-1, 14)
    ids = numeric_values(table["episode_index"].chunk(0), "int64")
    frames = numeric_values(table["frame_index"].chunk(0), "int64")
    timestamps = numeric_values(table["timestamp"].chunk(0), "float32")
    assert len(state) == info["total_frames"]
    assert np.isfinite(state).all() and np.isfinite(action).all()
    same_episode = ids[1:] == ids[:-1]
    assert np.all((frames[1:] - frames[:-1])[same_episode] == 1)
    assert np.all(frames[np.r_[True, ~same_episode]] == 0)
    actual_counts = Counter(ids.tolist())
    assert all(actual_counts[e["episode_index"]] == e["length"] for e in episodes)
    assert max(abs(timestamps - frames / info["fps"])) < 1e-4
    error = np.abs(action[:-1][same_episode] - state[1:][same_episode])
    terminal = np.r_[~same_episode, True]
    terminal_error = np.abs(action[terminal] - state[terminal])
    episode_lookup = {e["episode_index"]: e for e in episodes}
    grippers = {}
    for feature, array in (("state", state), ("action", action)):
        grippers[feature] = []
        for index in (6, 13):
            values = array[:, index]
            interior = (values > 1e-4) & (values < 1 - 1e-4)
            grippers[feature].append({
                "index": index, "min": float(values.min()), "max": float(values.max()),
                "interior_count": int(interior.sum()),
                "interior_fraction": float(interior.mean()),
                "quantiles": np.quantile(values, [0, .1, .25, .5, .75, .9, 1]).tolist(),
                "rounded_4dp_unique_count": len(np.unique(np.round(values, 4))),
            })
    report = {
        "scope": "All numeric state/action rows; selected video timestamps only. No training or data modification.",
        "numeric_files": [str(p.relative_to(DATA)) for p in paths],
        "rows_checked": len(state),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "state_action_finite": True,
        "frame_indices_and_lengths_valid": True,
        "max_timestamp_error_seconds": float(max(abs(timestamps - frames / info["fps"]))),
        "next_state_alignment": {
            "within_episode_pairs": int(same_episode.sum()),
            "max_abs_error_by_dimension": error.max(axis=0).tolist(),
            "mean_abs_error_by_dimension": error.mean(axis=0, dtype=np.float64).tolist(),
            "pairs_exceeding_1e-5": int(np.any(error > 1e-5, axis=1).sum()),
        },
        "terminal_action_state_max_abs_error": float(terminal_error.max()),
        "terminal_action_state_differing_episodes": {
            "tolerance": 1e-5,
            "count": int(np.any(terminal_error > 1e-5, axis=1).sum()),
            "episode_ids": ids[terminal][np.any(terminal_error > 1e-5, axis=1)].tolist(),
        },
        "grippers": grippers,
        "video_checks": [],
    }
    selected = {0, 200, 300, 500, 600, 800}
    sheet = Image.new("RGB", (3 * 320, len(selected) * 265), "white")
    draw = ImageDraw.Draw(sheet)
    row = 0
    for episode in episodes:
        episode_id, length = episode["episode_index"], episode["length"]
        for column, camera in enumerate(CAMERAS):
            prefix = f"videos/observation.images.{camera}"
            start, end = episode[f"{prefix}/from_timestamp"], episode[f"{prefix}/to_timestamp"]
            delta = round((end - start) * info["fps"]) - length
            if not delta and episode_id not in selected:
                continue
            path = DATA / info["video_path"].format(
                video_key=f"observation.images.{camera}",
                chunk_index=episode[f"{prefix}/chunk_index"],
                file_index=episode[f"{prefix}/file_index"],
            )
            checks = []
            neighbor = episode_lookup.get(episode_id + 1)
            next_episode_shares_boundary = bool(
                neighbor is not None
                and neighbor[f"{prefix}/chunk_index"] == episode[f"{prefix}/chunk_index"]
                and neighbor[f"{prefix}/file_index"] == episode[f"{prefix}/file_index"]
                and abs(neighbor[f"{prefix}/from_timestamp"] - end) < 1e-6
            )
            positions = (0, length // 2, length - 1) if episode_id in selected else (0, length - 1)
            for position in positions:
                requested = start + position / info["fps"]
                image, actual, distance = nearest_frame(path, requested)
                checks.append({
                    "frame_index": position, "requested_seconds": requested,
                    "decoded_pts_seconds": actual, "error_seconds": distance,
                    "decoded_pts_within_declared_span": start - 1e-6 <= actual < end - 1e-6,
                    "rgb_size": list(image.size),
                })
                if episode_id in selected and position == length // 2:
                    image = image.resize((320, 240))
                    sheet.paste(image, (column * 320, row * 265 + 25))
                    draw.text((column * 320 + 4, row * 265 + 5),
                              f"episode {episode_id} / {camera}", fill="black")
            report["video_checks"].append({
                "episode_id": episode_id, "camera": camera,
                "video": str(path.relative_to(DATA)),
                "metadata_span_delta_frames": delta,
                "declared_start": start, "declared_end": end, "checks": checks,
                "next_episode_shares_boundary_in_same_video": next_episode_shares_boundary,
            })
        if episode_id in selected:
            row += 1
        if episode_id % 100 == 0:
            print(f"Checked video selections through episode {episode_id}", flush=True)
    report["video_check_summary"] = {
        "segments_checked": len(report["video_checks"]),
        "anomalous_metadata_segments_checked": sum(
            item["metadata_span_delta_frames"] != 0 for item in report["video_checks"]),
        "timestamps_checked": sum(len(item["checks"]) for item in report["video_checks"]),
        "max_nearest_frame_error_seconds": max(
            check["error_seconds"] for item in report["video_checks"] for check in item["checks"]),
        "timestamps_outside_declared_span": sum(
            not check["decoded_pts_within_declared_span"]
            for item in report["video_checks"] for check in item["checks"]),
        "out_of_span_timestamps_at_next_episode_start": sum(
            item["next_episode_shares_boundary_in_same_video"]
            and abs(check["decoded_pts_seconds"] - item["declared_end"]) < 1e-6
            for item in report["video_checks"] for check in item["checks"]
            if not check["decoded_pts_within_declared_span"]),
    }
    sheet.save(output / "three-camera-samples.jpg")
    with (output / "report.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({
        key: report[key] for key in (
            "rows_checked", "next_state_alignment", "terminal_action_state_max_abs_error",
            "terminal_action_state_differing_episodes", "grippers", "video_check_summary",
        )
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
