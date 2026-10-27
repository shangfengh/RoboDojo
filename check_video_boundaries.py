"""Inspect packet timestamps and decoded tails for the three non-exact requests."""

import hashlib
import json
from pathlib import Path

import av
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parent
DATA = Path("/Data/songhuang/Data/RoboDojo_lerobot_v30_video")


def main():
    source = ROOT / "sample-check-v2/report.json"
    earlier = json.loads(source.read_text())
    episodes = pq.ParquetFile(
        DATA / "meta/episodes/chunk-000/file-000.parquet"
    ).read(columns=[
        "episode_index", "length",
        "videos/observation.images.cam_high/chunk_index",
        "videos/observation.images.cam_high/file_index",
    ]).to_pylist()
    records = []
    for segment in earlier["video_checks"]:
        for check in segment["checks"]:
            if check["error_seconds"] < .001:
                continue
            path = DATA / segment["video"]
            with av.open(str(path)) as container:
                stream = container.streams.video[0]
                time_base = float(stream.time_base)
                pts = [packet.pts for packet in container.demux(stream)
                       if packet.size and packet.pts is not None]
                duration = float(stream.duration * stream.time_base)
            chunk = int(path.parent.name.split("-")[1])
            file_index = int(path.stem.split("-")[1])
            owners = [
                e for e in episodes
                if e["videos/observation.images.cam_high/chunk_index"] == chunk
                and e["videos/observation.images.cam_high/file_index"] == file_index
            ]
            with av.open(str(path)) as container:
                stream = container.streams.video[0]
                container.seek(max(0, int((duration - 1) / time_base)),
                               stream=stream, backward=True)
                tail = [float(frame.pts * stream.time_base)
                        for frame in container.decode(stream)]
            records.append({
                "episode_id": segment["episode_id"],
                "video": segment["video"],
                "requested_seconds": check["requested_seconds"],
                "declared_episode_end": segment["declared_end"],
                "stream_duration_seconds": duration,
                "packet_count": len(pts),
                "packet_pts_unique_count": len(set(pts)),
                "packet_last_pts_seconds": max(pts) * time_base,
                "decoded_tail_last_pts_seconds": max(tail),
                "decoded_tail_frames": len(tail),
                "sum_numeric_episode_lengths_in_file": sum(e["length"] for e in owners),
                "numeric_episodes_in_file": [e["episode_index"] for e in owners],
                "is_last_numeric_episode_in_file": segment["episode_id"] == owners[-1]["episode_index"],
            })
    report = {
        "scope": "Three previously non-exact requests; packet inventory and decoded video tails only.",
        "input_report_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "records": records,
        "correction": (
            "The prior decoded_pts_within_declared_span field assumes a half-open "
            "interval. It is not proof of mispaired image/state content. No frames "
            "are removed or replaced, and metadata boundaries alone cannot establish "
            "whether neighboring episodes intentionally share an observation."
        ),
        "source_findings": {
            "xpolicylab_converter": "Adds one image per numeric row and delegates video writing to LeRobot.",
            "local_lerobot_writer": "Builds episode endpoints from measured encoded video duration.",
            "local_lola_reader": "Queries episode video start plus local row timestamp.",
            "limitation": "Local writer code is not proven to be the code version that produced the published data.",
        },
    }
    with (ROOT / "video-boundary-review.json").open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    for record in records:
        print(json.dumps({k: v for k, v in record.items()
                          if k != "numeric_episodes_in_file"}))


if __name__ == "__main__":
    main()
