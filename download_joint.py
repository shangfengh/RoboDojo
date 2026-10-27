import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.hf_api import RepoFile


ROOT = Path("/Data/songhuang/Data")
REVISION = "269457a3b42d8d0ae69898a601f2c1487c2f826e"
PREFIX = "data/RoboDojo_lerobot_v30_video"
REPO = "RoboDojo-Benchmark/RoboDojo"
STAGING = ROOT / f".robodojo-joint-{REVISION}"
TARGET = ROOT / "RoboDojo_lerobot_v30_video"


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    STAGING.mkdir(exist_ok=True)
    dataset = STAGING / PREFIX
    if TARGET.is_symlink():
        if TARGET.resolve() != dataset.resolve():
            raise RuntimeError(f"Target points to a different dataset: {TARGET}")
    elif TARGET.exists():
        raise RuntimeError(f"Refusing to overwrite existing target: {TARGET}")
    else:
        TARGET.symlink_to(dataset, target_is_directory=True)

    files = [
        item
        for item in HfApi().list_repo_tree(
            REPO, path_in_repo=PREFIX, recursive=True,
            revision=REVISION, repo_type="dataset",
        )
        if isinstance(item, RepoFile)
    ]
    if not files:
        raise RuntimeError("Remote file inventory is empty")
    inventory = {
        "repo": REPO,
        "revision": REVISION,
        "target": str(TARGET),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "total_bytes": sum(item.size for item in files),
        "files": [
            {
                "path": item.path,
                "bytes": item.size,
                "sha256": item.lfs.sha256 if item.lfs else None,
                "git_blob_oid": item.blob_id,
            }
            for item in files
        ],
    }
    write_json(STAGING / "download-manifest.json", inventory)
    print(f"Downloading {len(files)} files, {inventory['total_bytes']} bytes", flush=True)
    def download(item):
        return hf_hub_download(
            REPO, filename=item.path, repo_type="dataset",
            revision=REVISION, local_dir=STAGING,
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        for index, _ in enumerate(executor.map(download, files), 1):
            print(f"Downloaded {index}/{len(files)} files", flush=True)
    print("Download returned; verifying all remote file sizes and hashes", flush=True)
    for index, item in enumerate(inventory["files"], 1):
        path = STAGING / item["path"]
        if path.stat().st_size != item["bytes"]:
            raise RuntimeError(f"File size mismatch: {path}")
        digest = hashlib.sha256() if item["sha256"] else hashlib.sha1()
        if not item["sha256"]:
            digest.update(f"blob {item['bytes']}\0".encode())
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        expected = item["sha256"] or item["git_blob_oid"]
        if digest.hexdigest() != expected:
            raise RuntimeError(f"File hash mismatch: {path}")
        if index % 25 == 0 or index == len(files):
            print(f"Verified {index}/{len(files)} files", flush=True)
    write_json(STAGING / "DOWNLOAD_COMPLETE.json", {
        "repo": REPO,
        "revision": REVISION,
        "target": str(TARGET),
        "verified_files": len(files),
        "verified_bytes": inventory["total_bytes"],
        "completed_at": datetime.now(timezone.utc).isoformat(),
    })
    print(f"COMPLETE: {TARGET}", flush=True)


if __name__ == "__main__":
    main()
