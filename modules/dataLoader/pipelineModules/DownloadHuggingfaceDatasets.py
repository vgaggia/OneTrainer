import hashlib
import json
import os
import shutil
import tarfile
import zipfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from pathlib import Path

import huggingface_hub
from filelock import FileLock
from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


HF_ARCHIVE_PREFIX = "hf-archive:"


def is_hf_archive_path(path: str) -> bool:
    return path.startswith(HF_ARCHIVE_PREFIX)


def parse_hf_archive_path(path: str) -> tuple[str, str]:
    spec = path[len(HF_ARCHIVE_PREFIX):].strip()
    parts = spec.split("/", 2)
    if len(parts) != 3 or not all(parts):
        raise ValueError(
            "HF archive paths must look like "
            "'hf-archive:owner/dataset_repo/archive.zip'"
        )

    filename = parts[2]
    lower_filename = filename.lower()
    if not lower_filename.endswith((".zip", ".tar", ".tar.gz", ".tgz")):
        raise ValueError(
            "HF archive paths must point to a .zip, .tar, .tar.gz, or .tgz file"
        )

    return f"{parts[0]}/{parts[1]}", filename


def _safe_extract_zip(archive_path: Path, destination: Path):
    destination = destination.resolve()
    destination_str = str(destination)
    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        for info in infos:
            # ponytail: normpath instead of Path.resolve() - resolve() hits the filesystem once
            # per member, which is ~10 minutes of FUSE round-trips for a 150k-entry archive on a
            # network volume. The destination is freshly created and empty, so there are no
            # symlinks inside it for resolve() to follow that this lexical check would miss.
            target = os.path.normpath(os.path.join(destination_str, info.filename))
            if target != destination_str and not target.startswith(destination_str + os.sep):
                raise RuntimeError(f"Unsafe archive member path: {info.filename}")

        # create directories up front, so the parallel workers below never race on mkdir
        for name in sorted({os.path.dirname(info.filename) for info in infos if os.path.dirname(info.filename)}):
            (destination / name).mkdir(parents=True, exist_ok=True)
        names = [info.filename for info in infos if not info.is_dir()]

    # ponytail: extracting onto a network filesystem is per-file latency bound, not bandwidth
    # bound - serial extractall managed ~18 files/s on RunPod's MooseFS volume. Threads (not
    # processes) because these archives are typically stored uncompressed, so extract() is a
    # GIL-releasing copy, and a thread pool cannot upset an already-initialised CUDA context.
    def extract_names(chunk: list[str]):
        with zipfile.ZipFile(archive_path) as thread_archive:
            for name in chunk:
                thread_archive.extract(name, destination)

    workers = min(48, ((os.cpu_count() or 8) * 4))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in pool.map(extract_names, [names[i::workers] for i in range(workers)]):
            pass


def _safe_extract_tar(archive_path: Path, destination: Path):
    destination = destination.resolve()
    with tarfile.open(archive_path) as archive:
        for member in archive.getmembers():
            target = (destination / member.name).resolve()
            if not str(target).startswith(str(destination) + os.sep):
                raise RuntimeError(f"Unsafe archive member path: {member.name}")
        archive.extractall(destination)


def _archive_cache_root() -> Path:
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home) / "onetainer_archives"
    return Path.home() / ".cache" / "huggingface" / "onetainer_archives"


def _archive_digest(repo_id: str, filename: str, archive_path: Path) -> str:
    stat = archive_path.stat()
    payload = f"{repo_id}\n{filename}\n{archive_path.resolve()}\n{stat.st_size}\n{stat.st_mtime_ns}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _dataset_root(extract_dir: Path) -> Path:
    visible_entries = [
        entry for entry in extract_dir.iterdir()
        if not entry.name.startswith(".")
    ]
    visible_dirs = [entry for entry in visible_entries if entry.is_dir()]
    visible_files = [entry for entry in visible_entries if entry.is_file()]

    if len(visible_dirs) == 1 and not visible_files:
        return visible_dirs[0]

    return extract_dir


def _extract_archive(archive_path: Path, extract_dir: Path, repo_id: str, filename: str) -> Path:
    marker_path = extract_dir / ".extract_complete.json"
    lock_path = extract_dir.with_suffix(".lock")

    extract_dir.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(lock_path)):
        if marker_path.is_file():
            return _dataset_root(extract_dir)

        tmp_extract_dir = extract_dir.with_name(extract_dir.name + ".tmp")
        shutil.rmtree(tmp_extract_dir, ignore_errors=True)
        shutil.rmtree(extract_dir, ignore_errors=True)
        tmp_extract_dir.mkdir(parents=True, exist_ok=True)

        if zipfile.is_zipfile(archive_path):
            _safe_extract_zip(archive_path, tmp_extract_dir)
        elif tarfile.is_tarfile(archive_path):
            _safe_extract_tar(archive_path, tmp_extract_dir)
        else:
            raise ValueError(f"Unsupported archive type: {archive_path}")

        marker = {
            "repo_id": repo_id,
            "filename": filename,
            "archive_path": str(archive_path),
            "archive_size": archive_path.stat().st_size,
        }
        (tmp_extract_dir / marker_path.name).write_text(json.dumps(marker, indent=4), encoding="utf-8")
        tmp_extract_dir.replace(extract_dir)

    return _dataset_root(extract_dir)


def download_hf_archive_dataset(path: str, local_files_only: bool = False) -> str:
    repo_id, filename = parse_hf_archive_path(path)
    archive_path = Path(huggingface_hub.hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        local_files_only=local_files_only,
    ))

    safe_repo_name = repo_id.replace("/", "--")
    extract_dir = _archive_cache_root() / safe_repo_name / _archive_digest(repo_id, filename, archive_path)
    return str(_extract_archive(archive_path, extract_dir, repo_id, filename))


class DownloadHuggingfaceDatasets(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(
            self,
            concept_in_name: str,
            path_in_name: str,
            enabled_in_name: str,
            concept_out_name: str,
    ):
        super(DownloadHuggingfaceDatasets, self).__init__()

        self.concept_in_name = concept_in_name
        self.enabled_in_name = enabled_in_name
        self.path_in_name = path_in_name

        self.concept_out_name = concept_out_name

        self.concepts = []

    def length(self) -> int:
        return len(self.concepts)

    def get_inputs(self) -> list[str]:
        return [self.concept_in_name]

    def get_outputs(self) -> list[str]:
        return [self.concept_out_name]

    def start(self, variation: int):
        for in_index in range(self._get_previous_length(self.concept_in_name)):
            concept = self._get_previous_item(variation, self.concept_in_name, in_index)
            enabled = concept[self.enabled_in_name]
            if enabled:
                path = concept[self.path_in_name]

                is_local = os.path.isdir(path)
                if not is_local:
                    if is_hf_archive_path(path):
                        hf_path = download_hf_archive_dataset(path)
                        concept = concept.copy()
                        concept[self.path_in_name] = hf_path
                    else:
                        with suppress(huggingface_hub.errors.HFValidationError):
                            hf_path = huggingface_hub.snapshot_download(
                                repo_id=path,
                                repo_type="dataset",
                                max_workers=16,
                            )
                            concept = concept.copy()
                            concept[self.path_in_name] = hf_path

                self.concepts.append(concept)

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        return {
            self.concept_out_name: self.concepts[index],
        }
