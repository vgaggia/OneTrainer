from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import queue
import sys
import threading
import traceback
import zipfile
from dataclasses import dataclass
from pathlib import Path
from random import Random
from tkinter import filedialog, messagebox, ttk
import tkinter as tk

try:
    from PIL import Image, ImageOps, ImageTk
except Exception as exc:  # pragma: no cover - shown in the GUI path.
    Image = None
    ImageOps = None
    ImageTk = None
    PIL_IMPORT_ERROR = exc
else:
    PIL_IMPORT_ERROR = None


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRESET = REPO_ROOT / "training_presets" / "krea2 FT RTX 5090 Cloud.json"
BASE_SEED = 42

IMAGE_EXTENSIONS = {
    ".apng",
    ".bmp",
    ".gif",
    ".jfif",
    ".jpe",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
ARCHIVE_EXTENSIONS = (".zip",)


@dataclass(frozen=True)
class Sample:
    path: str
    rel_path: str
    width: int
    height: int
    caption_exists: bool
    source_kind: str
    source_root: str


@dataclass(frozen=True)
class OrderedItem:
    global_step: int
    epoch: int
    epoch_step: int
    slot: int
    sample_index: int
    bucket: tuple[int, ...]
    sample: Sample


@dataclass(frozen=True)
class ModuleIndices:
    aspect_bucket_index: int
    aspect_sort_index: int
    note: str


def resolve_repo_path(value: str | Path, base: Path = REPO_ROOT) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    if not path.is_absolute():
        path = base / path
    return path


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_error(message: str):
    print(message, file=sys.stderr)


def is_krea_config(config: dict) -> bool:
    return str(config.get("model_type", "")).upper() == "KREA_2"


def model_has_mask_input(config: dict) -> bool:
    model_type = str(config.get("model_type", "")).upper()
    return model_type in {
        "STABLE_DIFFUSION_15_INPAINTING",
        "STABLE_DIFFUSION_20_INPAINTING",
        "STABLE_DIFFUSION_XL_10_BASE_INPAINTING",
        "FLUX_FILL_DEV_1",
    }


def model_has_conditioning_image_input(config: dict) -> bool:
    return model_has_mask_input(config)


def model_has_depth_input(config: dict) -> bool:
    return str(config.get("model_type", "")).upper() == "STABLE_DIFFUSION_20_DEPTH"


def model_has_multiple_text_encoders(config: dict) -> bool:
    model_type = str(config.get("model_type", "")).upper()
    return model_type in {
        "STABLE_DIFFUSION_3",
        "STABLE_DIFFUSION_35",
        "STABLE_DIFFUSION_XL_10_BASE",
        "STABLE_DIFFUSION_XL_10_BASE_INPAINTING",
        "FLUX_DEV_1",
        "FLUX_FILL_DEV_1",
        "FLUX_2",
        "HUNYUAN_VIDEO",
        "HI_DREAM_FULL",
    }


def train_any_embedding(config: dict) -> bool:
    embedding = config.get("embedding") or {}
    training_method = str(config.get("training_method", "")).upper()
    if training_method == "EMBEDDING" and not bool(embedding.get("is_output_embedding", False)):
        return True
    for extra in config.get("additional_embeddings") or []:
        if bool(extra.get("train", False)) and not bool(extra.get("is_output_embedding", False)):
            return True
    return False


def train_text_encoder_or_embedding(config: dict) -> bool:
    text_encoder = config.get("text_encoder") or {}
    embedding = config.get("embedding") or {}
    training_method = str(config.get("training_method", "")).upper()

    trains_encoder = (
        bool(text_encoder.get("train", False))
        and training_method != "EMBEDDING"
        and not bool(embedding.get("is_output_embedding", False))
    )
    trains_embedding_through_encoder = (
        bool(text_encoder.get("train_embedding", False)) or not model_has_multiple_text_encoders(config)
    ) and train_any_embedding(config)
    return trains_encoder or trains_embedding_through_encoder


def compute_module_indices(config: dict) -> ModuleIndices:
    """Mirror the relevant Krea data-loader module count.

    MGDS uses hash((base_seed, module_index, variation, index)) for random order.
    The Krea cloud preset currently puts AspectBatchSorting at index 35.
    """
    index = 2  # ConceptPipelineModule + SettingsPipelineModule

    masked = bool(config.get("masked_training", False)) or model_has_mask_input(config)
    custom_conditioning = bool(config.get("custom_conditioning_image", False))
    conditioning_model = model_has_conditioning_image_input(config)
    depth_model = model_has_depth_input(config)
    latent_caching = bool(config.get("latent_caching", False))
    train_text = train_text_encoder_or_embedding(config)
    krea = is_krea_config(config)
    vae_frame_dim = krea

    # _enumerate_input_modules
    index += 3  # DownloadHuggingfaceDatasets, CollectPaths, sample_prompt_path
    if masked:
        index += 1
    if custom_conditioning:
        index += 1

    # _load_input_modules
    index += 2  # LoadImage, LoadVideo
    if vae_frame_dim:
        index += 1  # ImageToVideo
    index += 5  # text loading/select modules
    if masked:
        index += 1  # generated mask
    if custom_conditioning:
        index += 1
    if vae_frame_dim and masked:
        index += 1  # mask_to_video

    # _mask_augmentation_modules
    if masked:
        index += 2

    # _aspect_bucketing_in
    index += 1  # CalcAspect
    aspect_bucket_index = index
    index += 1  # AspectBucketing or SingleAspectCalculation

    # _crop_modules
    index += 1

    # _augmentation_modules
    index += 9

    # _inpainting_modules
    if conditioning_model:
        index += 2

    # Krea _preparation_modules. Non-Krea is still useful, but less exact.
    if krea:
        index += 3  # RescaleImageChannels, EncodeVAE, SampleVAEDistribution
        if masked:
            index += 1
        index += 1  # Tokenize
        if not train_text:
            index += 1  # EncodeQwenText
        if latent_caching and not train_text:
            index += 1  # PruneMaskedTokens
    else:
        index += 1

    # Krea _cache_modules_from_names.
    if latent_caching:
        index += 1  # image DiskCache
        if not train_text:
            index += 1  # text DiskCache

    # VariationSorting remains in the Krea path because prompt/concept are still sorted.
    index += 1

    # Krea _output_modules prepends PadMaskedTokens before AspectBatchSorting.
    if krea and latent_caching and not train_text:
        index += 1

    note = "exact for current KREA_2 data-loader path" if krea else "approximate for non-KREA_2 configs"
    return ModuleIndices(aspect_bucket_index=aspect_bucket_index, aspect_sort_index=index, note=note)


def parse_concepts(concept_path: Path) -> list[dict]:
    raw = read_json(concept_path)
    concepts = raw if isinstance(raw, list) else [raw]
    return [
        concept
        for concept in concepts
        if bool(concept.get("enabled", True))
        and str(concept.get("type", "STANDARD")).upper() != "VALIDATION"
    ]


def list_folder_files(root: Path, include_subdirectories: bool) -> list[Path]:
    def walk(path: Path) -> list[Path]:
        entries = [path / name for name in os.listdir(path)]
        files = [entry for entry in entries if entry.is_file()]
        if include_subdirectories:
            for subdir in [entry for entry in entries if entry.is_dir()]:
                if subdir.name.startswith("."):
                    continue
                files.extend(walk(subdir))
        return files

    files = walk(root)
    files = [
        path
        for path in files
        if path.suffix.lower() in IMAGE_EXTENSIONS
        and not path.with_suffix("").name.endswith(("-masklabel", "-condlabel"))
    ]
    return sorted(files, key=lambda path: str(path))


def visible_zip_prefix(names: list[str]) -> str:
    visible = [name for name in names if name and not name.split("/", 1)[0].startswith(".")]
    top_levels = {name.split("/", 1)[0] for name in visible if "/" in name}
    root_files = [name for name in visible if "/" not in name]
    if len(top_levels) == 1 and not root_files:
        return next(iter(top_levels)) + "/"
    return ""


def list_zip_images(zip_path: Path) -> tuple[list[str], set[str], str]:
    with zipfile.ZipFile(zip_path) as archive:
        names = [info.filename.replace("\\", "/") for info in archive.infolist() if not info.is_dir()]
    prefix = visible_zip_prefix(names)
    image_names = []
    text_names = set()
    for name in names:
        rel = name[len(prefix):] if prefix and name.startswith(prefix) else name
        if not rel or rel.startswith("."):
            continue
        suffix = Path(rel).suffix.lower()
        if suffix == ".txt":
            text_names.add(rel)
        elif suffix in IMAGE_EXTENSIONS and not Path(rel).with_suffix("").name.endswith(("-masklabel", "-condlabel")):
            image_names.append(rel)
    return sorted(image_names), text_names, prefix


def image_size_from_folder(path: Path) -> tuple[int, int]:
    if Image is None:
        raise RuntimeError(f"Pillow failed to import: {PIL_IMPORT_ERROR}")
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        return image.size


def image_size_from_zip(archive: zipfile.ZipFile, member: str, prefix: str) -> tuple[int, int]:
    if Image is None:
        raise RuntimeError(f"Pillow failed to import: {PIL_IMPORT_ERROR}")
    name = prefix + member
    with archive.open(name) as handle:
        with Image.open(handle) as image:
            image = ImageOps.exif_transpose(image)
            return image.size


def resolve_hf_archive_local(path: str, allow_download: bool) -> Path:
    sys.path.insert(0, str(REPO_ROOT))
    from modules.dataLoader.pipelineModules.DownloadHuggingfaceDatasets import download_hf_archive_dataset

    return Path(download_hf_archive_dataset(path, local_files_only=not allow_download))


def resolve_hf_dataset_local(path: str, allow_download: bool) -> Path:
    import huggingface_hub

    return Path(huggingface_hub.snapshot_download(repo_id=path, repo_type="dataset", local_files_only=not allow_download))


def resolve_concept_source(concept: dict, concept_file: Path, override: str, allow_download: bool) -> tuple[str, Path]:
    if override.strip():
        path = resolve_repo_path(override.strip(), concept_file.parent)
        if path.is_file() and path.suffix.lower() in ARCHIVE_EXTENSIONS:
            return "zip", path
        if path.is_dir():
            return "folder", path
        raise FileNotFoundError(f"Dataset override does not exist or is not supported: {path}")

    raw_path = str(concept.get("path", "")).strip()
    if raw_path.startswith("hf-archive:"):
        try:
            return "folder", resolve_hf_archive_local(raw_path, allow_download)
        except Exception as exc:
            raise RuntimeError(
                "The concept uses hf-archive, but the archive is not available in the local HF cache. "
                "Tick 'Allow HF download' to download/extract it from Hugging Face, or use the "
                "Dataset override field to select the extracted folder or the .zip file. "
                f"Original error: {exc}"
            ) from exc

    local_path = resolve_repo_path(raw_path, concept_file.parent)
    if local_path.is_file() and local_path.suffix.lower() in ARCHIVE_EXTENSIONS:
        return "zip", local_path
    if local_path.is_dir():
        return "folder", local_path

    if "/" in raw_path and "\\" not in raw_path and ":" not in raw_path:
        try:
            return "folder", resolve_hf_dataset_local(raw_path, allow_download)
        except Exception as exc:
            raise RuntimeError(
                "The concept looks like a Hugging Face dataset repo, but it is not present in the local HF cache. "
                "Tick 'Allow HF download' to download it, or use Dataset override to select a local copy. "
                f"Original error: {exc}"
            ) from exc

    raise FileNotFoundError(f"Could not resolve concept path: {raw_path}")


def quantize_resolution(resolution: tuple[float, float], quantization: int) -> tuple[int, int]:
    return (
        round(resolution[0] / quantization) * quantization,
        round(resolution[1] / quantization) * quantization,
    )


def automatic_buckets(target_resolutions: list[int], quantization: int) -> tuple[dict[int, list[tuple[int, int]]], dict[int, list[float]]]:
    all_possible_input_aspects = [
        (1.0, 1.0),
        (1.0, 1.25),
        (1.0, 1.5),
        (1.0, 1.75),
        (1.0, 2.0),
        (1.0, 2.5),
        (1.0, 3.0),
        (1.0, 3.5),
        (1.0, 4.0),
    ]
    possible_resolutions: dict[int, list[tuple[int, int]]] = {}
    possible_aspects: dict[int, list[float]] = {}
    for target_resolution in target_resolutions:
        new_resolutions = [
            (
                h / math.sqrt(h * w) * target_resolution,
                w / math.sqrt(h * w) * target_resolution,
            )
            for (h, w) in all_possible_input_aspects
        ]
        new_resolutions = new_resolutions + [(w, h) for (h, w) in new_resolutions]
        quantized = [quantize_resolution(resolution, quantization) for resolution in new_resolutions]
        unique = list(set(quantized))
        possible_resolutions[target_resolution] = unique
        possible_aspects[target_resolution] = [h / w for (h, w) in unique]
    return possible_resolutions, possible_aspects


def parse_resolution_values(value: str) -> list[int]:
    return [int(part.strip()) for part in str(value).split(",") if part.strip()]


def crop_resolution_for_sample(
    sample: Sample,
    concept: dict,
    config: dict,
    indices: ModuleIndices,
    epoch: int,
    sample_index: int,
) -> tuple[int, ...]:
    resolution = (1, sample.height, sample.width) if is_krea_config(config) else (sample.height, sample.width)
    target_spec = str(config.get("resolution", "512"))
    image_config = concept.get("image") or {}
    if image_config.get("enable_resolution_override", False):
        target_spec = str(image_config.get("resolution_override", target_spec))

    quantization = 64 if is_krea_config(config) else 64

    if bool(config.get("aspect_ratio_bucketing", False)):
        if "x" in target_spec and "," not in target_spec:
            width, height = [int(part.strip()) for part in target_spec.split("x", 1)]
            target_resolution = quantize_resolution((height, width), quantization)
        else:
            target_resolutions = parse_resolution_values(target_spec)
            rand = Random(hash((BASE_SEED, indices.aspect_bucket_index, epoch, sample_index)))
            target_base = rand.choice(target_resolutions)
            bucket_resolutions, bucket_aspects = automatic_buckets(target_resolutions, quantization)
            aspect = resolution[-2] / resolution[-1]
            aspects = bucket_aspects[target_base]
            bucket_index = min(range(len(aspects)), key=lambda i: abs(aspects[i] - aspect))
            target_resolution = bucket_resolutions[target_base][bucket_index]

        return (*resolution[:-2], *target_resolution)

    target = parse_resolution_values(target_spec)[0]
    return (target, target)


def build_samples(
    config: dict,
    concept_path: Path,
    dataset_override: str,
    allow_download: bool,
    progress,
) -> tuple[list[Sample], dict]:
    concepts = parse_concepts(concept_path)
    if not concepts:
        raise RuntimeError("No enabled non-validation concepts found.")
    if len(concepts) > 1:
        progress(f"Using {len(concepts)} enabled concepts")

    all_samples: list[Sample] = []
    source_summary = {}

    for concept_index, concept in enumerate(concepts):
        kind, source = resolve_concept_source(
            concept,
            concept_path,
            dataset_override if concept_index == 0 else "",
            allow_download,
        )
        include_subdirs = bool(concept.get("include_subdirectories", True))
        progress(f"Enumerating {concept.get('name', source.name)} from {source}")

        if kind == "folder":
            image_paths = list_folder_files(source, include_subdirs)
            for idx, path in enumerate(image_paths):
                if idx % 100 == 0:
                    progress(f"Reading image headers {idx}/{len(image_paths)}")
                width, height = image_size_from_folder(path)
                rel = os.path.relpath(path, source)
                caption = path.with_suffix(".txt").is_file()
                all_samples.append(Sample(str(path), rel, width, height, caption, "folder", str(source)))
            source_summary[str(source)] = len(image_paths)
        elif kind == "zip":
            image_names, text_names, prefix = list_zip_images(source)
            with zipfile.ZipFile(source) as archive:
                for idx, member in enumerate(image_names):
                    if idx % 100 == 0:
                        progress(f"Reading archive image headers {idx}/{len(image_names)}")
                    width, height = image_size_from_zip(archive, member, prefix)
                    caption = str(Path(member).with_suffix(".txt")).replace("\\", "/") in text_names
                    all_samples.append(Sample(member, member, width, height, caption, "zip", str(source)))
            source_summary[str(source)] = len(image_names)
        else:
            raise RuntimeError(f"Unsupported source kind: {kind}")

    return all_samples, source_summary


def build_epoch_order(
    samples: list[Sample],
    concept: dict,
    config: dict,
    batch_size: int,
    world_size: int,
    rank: int,
    epoch: int,
    indices: ModuleIndices,
) -> tuple[list[int], dict[int, tuple[int, ...]], dict]:
    effective_batch_size = batch_size * world_size
    bucket_dict: dict[tuple[int, ...], list[int]] = {}
    sample_buckets: dict[int, tuple[int, ...]] = {}

    for sample_index, sample in enumerate(samples):
        bucket = crop_resolution_for_sample(sample, concept, config, indices, epoch, sample_index)
        bucket_dict.setdefault(bucket, []).append(sample_index)
        sample_buckets[sample_index] = bucket

    rand = Random(hash((BASE_SEED, indices.aspect_sort_index, epoch, -1)))
    work_buckets = {key: value.copy() for key, value in bucket_dict.items()}
    batches: list[tuple[tuple[int, ...], int]] = []
    dropped_by_bucket: dict[tuple[int, ...], int] = {}

    for bucket_key, bucket_samples in work_buckets.items():
        batch_count = int(len(bucket_samples) / effective_batch_size)
        batches.extend((bucket_key, i) for i in range(batch_count))

    rand.shuffle(batches)

    for bucket_key, bucket_samples in work_buckets.items():
        rand.shuffle(bucket_samples)

    for bucket_key, bucket_samples in work_buckets.items():
        samples_to_drop = len(bucket_samples) % effective_batch_size
        dropped_by_bucket[bucket_key] = samples_to_drop
        for _ in range(samples_to_drop):
            bucket_samples.pop()

    global_order: list[int] = []
    for bucket_key, bucket_index in batches:
        for i in range(bucket_index * effective_batch_size, (bucket_index + 1) * effective_batch_size):
            global_order.append(work_buckets[bucket_key][i])

    if world_size > 1:
        per_rank_length = len(global_order) // world_size
        rank_order = [global_order[(i * world_size) + rank] for i in range(per_rank_length)]
    else:
        rank_order = global_order

    stats = {
        "bucket_count": len(bucket_dict),
        "effective_samples": len(global_order),
        "rank_samples": len(rank_order),
        "steps": len(rank_order) // batch_size,
        "dropped": sum(dropped_by_bucket.values()),
        "dropped_by_bucket": dropped_by_bucket,
        "effective_batch_size": effective_batch_size,
    }
    return rank_order, sample_buckets, stats


def global_step_for_epoch(
    samples: list[Sample],
    concept: dict,
    config: dict,
    batch_size: int,
    world_size: int,
    rank: int,
    epoch: int,
    epoch_step: int,
    indices: ModuleIndices,
) -> int:
    total = 0
    for prior_epoch in range(epoch):
        rank_order, _, _ = build_epoch_order(samples, concept, config, batch_size, world_size, rank, prior_epoch, indices)
        total += len(rank_order) // batch_size
    return total + epoch_step


def make_ordered_rows(
    samples: list[Sample],
    concept: dict,
    config: dict,
    batch_size: int,
    world_size: int,
    rank: int,
    epoch: int,
    start_step: int,
    step_count: int,
    indices: ModuleIndices,
) -> tuple[list[OrderedItem], dict]:
    rank_order, sample_buckets, stats = build_epoch_order(samples, concept, config, batch_size, world_size, rank, epoch, indices)
    max_steps = len(rank_order) // batch_size
    if start_step >= max_steps:
        raise ValueError(f"Start step {start_step} is outside this epoch. Max step is {max_steps - 1}.")
    first_global_step = global_step_for_epoch(samples, concept, config, batch_size, world_size, rank, epoch, start_step, indices)

    rows: list[OrderedItem] = []
    for local_offset, step in enumerate(range(start_step, min(start_step + step_count, max_steps))):
        for slot in range(batch_size):
            order_index = (step * batch_size) + slot
            sample_index = rank_order[order_index]
            rows.append(
                OrderedItem(
                    global_step=first_global_step + local_offset,
                    epoch=epoch,
                    epoch_step=step,
                    slot=slot,
                    sample_index=sample_index,
                    bucket=sample_buckets[sample_index],
                    sample=samples[sample_index],
                )
            )
    return rows, stats


def read_caption(sample: Sample) -> str:
    caption_rel = str(Path(sample.rel_path).with_suffix(".txt")).replace("\\", "/")
    if sample.source_kind == "folder":
        caption_path = Path(sample.source_root) / caption_rel
        if not caption_path.is_file():
            return ""
        try:
            return caption_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return caption_path.read_text(encoding="utf-8", errors="replace")

    if sample.source_kind == "zip":
        with zipfile.ZipFile(sample.source_root) as archive:
            names = [info.filename.replace("\\", "/") for info in archive.infolist() if not info.is_dir()]
            prefix = visible_zip_prefix(names)
            member = prefix + caption_rel
            try:
                data = archive.read(member)
            except KeyError:
                return ""
            return data.decode("utf-8", errors="replace")

    return ""


def load_preview_image(sample: Sample, max_size: tuple[int, int]) -> ImageTk.PhotoImage:
    if Image is None or ImageTk is None:
        raise RuntimeError(f"Pillow failed to import: {PIL_IMPORT_ERROR}")

    if sample.source_kind == "folder":
        image = Image.open(sample.path)
    elif sample.source_kind == "zip":
        with zipfile.ZipFile(sample.source_root) as archive:
            names = [info.filename.replace("\\", "/") for info in archive.infolist() if not info.is_dir()]
            prefix = visible_zip_prefix(names)
            with archive.open(prefix + sample.rel_path) as handle:
                image = Image.open(io.BytesIO(handle.read()))
    else:
        raise RuntimeError(f"Unsupported source kind: {sample.source_kind}")

    image = ImageOps.exif_transpose(image).convert("RGB")
    image.thumbnail(max_size)
    return ImageTk.PhotoImage(image)


class BatchInspectorApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("OneTrainer Batch Inspector")
        self.root.geometry("1420x860")

        self.task_queue: queue.Queue = queue.Queue()
        self.samples: list[Sample] = []
        self.config: dict | None = None
        self.concepts: list[dict] = []
        self.indices: ModuleIndices | None = None
        self.current_rows: list[OrderedItem] = []
        self.preview_image = None

        self.preset_var = tk.StringVar(value=str(DEFAULT_PRESET))
        self.concept_var = tk.StringVar(value="")
        self.override_var = tk.StringVar(value="")
        self.allow_download_var = tk.BooleanVar(value=False)
        self.batch_var = tk.StringVar(value="8")
        self.world_var = tk.StringVar(value="1")
        self.rank_var = tk.StringVar(value="0")
        self.epoch_var = tk.StringVar(value="0")
        self.step_var = tk.StringVar(value="0")
        self.count_var = tk.StringVar(value="1")
        self.status_var = tk.StringVar(value="Load a preset, then build the order.")
        self.summary_var = tk.StringVar(value="")

        self._build_layout()
        self._load_preset_fields()
        self.root.after(100, self._poll_queue)

    def _build_layout(self):
        root = self.root
        outer = ttk.Frame(root, padding=8)
        outer.pack(fill=tk.BOTH, expand=True)

        top = ttk.LabelFrame(outer, text="Config")
        top.pack(fill=tk.X)

        def add_path_row(row: int, label: str, var: tk.StringVar, command):
            ttk.Label(top, text=label).grid(row=row, column=0, sticky="w", padx=(8, 4), pady=4)
            ttk.Entry(top, textvariable=var).grid(row=row, column=1, sticky="ew", padx=4, pady=4)
            ttk.Button(top, text="Browse", command=command).grid(row=row, column=2, padx=4, pady=4)

        top.columnconfigure(1, weight=1)
        add_path_row(0, "Preset JSON", self.preset_var, self._browse_preset)
        add_path_row(1, "Concept JSON", self.concept_var, self._browse_concept)
        add_path_row(2, "Dataset override folder/zip", self.override_var, self._browse_dataset)
        ttk.Checkbutton(
            top,
            text="Allow HF download/extract if archive is not cached",
            variable=self.allow_download_var,
        ).grid(row=3, column=1, sticky="w", padx=4, pady=(2, 0))

        controls = ttk.Frame(top)
        controls.grid(row=4, column=0, columnspan=3, sticky="ew", padx=8, pady=6)
        for col, (label, var, width) in enumerate(
            [
                ("Batch", self.batch_var, 6),
                ("World", self.world_var, 6),
                ("Rank", self.rank_var, 6),
                ("Epoch", self.epoch_var, 6),
                ("Start step", self.step_var, 8),
                ("Steps", self.count_var, 6),
            ]
        ):
            ttk.Label(controls, text=label).grid(row=0, column=col * 2, padx=(0, 4))
            ttk.Entry(controls, textvariable=var, width=width).grid(row=0, column=(col * 2) + 1, padx=(0, 12))

        ttk.Button(controls, text="Reload Preset Fields", command=self._load_preset_fields).grid(row=0, column=12, padx=6)
        ttk.Button(controls, text="Build Order", command=self._start_build).grid(row=0, column=13, padx=6)
        ttk.Button(controls, text="Show Step", command=self._show_step).grid(row=0, column=14, padx=6)
        ttk.Button(controls, text="Export CSV", command=self._export_csv).grid(row=0, column=15, padx=6)

        ttk.Label(outer, textvariable=self.summary_var, anchor="w").pack(fill=tk.X, pady=(6, 0))
        ttk.Label(outer, textvariable=self.status_var, anchor="w").pack(fill=tk.X, pady=(2, 8))

        main = ttk.PanedWindow(outer, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(main)
        right = ttk.Frame(main)
        main.add(left, weight=3)
        main.add(right, weight=2)

        columns = ("global", "epoch", "step", "slot", "bucket", "size", "caption", "file")
        self.tree = ttk.Treeview(left, columns=columns, show="headings", height=24)
        for column, heading, width in [
            ("global", "Global", 70),
            ("epoch", "Epoch", 60),
            ("step", "Step", 70),
            ("slot", "Slot", 50),
            ("bucket", "Bucket", 125),
            ("size", "Image", 90),
            ("caption", "Txt", 45),
            ("file", "File", 620),
        ]:
            self.tree.heading(column, text=heading)
            self.tree.column(column, width=width, anchor="w")
        y_scroll = ttk.Scrollbar(left, orient=tk.VERTICAL, command=self.tree.yview)
        x_scroll = ttk.Scrollbar(left, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)
        self.tree.bind("<<TreeviewSelect>>", self._on_select)

        preview_frame = ttk.LabelFrame(right, text="Preview")
        preview_frame.pack(fill=tk.BOTH, expand=True)
        self.preview_label = ttk.Label(preview_frame, anchor="center")
        self.preview_label.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        caption_frame = ttk.LabelFrame(right, text="Caption")
        caption_frame.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        self.caption_text = tk.Text(caption_frame, height=9, wrap=tk.WORD)
        self.caption_text.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

    def _browse_preset(self):
        path = filedialog.askopenfilename(
            initialdir=str(REPO_ROOT / "training_presets"),
            title="Select OneTrainer preset",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if path:
            self.preset_var.set(path)
            self._load_preset_fields()

    def _browse_concept(self):
        path = filedialog.askopenfilename(
            initialdir=str(REPO_ROOT / "training_concepts"),
            title="Select concept JSON",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if path:
            self.concept_var.set(path)

    def _browse_dataset(self):
        path = filedialog.askdirectory(title="Select extracted dataset folder")
        if not path:
            path = filedialog.askopenfilename(
                title="Or select dataset zip",
                filetypes=[("Zip archives", "*.zip"), ("All files", "*.*")],
            )
        if path:
            self.override_var.set(path)

    def _load_preset_fields(self):
        try:
            preset_path = resolve_repo_path(self.preset_var.get())
            config = read_json(preset_path)
            concept_name = config.get("concept_file_name", "")
            if concept_name:
                self.concept_var.set(str(resolve_repo_path(concept_name, REPO_ROOT)))
            self.batch_var.set(str(config.get("batch_size", self.batch_var.get())))
            self.world_var.set("1")
            self.rank_var.set("0")
            self.config = config
            self.indices = compute_module_indices(config)
            self.status_var.set(
                f"Preset loaded. Aspect sort module index: {self.indices.aspect_sort_index}. "
                "Select a dataset override or tick Allow HF download before Build Order if the archive is not cached."
            )
        except Exception as exc:
            messagebox.showerror("Preset error", str(exc))

    def _read_inputs(self):
        preset_path = resolve_repo_path(self.preset_var.get())
        concept_path = resolve_repo_path(self.concept_var.get())
        config = read_json(preset_path)
        batch_size = int(self.batch_var.get())
        world_size = int(self.world_var.get())
        rank = int(self.rank_var.get())
        epoch = int(self.epoch_var.get())
        start_step = int(self.step_var.get())
        step_count = int(self.count_var.get())
        allow_download = bool(self.allow_download_var.get())
        if batch_size < 1:
            raise ValueError("Batch size must be >= 1")
        if world_size < 1:
            raise ValueError("World size must be >= 1")
        if rank < 0 or rank >= world_size:
            raise ValueError("Rank must be in [0, world_size)")
        if epoch < 0 or start_step < 0 or step_count < 1:
            raise ValueError("Epoch/start step/steps must be non-negative, with steps >= 1")
        concepts = parse_concepts(concept_path)
        if not concepts:
            raise ValueError("Concept JSON has no enabled non-validation concepts.")
        return config, concept_path, concepts, batch_size, world_size, rank, epoch, start_step, step_count, allow_download

    def _start_build(self):
        try:
            config, concept_path, concepts, batch_size, world_size, rank, epoch, start_step, step_count, allow_download = self._read_inputs()
        except Exception as exc:
            messagebox.showerror("Input error", str(exc))
            return

        self.status_var.set("Building sample order...")
        self.summary_var.set("")
        self.samples = []
        self.current_rows = []
        for item in self.tree.get_children():
            self.tree.delete(item)

        def worker():
            try:
                def progress(text):
                    self.task_queue.put(("status", text))

                samples, source_summary = build_samples(
                    config,
                    concept_path,
                    self.override_var.get(),
                    allow_download,
                    progress,
                )
                indices = compute_module_indices(config)
                rows, stats = make_ordered_rows(
                    samples,
                    concepts[0],
                    config,
                    batch_size,
                    world_size,
                    rank,
                    epoch,
                    start_step,
                    step_count,
                    indices,
                )
                self.task_queue.put(("done", config, concepts, indices, samples, rows, stats, source_summary))
            except Exception:
                self.task_queue.put(("error", traceback.format_exc()))

        threading.Thread(target=worker, daemon=True).start()

    def _poll_queue(self):
        try:
            while True:
                message = self.task_queue.get_nowait()
                kind = message[0]
                if kind == "status":
                    self.status_var.set(message[1])
                elif kind == "done":
                    _, config, concepts, indices, samples, rows, stats, source_summary = message
                    self.config = config
                    self.concepts = concepts
                    self.indices = indices
                    self.samples = samples
                    self.current_rows = rows
                    self._populate_rows(rows)
                    summary = (
                        f"samples={len(samples)} | effective_samples/epoch={stats['effective_samples']} | "
                        f"steps/epoch={stats['steps']} | dropped={stats['dropped']} | "
                        f"buckets={stats['bucket_count']} | sort_batch={stats['effective_batch_size']} | "
                        f"aspect_sort_module={indices.aspect_sort_index} ({indices.note})"
                    )
                    self.summary_var.set(summary)
                    self.status_var.set(f"Loaded sources: {source_summary}")
                elif kind == "error":
                    self.status_var.set("Build failed.")
                    messagebox.showerror("Build failed", message[1])
        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)

    def _show_step(self):
        if not self.samples or not self.config or not self.concepts or not self.indices:
            self._start_build()
            return
        try:
            _, _, _, batch_size, world_size, rank, epoch, start_step, step_count, _allow_download = self._read_inputs()
            rows, stats = make_ordered_rows(
                self.samples,
                self.concepts[0],
                self.config,
                batch_size,
                world_size,
                rank,
                epoch,
                start_step,
                step_count,
                self.indices,
            )
            self.current_rows = rows
            self._populate_rows(rows)
            self.summary_var.set(
                f"samples={len(self.samples)} | effective_samples/epoch={stats['effective_samples']} | "
                f"steps/epoch={stats['steps']} | dropped={stats['dropped']} | buckets={stats['bucket_count']} | "
                f"sort_batch={stats['effective_batch_size']} | aspect_sort_module={self.indices.aspect_sort_index}"
            )
        except Exception as exc:
            messagebox.showerror("Step error", str(exc))

    def _populate_rows(self, rows: list[OrderedItem]):
        for item in self.tree.get_children():
            self.tree.delete(item)
        for idx, row in enumerate(rows):
            sample = row.sample
            self.tree.insert(
                "",
                "end",
                iid=str(idx),
                values=(
                    row.global_step,
                    row.epoch,
                    row.epoch_step,
                    row.slot,
                    "x".join(str(x) for x in row.bucket),
                    f"{sample.width}x{sample.height}",
                    "yes" if sample.caption_exists else "no",
                    sample.rel_path,
                ),
            )

    def _on_select(self, _event=None):
        selected = self.tree.selection()
        if not selected:
            return
        row = self.current_rows[int(selected[0])]
        sample = row.sample
        self.caption_text.delete("1.0", tk.END)
        header = (
            f"global_step={row.global_step} epoch={row.epoch} step={row.epoch_step} "
            f"slot={row.slot} bucket={row.bucket}\n{sample.path}\n\n"
        )
        self.caption_text.insert(tk.END, header + read_caption(sample))
        try:
            self.preview_image = load_preview_image(sample, (640, 430))
            self.preview_label.configure(image=self.preview_image, text="")
        except Exception as exc:
            self.preview_image = None
            self.preview_label.configure(image="", text=f"Preview failed:\n{exc}")

    def _export_csv(self):
        if not self.current_rows:
            messagebox.showinfo("Export CSV", "No rows to export.")
            return
        path = filedialog.asksaveasfilename(
            title="Export batch rows",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "global_step",
                "epoch",
                "epoch_step",
                "slot",
                "sample_index",
                "bucket",
                "width",
                "height",
                "caption_exists",
                "source_kind",
                "source_root",
                "path",
                "rel_path",
            ])
            for row in self.current_rows:
                sample = row.sample
                writer.writerow([
                    row.global_step,
                    row.epoch,
                    row.epoch_step,
                    row.slot,
                    row.sample_index,
                    "x".join(str(x) for x in row.bucket),
                    sample.width,
                    sample.height,
                    sample.caption_exists,
                    sample.source_kind,
                    sample.source_root,
                    sample.path,
                    sample.rel_path,
                ])
        self.status_var.set(f"Exported {len(self.current_rows)} rows to {path}")


def self_test() -> int:
    config = read_json(DEFAULT_PRESET)
    indices = compute_module_indices(config)
    concept = resolve_repo_path(config.get("concept_file_name", "training_concepts/concepts.json"))
    concepts = parse_concepts(concept)
    print(f"preset={DEFAULT_PRESET}")
    print(f"concept={concept}")
    print(f"batch_size={config.get('batch_size')} resolution={config.get('resolution')}")
    print(f"concepts={len(concepts)}")
    print(f"aspect_bucket_index={indices.aspect_bucket_index}")
    print(f"aspect_sort_index={indices.aspect_sort_index}")
    print(f"note={indices.note}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect OneTrainer image batch order.")
    parser.add_argument("--self-test", action="store_true", help="Load the default preset and print computed indices.")
    args = parser.parse_args()
    if args.self_test:
        return self_test()

    root = tk.Tk()
    BatchInspectorApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
