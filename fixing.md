# Parquet Dataset Support Implementation

## Goal
Enable OneTrainer to work with HuggingFace parquet datasets on RunPod cloud training to bypass the 100k file limit and 2500 API requests/5min rate limit.

## What Was Implemented
- ✅ Parquet extraction pipeline module
- ✅ Parallel extraction (auto-detects CPU cores)
- ✅ All datasets converted to parquet format
- ✅ Integration with existing dataloader pipeline

## Files Modified

### 1. `external/mgds/src/mgds/pipelineModules/ExtractParquetDataset.py` (CREATED)
- Detects parquet datasets in HF cache (checks subdirectories for .parquet files)
- Uses `load_dataset('parquet', data_files=[...])` with explicit file list
- Parallel extraction with auto-detected CPU cores (max 16)
- Saves to cache: `/tmp/onetrainer_parquet_cache/{dataset_name}/` or config.cache_dir
- Uses `.extracted_complete` marker to skip re-extraction

### 2. `modules/dataLoader/mixin/DataLoaderText2ImageMixin.py` (MODIFIED)
- Added `ExtractParquetDataset` to pipeline between `DownloadHuggingfaceDatasets` and `CollectPaths`
- Module runs after downloading but before path collection

### 3. `requirements-global.txt` (MODIFIED)
- Added: `-e external/mgds` (editable install)
- Added: `datasets==3.2.0` (parquet support)

### 4. `lib.include.sh` (MODIFIED)
- Line 366: `export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_MGDS="0.1.0+parquet"`
- Needed because external/mgds doesn't have .git directory

### 5. `.gitignore` (MODIFIED)
- Changed `/external` to `/external/models`
- Allows committing external/mgds while ignoring model files

### 6. `scripts/convert_to_hf_dataset.py` (LOCAL ONLY)
- Converts local image+caption datasets to parquet format
- Uploads to HuggingFace as private datasets

## Key Commits (parquet-dataset-support branch)
- `1ee2fd8` - Detect HuggingFace dataset identifiers for parquet detection
- `953b01c` - Check subdirectories for parquet files
- `b690068` - Use data_files parameter to load parquet (handles symlinks)
- `19a328d` - Add parallel extraction for faster performance
- `83fc63a` - Auto-detect CPU cores for parallel extraction
- `57e8122` - Add more debug output to track extraction path updates

## Datasets Converted to Parquet
- `vgaggia/furryendall` (123k samples)
- `vgaggia/gelbooru1024` (38.3k samples)
- `vgaggia/newbig` (102k samples)
- `vgaggia/anime_small`
- `vgaggia/favorite_artists`

## RunPod Setup (IMPORTANT)

### Cloud Config Settings
In your OneTrainer config, set these cloud settings:

```
install_cmd: "git clone -b parquet-dataset-support https://github.com/vgaggia/OneTrainer"
update_onetrainer: false  # IMPORTANT: prevent git pull from switching to master
```

### Manual Setup (if already installed)
If you already have OneTrainer installed from upstream:

```bash
cd /workspace/OneTrainer
git remote set-url origin https://github.com/vgaggia/OneTrainer
git fetch origin
git checkout parquet-dataset-support
git pull origin parquet-dataset-support

# Reinstall mgds (IMPORTANT: use --no-cache-dir to avoid cached upstream version)
source venv/bin/activate
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_MGDS='0.1.0+parquet'
pip uninstall mgds -y
pip install -e ./external/mgds --no-cache-dir --force-reinstall
```

### Verify Installation
```bash
pip show mgds | grep Version
# Should show: Version: 0.1.0+parquet
# NOT: Version: 0.1.dev153+g50a2394c6
```

## Usage

1. Convert local dataset to parquet:
   ```bash
   python scripts/convert_to_hf_dataset.py --source "D:\path\to\dataset" --dataset-name "username/dataset-name"
   ```

2. In OneTrainer config, use HuggingFace dataset path (e.g., `vgaggia/gelbooru1024`)

3. ExtractParquetDataset automatically detects and extracts on first run

## Expected Training Output

```
Fetching N files: 100%|██████████| N/N [...]
[ExtractParquetDataset] Checking path: /workspace/huggingface_cache/hub/datasets--vgaggia--gelbooru1024/snapshots/[hash]
[ExtractParquetDataset] Is parquet dataset: True
Detected parquet dataset at /workspace/huggingface_cache/..., extracting images...
Found 24 parquet files
Dataset length: 38300
Extracting 38300 images to /tmp/onetrainer_parquet_cache/vgaggia_gelbooru1024...
  Using 16 parallel workers for extraction
  Extracted 1000/38300 images...
  ...
  Extraction complete: 38300/38300 images
[ExtractParquetDataset] Updated concept path to: /tmp/onetrainer_parquet_cache/vgaggia_gelbooru1024
enumerating sample paths: 100%|██████████| 38300/38300 [...]
step: 42it [01:23, 2.1s/it]  # Actual training steps
```
