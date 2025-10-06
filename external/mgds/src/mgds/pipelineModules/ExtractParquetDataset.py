import os
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule
from contextlib import suppress


class ExtractParquetDataset(
    PipelineModule,
    RandomAccessPipelineModule,
):
    """
    Extracts images from HuggingFace parquet datasets to a temporary directory.
    This allows OneTrainer to work with parquet-based datasets that bypass the 100k file limit.

    If the path contains parquet files, this module:
    1. Loads the dataset using the datasets library
    2. Extracts images to a cache directory
    3. Creates corresponding txt files with captions
    4. Updates the concept path to point to the extracted files

    If the path doesn't contain parquet files, it passes through unchanged.
    """

    def __init__(
            self,
            concept_in_name: str,
            path_in_name: str,
            enabled_in_name: str,
            concept_out_name: str,
            cache_dir: str | None = None,
    ):
        super(ExtractParquetDataset, self).__init__()

        self.concept_in_name = concept_in_name
        self.path_in_name = path_in_name
        self.enabled_in_name = enabled_in_name
        self.concept_out_name = concept_out_name
        self.cache_dir = cache_dir

        self.concepts = []

    def length(self) -> int:
        return len(self.concepts)

    def get_inputs(self) -> list[str]:
        return [self.concept_in_name]

    def get_outputs(self) -> list[str]:
        return [self.concept_out_name]

    def __is_parquet_dataset(self, path: str) -> bool:
        """Check if the path is a HuggingFace dataset or contains parquet files"""
        # Check if it's a HuggingFace dataset identifier (format: username/dataset-name)
        if '/' in path and not os.path.exists(path):
            # Likely a HuggingFace dataset identifier
            return True

        if not os.path.isdir(path):
            return False

        # Check for parquet files in the directory or immediate subdirectories
        for item in os.listdir(path):
            if item.endswith('.parquet'):
                return True
            item_path = os.path.join(path, item)
            if os.path.isdir(item_path):
                for subfile in os.listdir(item_path):
                    if subfile.endswith('.parquet'):
                        return True
        return False

    def __extract_parquet_dataset(self, path: str, concept: dict) -> str:
        """
        Extract images from parquet dataset to a cache directory.
        Returns the path to the extracted files.
        """
        try:
            from datasets import load_dataset
        except ImportError:
            print("Warning: 'datasets' library not found. Cannot extract parquet datasets.")
            print("Install with: pip install datasets")
            return path

        print(f"Detected parquet dataset at {path}, extracting images...")

        # Load the dataset
        try:
            # Find all parquet files (including in subdirectories)
            parquet_files = []

            # Check root directory
            for item in os.listdir(path):
                item_path = os.path.join(path, item)
                if item.endswith('.parquet'):
                    parquet_files.append(item_path)

            # Check subdirectories (HF cached datasets have parquet files in data/ subdirectory)
            if not parquet_files:
                for item in os.listdir(path):
                    item_path = os.path.join(path, item)
                    if os.path.isdir(item_path):
                        for subfile in os.listdir(item_path):
                            if subfile.endswith('.parquet'):
                                parquet_files.append(os.path.join(item_path, subfile))

            if not parquet_files:
                print(f"Warning: No parquet files found in {path}")
                return path

            print(f"Found {len(parquet_files)} parquet files")

            # Load dataset from parquet files
            dataset = load_dataset('parquet', data_files=parquet_files, split='train')
        except Exception as e:
            print(f"Warning: Failed to load parquet dataset from {path}: {e}")
            return path

        # Create cache directory for extracted files
        if self.cache_dir:
            cache_base = Path(self.cache_dir)
        else:
            cache_base = Path(tempfile.gettempdir()) / "onetrainer_parquet_cache"

        cache_base.mkdir(parents=True, exist_ok=True)

        # Create a unique directory for this dataset
        # Replace slashes in HuggingFace identifiers with underscores
        dataset_name = path.replace('/', '_')
        extract_dir = cache_base / dataset_name
        extract_dir.mkdir(parents=True, exist_ok=True)

        # Check if already extracted
        marker_file = extract_dir / ".extracted_complete"
        if marker_file.exists():
            print(f"Dataset already extracted to {extract_dir}, using cached version")
            return str(extract_dir)

        print(f"Extracting {len(dataset)} images to {extract_dir}...")

        # Helper function to extract a single sample
        def extract_sample(idx_sample):
            idx, sample = idx_sample
            try:
                # Get image and caption
                image = sample.get('image')
                caption = sample.get('text', '')

                # Get original filename if available, otherwise generate one
                if 'original_filename' in sample:
                    base_name = Path(sample['original_filename']).stem
                else:
                    base_name = f"image_{idx:08d}"

                # Save image
                if image is not None:
                    # Determine image format
                    if hasattr(image, 'format') and image.format:
                        ext = image.format.lower()
                    else:
                        ext = 'png'  # default

                    if ext == 'jpeg':
                        ext = 'jpg'

                    image_path = extract_dir / f"{base_name}.{ext}"
                    image.save(image_path)

                    # Save caption
                    txt_path = extract_dir / f"{base_name}.txt"
                    with open(txt_path, 'w', encoding='utf-8') as f:
                        f.write(caption)
                    return True
                return False
            except Exception as e:
                print(f"Warning: Failed to extract sample {idx}: {e}")
                return False

        # Extract images in parallel using ThreadPoolExecutor
        # Use number of CPU cores, with a reasonable max to avoid overwhelming the system
        max_workers = min(os.cpu_count() or 4, 16)
        print(f"  Using {max_workers} parallel workers for extraction")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(extract_sample, (idx, sample)): idx
                      for idx, sample in enumerate(dataset)}

            completed = 0
            for future in as_completed(futures):
                completed += 1
                if completed % 1000 == 0:
                    print(f"  Extracted {completed}/{len(dataset)} images...")

            print(f"  Extraction complete: {completed}/{len(dataset)} images")

        # Create marker file to indicate extraction is complete
        marker_file.touch()

        return str(extract_dir)

    def start(self, variation: int):
        for in_index in range(self._get_previous_length(self.concept_in_name)):
            concept = self._get_previous_item(variation, self.concept_in_name, in_index)
            enabled = concept[self.enabled_in_name]

            if enabled:
                path = concept[self.path_in_name]

                print(f"[ExtractParquetDataset] Checking path: {path}")
                print(f"[ExtractParquetDataset] Path exists: {os.path.exists(path)}")
                print(f"[ExtractParquetDataset] Is directory: {os.path.isdir(path)}")

                # Check if this is a parquet dataset
                is_parquet = self.__is_parquet_dataset(path)
                print(f"[ExtractParquetDataset] Is parquet dataset: {is_parquet}")

                if is_parquet:
                    # Extract images from parquet
                    extracted_path = self.__extract_parquet_dataset(path, concept)

                    # Update concept with extracted path
                    concept = concept.copy()
                    concept[self.path_in_name] = extracted_path
                    print(f"[ExtractParquetDataset] Updated concept path to: {extracted_path}")
                else:
                    print(f"[ExtractParquetDataset] Not a parquet dataset, keeping original path")

                self.concepts.append(concept)
                print(f"[ExtractParquetDataset] Added concept with path: {concept[self.path_in_name]}")

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        return {
            self.concept_out_name: self.concepts[index],
        }
