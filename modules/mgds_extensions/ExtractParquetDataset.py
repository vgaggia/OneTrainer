import os
import tempfile
from pathlib import Path

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
        """Check if the path contains parquet files"""
        if not os.path.isdir(path):
            return False

        # Check for parquet files in the directory
        for file in os.listdir(path):
            if file.endswith('.parquet'):
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
            dataset = load_dataset(path, split='train')
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
        dataset_name = Path(path).name
        extract_dir = cache_base / dataset_name
        extract_dir.mkdir(parents=True, exist_ok=True)

        # Check if already extracted
        marker_file = extract_dir / ".extracted_complete"
        if marker_file.exists():
            print(f"Dataset already extracted to {extract_dir}, using cached version")
            return str(extract_dir)

        print(f"Extracting {len(dataset)} images to {extract_dir}...")

        # Extract images and create caption files
        for idx, sample in enumerate(dataset):
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

                # Progress indicator
                if (idx + 1) % 1000 == 0:
                    print(f"  Extracted {idx + 1}/{len(dataset)} images...")

            except Exception as e:
                print(f"Warning: Failed to extract sample {idx}: {e}")
                continue

        # Create marker file to indicate extraction is complete
        marker_file.touch()

        print(f"Extraction complete: {len(os.listdir(extract_dir))} files in {extract_dir}")
        return str(extract_dir)

    def start(self, variation: int):
        for in_index in range(self._get_previous_length(self.concept_in_name)):
            concept = self._get_previous_item(variation, self.concept_in_name, in_index)
            enabled = concept[self.enabled_in_name]

            if enabled:
                path = concept[self.path_in_name]

                # Check if this is a parquet dataset
                if self.__is_parquet_dataset(path):
                    # Extract images from parquet
                    extracted_path = self.__extract_parquet_dataset(path, concept)

                    # Update concept with extracted path
                    concept = concept.copy()
                    concept[self.path_in_name] = extracted_path

                self.concepts.append(concept)

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        return {
            self.concept_out_name: self.concepts[index],
        }
