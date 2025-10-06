#!/usr/bin/env python3
"""
Convert image+caption datasets to HuggingFace Dataset (parquet) format.
Solves the 100k file limit by storing everything in parquet files.
"""

import os
import argparse
import time
from pathlib import Path
from datasets import Dataset, Features, Image, Value
from tqdm import tqdm

class HuggingFaceDatasetConverter:
    def __init__(self, source_dir, dataset_name, private=True):
        """
        Args:
            source_dir: Source dataset directory with image and txt files
            dataset_name: HuggingFace dataset name (username/dataset-name)
            private: Whether to make the dataset private (default: True)
        """
        self.source_dir = Path(source_dir)
        self.dataset_name = dataset_name
        self.private = private

        if not self.source_dir.exists():
            raise ValueError(f"Source directory does not exist: {source_dir}")

    def log(self, message):
        """Log with timestamp"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")

    def find_image_caption_pairs(self):
        """Find all image+txt pairs in source directory, flattening all subdirectories"""
        self.log(f"Scanning {self.source_dir} for image and txt files...")

        # Image extensions we support
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}

        pairs = []
        missing_txt = []
        seen_files = set()

        # Recursively find all image files
        for root, dirs, files in os.walk(self.source_dir):
            # Skip cache directories
            if '.cache' in Path(root).parts or 'restructured' in Path(root).parts:
                continue

            for file in files:
                file_path = Path(root) / file
                ext = file_path.suffix.lower()

                if ext in image_extensions:
                    # Avoid duplicates in case of weird directory structures
                    if file_path in seen_files:
                        continue
                    seen_files.add(file_path)

                    # Look for corresponding txt file
                    txt_file = file_path.with_suffix('.txt')

                    if txt_file.exists():
                        pairs.append((file_path, txt_file))
                    else:
                        missing_txt.append(file_path)

        self.log(f"Found {len(pairs)} image+caption pairs")
        if missing_txt:
            self.log(f"Warning: {len(missing_txt)} image files without matching txt files")
            if len(missing_txt) <= 10:
                for img in missing_txt:
                    self.log(f"  Missing txt: {img}")

        return pairs

    def create_dataset(self, pairs, max_samples=None):
        """
        Create HuggingFace Dataset from image+caption pairs.

        Args:
            pairs: List of (image_path, txt_path) tuples
            max_samples: Limit number of samples (for testing)
        """
        if max_samples:
            pairs = pairs[:max_samples]
            self.log(f"Limiting to {max_samples} samples for testing")

        self.log(f"Creating HuggingFace Dataset with {len(pairs)} samples...")

        # Prepare data
        data = {
            "image": [],
            "text": [],
            "original_filename": [],
        }

        # Use tqdm for progress bar
        for image_path, txt_path in tqdm(pairs, desc="Processing files"):
            try:
                # Store image path (datasets library will load it)
                data["image"].append(str(image_path))

                # Read caption
                with open(txt_path, 'r', encoding='utf-8') as f:
                    caption = f.read().strip()
                data["text"].append(caption)

                # Store original filename for reference
                data["original_filename"].append(image_path.name)

            except Exception as e:
                self.log(f"  Error processing {image_path}: {e}")
                continue

        self.log(f"Creating Dataset object (this may take a while)...")

        # Create dataset with proper features
        dataset = Dataset.from_dict(
            data,
            features=Features({
                "image": Image(),
                "text": Value("string"),
                "original_filename": Value("string"),
            })
        )

        return dataset

    def upload_to_hub(self, dataset, dry_run=False):
        """
        Upload dataset to HuggingFace Hub.

        Args:
            dataset: HuggingFace Dataset object
            dry_run: If True, don't actually upload
        """
        if dry_run:
            self.log("DRY RUN - Dataset would be uploaded to HuggingFace Hub")
            self.log(f"  Dataset name: {self.dataset_name}")
            self.log(f"  Private: {self.private}")
            self.log(f"  Number of samples: {len(dataset)}")
            self.log(f"  Features: {dataset.features}")
            return

        self.log(f"Uploading dataset to HuggingFace Hub: {self.dataset_name}")
        self.log(f"  Private: {self.private}")
        self.log(f"  Number of samples: {len(dataset)}")
        self.log("  This may take a while for large datasets...")

        try:
            dataset.push_to_hub(
                self.dataset_name,
                private=self.private,
            )
            self.log("="*80)
            self.log("Upload completed successfully!")
            self.log(f"Dataset URL: https://huggingface.co/datasets/{self.dataset_name}")

        except Exception as e:
            self.log(f"ERROR uploading dataset: {e}")
            raise

    def convert_and_upload(self, dry_run=False, max_samples=None):
        """
        Full pipeline: find pairs, create dataset, upload.

        Args:
            dry_run: If True, don't upload
            max_samples: Limit samples (for testing)
        """
        # Find image+caption pairs
        pairs = self.find_image_caption_pairs()

        if not pairs:
            self.log("ERROR: No image+caption pairs found!")
            return False

        # Create dataset
        dataset = self.create_dataset(pairs, max_samples=max_samples)

        if len(dataset) == 0:
            self.log("ERROR: Dataset is empty after processing!")
            return False

        # Upload to HuggingFace Hub
        self.upload_to_hub(dataset, dry_run=dry_run)

        return True


def main():
    parser = argparse.ArgumentParser(
        description="Convert image+caption dataset to HuggingFace Dataset format and upload",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test run (dry-run with 10 samples)
  python convert_to_hf_dataset.py --username your-username --dry-run --test 10

  # Upload all datasets
  python convert_to_hf_dataset.py --username your-username

  # Upload specific dataset only
  python convert_to_hf_dataset.py --source D:\\datasets\\NewBig --dataset-name username/newbig
        """
    )
    parser.add_argument("--username", help="Your HuggingFace username (for batch mode)")
    parser.add_argument("--source", help="Source dataset directory (for single dataset mode)")
    parser.add_argument("--dataset-name", help="HuggingFace dataset name (for single dataset mode)")
    parser.add_argument("--public", action="store_true",
                       help="Make dataset public (default: private)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be done without uploading")
    parser.add_argument("--test", type=int, default=None,
                       help="Test mode: only convert first N samples")

    args = parser.parse_args()

    print("="*80)
    print("HuggingFace Dataset Converter")
    print("="*80)

    # Batch mode: convert all datasets
    if args.username and not args.source:
        datasets = [
            ("D:\\app\\datasets\\NewBig", f"{args.username}/newbig"),
            ("D:\\app\\datasets\\Gelbooru1024", f"{args.username}/gelbooru1024"),
            ("D:\\app\\datasets\\FurryEndAll", f"{args.username}/furryendall"),
        ]

        print(f"\nBatch mode: Converting {len(datasets)} datasets")
        print(f"Username: {args.username}")
        print(f"Private: {not args.public}")
        if args.test:
            print(f"Test mode: {args.test} samples per dataset")
        if args.dry_run:
            print("DRY RUN: No uploads will be performed")
        print()

        failed = []
        for source_dir, dataset_name in datasets:
            print("\n" + "="*80)
            print(f"Processing: {source_dir} -> {dataset_name}")
            print("="*80)

            try:
                converter = HuggingFaceDatasetConverter(
                    source_dir=source_dir,
                    dataset_name=dataset_name,
                    private=not args.public
                )

                success = converter.convert_and_upload(
                    dry_run=args.dry_run,
                    max_samples=args.test
                )

                if not success:
                    failed.append(dataset_name)

            except Exception as e:
                print(f"ERROR: Failed to process {dataset_name}: {e}")
                failed.append(dataset_name)

        print("\n" + "="*80)
        print("BATCH CONVERSION COMPLETE")
        print("="*80)
        if failed:
            print(f"Failed datasets: {', '.join(failed)}")
            exit(1)
        else:
            print("All datasets converted successfully!")

    # Single dataset mode
    elif args.source and args.dataset_name:
        converter = HuggingFaceDatasetConverter(
            source_dir=args.source,
            dataset_name=args.dataset_name,
            private=not args.public
        )

        success = converter.convert_and_upload(
            dry_run=args.dry_run,
            max_samples=args.test
        )

        if not success:
            exit(1)

    else:
        parser.error("Either --username (batch mode) or both --source and --dataset-name (single mode) required")


if __name__ == "__main__":
    main()
