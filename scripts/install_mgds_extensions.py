#!/usr/bin/env python3
"""
Install custom mgds extensions into the mgds package.
This allows OneTrainer to extend mgds functionality without forking it.
"""
import shutil
from pathlib import Path

def install_extensions():
    # Get paths
    onetrainer_root = Path(__file__).parent.parent
    extensions_dir = onetrainer_root / "modules" / "mgds_extensions"

    # Find mgds installation
    mgds_modules_dir = onetrainer_root / "external" / "mgds" / "src" / "mgds" / "pipelineModules"

    if not mgds_modules_dir.exists():
        print(f"Error: mgds not found at {mgds_modules_dir}")
        print("Make sure mgds is installed in external/mgds")
        return False

    if not extensions_dir.exists():
        print(f"No extensions found at {extensions_dir}")
        return True

    # Copy all extension files
    for ext_file in extensions_dir.glob("*.py"):
        if ext_file.name == "__init__.py":
            continue

        target = mgds_modules_dir / ext_file.name
        print(f"Installing {ext_file.name} -> {target}")
        shutil.copy2(ext_file, target)

    print("mgds extensions installed successfully!")
    return True

if __name__ == "__main__":
    success = install_extensions()
    exit(0 if success else 1)
