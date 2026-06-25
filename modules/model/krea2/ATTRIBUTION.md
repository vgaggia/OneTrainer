# Vendored Krea 2 transformer

`mmdit.py` (`SingleStreamDiT` and friends) is vendored from Krea's official open
Krea 2 release and ostris/ai-toolkit's training-ready adaptation of it:

- Krea 2 reference: https://huggingface.co/krea (Krea-2-Raw / Krea-2-Turbo) — released under Krea's community license (https://www.krea.ai/krea-2-licensing).
- ai-toolkit Krea2 extension: https://github.com/ostris/ai-toolkit (`extensions_built_in/diffusion_models/krea2`).

Kept close to upstream (see the module docstring for the small training/portability
deltas). The file is excluded from ruff in `pyproject.toml` so it stays diffable
against upstream.
