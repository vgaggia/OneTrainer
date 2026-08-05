# LongCat Image Edit

OneTrainer supports `meituan-longcat/LongCat-Image-Edit` for full fine-tuning and LoRA training. The released ComfyUI BF16 transformer file can also be used as a transformer override while the official Diffusers repository supplies the Qwen2.5-VL encoder, VAE, tokenizer, processor, and scheduler.

## Captioned-image training (default)

The RTX 5090 presets leave **Custom Conditioning Image** disabled. A normal OneTrainer image-and-caption dataset then trains LongCat's text-to-image path with its native image-captioning system prompt. No synthetic source image is generated and no edit reference is passed to the transformer.

This matches LongCat's joint text-to-image/edit training design and is the intended mode when the dataset contains only source images and captions and the goal is broader model understanding.

The implementation follows the [LongCat-Image paper](https://arxiv.org/html/2512.07584) and [official training examples](https://github.com/meituan-longcat/LongCat-Image/tree/main/train_examples): edit samples concatenate clean source latents after noisy target latents and use separate target/source/text modality IDs, while source-free samples retain the model's jointly trained text-to-image behavior.

## Paired edit training (optional)

Enable **Custom Conditioning Image** to train edit pairs. For each target image, add the source/reference image using OneTrainer's existing conditioning postfix:

```text
example.png                 # edited target
example.txt                 # edit instruction
example-condlabel.png       # unedited source/reference
```

The target and source receive the same crop and geometric augmentations. The full-resolution source is encoded by the VAE and concatenated after the noised target tokens; a half-resolution copy is also encoded through Qwen2.5-VL. Target, source, and text use LongCat's native modality position IDs.

## ComfyUI output

- Full fine-tunes: choose **Original Transformer** (the 5090 preset already does). The result uses the exact `longcat_image_edit_bf16.safetensors` tensor namespace and can be placed in ComfyUI's `models/diffusion_models` directory.
- LoRAs: choose **Comfy LoRA** (the 5090 preset already does). The result uses Comfy's `diffusion_model.*` LoRA prefix and LongCat's required fused projection layout, and can be placed in `models/loras`.
- To start from the Comfy BF16 file, keep `meituan-longcat/LongCat-Image-Edit` as the base model and select `longcat_image_edit_bf16.safetensors` as the transformer model override.

For edit previews in OneTrainer, enable the sampling UI's **Image Editing** switch and select a source image. LongCat does not request a mask because its reference image is not an inpainting mask.

## RTX 5090 presets

The included presets are under **LongCat Image Edit**:

- **Finetune RTX 5090**: full BF16 transformer, 1024 px, batch 1 with four-step accumulation, gradient checkpointing, Adafactor at `1e-5`, and Comfy-compatible transformer output.
- **LoRA RTX 5090**: BF16 transformer, 1024 px, batch 2 with two-step accumulation, rank 32/alpha 8, AdamW at `1e-4`, and Comfy LoRA output.

Both presets use CUDNN attention and compile the LongCat transformer blocks. PyTorch 2.12's mixed-order reduction scheduler has a symbolic divisibility bug that previously corrupted the compiled 1024 px sample and raised `CantSplit` during backward. OneTrainer disables only that affected Inductor scheduler for LongCat while leaving the rest of `torch.compile` enabled. The workaround was validated against the official Hugging Face model with a coherent 1024 px sample and full BF16 forward/backward optimizer updates.

On the RTX 5090 validation run, warmed accumulation microsteps improved from about **1.361 s to 1.145 s**. Including the slower fourth microstep that performs the fused optimizer update, the estimated steady accumulation cycle improved from **1.587 s/it to 1.413 s/it** (about 11%, or roughly 7.5 hours across 156,000 microsteps). The first step pauses while Inductor builds its kernels; subsequent steps use the compiled cache.

Both cache VAE and Qwen2.5-VL outputs and default to captioned-image training. Paired edit inputs roughly double the image-token sequence; if a 1024 px edit batch exceeds VRAM, reduce batch size first, then use 768 px or enable transformer activation/layer offloading.

These settings were derived from LongCat's official 1024 px SFT (`1e-5`) and rank-32 LoRA (`1e-4`) recipes, then cross-checked against the local `Flux2 Klein Lora2AI`, `krea2 LoRA RTX 5090`, and `krea2 FT RTX 5090 Long` custom presets as well as OneTrainer's built-in 24 GB Krea/Flux presets. Adafactor keeps the full-finetune optimizer state practical on a 32 GB card; AdamW remains appropriate for the much smaller LoRA parameter set.

Measured on an RTX 5090 with the included presets and a clean GPU:

- 1024 px LoRA, batch 2 / accumulation 2: **21.95 GiB** peak board usage, finite loss, Comfy LoRA saved.
- 1024 px BF16 full fine-tune, batch 1 / accumulation 4: **30.82 GiB** peak board usage, finite loss, 11.68 GiB native transformer saved. Only 0.61 GiB remained free, so close ComfyUI and other CUDA applications first.

The BF16 transformer preset is already a true full-weight fine-tune and does not need quantized-weight training (QWT). If the transformer is changed to `INT_W8A8`, enable QWT; otherwise quantized transformer weights are frozen and only unquantized parameters can update.
