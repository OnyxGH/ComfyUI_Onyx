from __future__ import annotations

import inspect
import logging
import math
import os
import random
import re
import time
from typing import Any

import comfy
import comfy.model_management as model_management
import comfy.sample
import comfy.samplers
import comfy.utils
import folder_paths
import latent_preview
import numpy as np
import nodes
import torch
import torch.nn.functional as F
from PIL import Image
from comfy import samplers
from comfy.k_diffusion import sampling as k_diffusion_sampling
from comfy_extras import nodes_custom_sampler, nodes_differential_diffusion

try:
    from comfy_extras.nodes_custom_sampler import Noise_EmptyNoise, Noise_RandomNoise
    import node_helpers
except Exception as exc:  # pragma: no cover - depends on ComfyUI runtime
    Noise_EmptyNoise = None
    Noise_RandomNoise = None
    node_helpers = None
    _SAMPLER_IMPORT_ERROR = exc
else:
    _SAMPLER_IMPORT_ERROR = None

from .segmentation import SEG

_LANCZOS = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
_ADDITIONAL_SCHEDULERS_BY_NODE = {
    "AlignYourStepsScheduler": ["AYS SDXL", "AYS SD1", "AYS SVD"],
    "GITSScheduler": ["GITS[coeff=1.2]"],
    "LTXVScheduler": ["LTXV[default]"],
    "OptimalStepsScheduler": ["OSS FLUX", "OSS Wan", "OSS Chroma"],
}


def get_scheduler_options() -> list[str]:
    schedulers = list(getattr(comfy.samplers, "SCHEDULER_HANDLERS", comfy.samplers.KSampler.SCHEDULERS))
    node_mappings = getattr(nodes, "NODE_CLASS_MAPPINGS", None)
    if not isinstance(node_mappings, dict):
        return schedulers
    for node_name, extra_values in _ADDITIONAL_SCHEDULERS_BY_NODE.items():
        if node_name in node_mappings:
            schedulers.extend(extra_values)
    return schedulers


def _ensure_sampler_support() -> None:
    if _SAMPLER_IMPORT_ERROR is not None or Noise_EmptyNoise is None or Noise_RandomNoise is None or node_helpers is None:
        raise RuntimeError("This ComfyUI build is missing sampler APIs required by Onyx's local detailer implementation.") from _SAMPLER_IMPORT_ERROR


class WildcardChooser:
    def __init__(self, items: list[tuple[int | None, str]], randomize_when_exhaust: bool) -> None:
        self.index = 0
        self.items = items
        self.randomize_when_exhaust = randomize_when_exhaust

    def get(self, _seg) -> tuple[int | None, str]:
        if self.index >= len(self.items):
            self.index = 0
            if self.randomize_when_exhaust:
                random.shuffle(self.items)
        item = self.items[self.index]
        self.index += 1
        return item


class WildcardChooserDict:
    def __init__(self, items: dict[str, str]) -> None:
        self.items = items

    def get(self, seg) -> str:
        text = self.items.get("ALL", "")
        if seg.label in self.items:
            text += self.items[seg.label]
        return text


def _starts_with_regex(pattern: str, text: str):
    return re.compile(pattern).match(text)


def _split_to_dict(text: str) -> dict[str, str]:
    pattern = r"\[([A-Za-z0-9_. ]+)\]([^\[]+)(?=\[|$)"
    matches = re.findall(pattern, text)
    return {key: value.strip() for key, value in matches}


def _split_string_with_sep(input_string: str) -> list[tuple[int | None, str]]:
    sep_pattern = r"\[SEP(?:\:\w+)?\]"
    substrings = re.split(sep_pattern, input_string)
    result_list: list[int | None | str] = [None]
    matches = re.findall(sep_pattern, input_string)
    for index, substring in enumerate(substrings):
        result_list.append(substring)
        if index < len(matches):
            if matches[index] == "[SEP]":
                result_list.append(None)
            elif matches[index] == "[SEP:R]":
                result_list.append(random.randint(0, 1125899906842624))
            else:
                try:
                    result_list.append(int(matches[index][5:-1]))
                except Exception:
                    result_list.append(None)
    iterator = iter(result_list)
    return list(zip(iterator, iterator))


def process_wildcard_for_segs(wildcard: str) -> tuple[str | None, WildcardChooser | WildcardChooserDict]:
    if wildcard.startswith("[LAB]"):
        items = {key: value.strip() for key, value in _split_to_dict(wildcard).items() if value.strip()}
        return "LAB", WildcardChooserDict(items)

    match = _starts_with_regex(r"\[(ASC-SIZE|DSC-SIZE|ASC|DSC|RND)\]", wildcard)
    if match:
        mode = match[1]
        items = _split_string_with_sep(wildcard[len(match[0]) :])
        if mode == "RND":
            random.shuffle(items)
            return mode, WildcardChooser(items, True)
        return mode, WildcardChooser(items, False)

    return None, WildcardChooser([(None, wildcard)], False)


def _extract_lora_values(text: str) -> list[tuple[str, float, float, str | None, float | None, float | None, str | None]]:
    matches = re.findall(r"<lora:([^>]+)>", text)
    items = [re.sub(r"LBW=[A-Za-z][A-Za-z0-9_-]*:", "LBW=", match.strip(":")) for match in matches]
    added: set[str] = set()
    result = []
    for item in items:
        parts = item.split(":")
        lora_name = parts[0] if parts else None
        model_weight: float | None = None
        clip_weight: float | None = None
        lbw = None
        lbw_a = None
        lbw_b = None
        loader = None

        for part in parts[1:]:
            try:
                numeric = float(part)
            except Exception:
                numeric = None
            if numeric is not None:
                if model_weight is None:
                    model_weight = numeric
                elif clip_weight is None:
                    clip_weight = numeric
                continue
            if part.startswith("LBW="):
                for lbw_item in part[4:].split(";"):
                    if lbw_item.startswith("A="):
                        try:
                            lbw_a = float(lbw_item[2:].strip())
                        except Exception:
                            lbw_a = None
                    elif lbw_item.startswith("B="):
                        try:
                            lbw_b = float(lbw_item[2:].strip())
                        except Exception:
                            lbw_b = None
                    elif lbw_item.strip():
                        lbw = lbw_item
            elif part.startswith("LOADER="):
                loader = part[7:]

        if lora_name is None or lora_name in added:
            continue
        if model_weight is None:
            model_weight = 1.0
        if clip_weight is None:
            clip_weight = model_weight
        result.append((lora_name, model_weight, clip_weight, lbw, lbw_a, lbw_b, loader))
        added.add(lora_name)
    return result


def _remove_lora_tags(text: str) -> str:
    return re.sub(r"<lora:[^>]+>", "", text)


def _resolve_lora_name(cache: list[str], name: str) -> str | None:
    if os.path.exists(name):
        return name
    if not cache:
        cache.extend(folder_paths.get_filename_list("loras"))
    for item in cache:
        if item.endswith(name):
            return item
    return None


def _encode_prompt_segments(clip, prompt_text: str):
    prompts = [part.strip() for part in prompt_text.split("BREAK")]
    prompts = [part for part in prompts if part]
    if not prompts:
        prompts = [""]

    result = None
    for prompt in prompts:
        current = nodes.CLIPTextEncode().encode(clip, prompt)[0]
        result = nodes.ConditioningConcat().concat(result, current)[0] if result is not None else current
    return result


def _process_prompt_with_loras(prompt_text: str, model, clip):
    lora_name_cache: list[str] = []
    stripped_prompt = _remove_lora_tags(prompt_text)

    for lora_name, model_weight, clip_weight, lbw, lbw_a, lbw_b, loader in _extract_lora_values(prompt_text):
        parts = lora_name.split(".")
        if ("." + parts[-1]) not in folder_paths.supported_pt_extensions:
            lora_name = lora_name + ".safetensors"

        resolved_name = _resolve_lora_name(lora_name_cache, lora_name)
        if resolved_name is None:
            logging.warning(f"[Onyx] LoRA not found: {lora_name}")
            continue

        if loader == "nunchaku" and "NunchakuFluxLoraLoader" in nodes.NODE_CLASS_MAPPINGS:
            model = nodes.NODE_CLASS_MAPPINGS["NunchakuFluxLoraLoader"]().load_lora(model, resolved_name, model_weight)[0]
            continue
        if loader and loader != "nunchaku":
            logging.warning(f"[Onyx] Unknown LoRA loader '{loader}'. Falling back to the default loader.")

        if lbw is not None and "LoraLoaderBlockWeight //Inspire" in nodes.NODE_CLASS_MAPPINGS:
            model, clip, _ = nodes.NODE_CLASS_MAPPINGS["LoraLoaderBlockWeight //Inspire"]().doit(
                model,
                clip,
                resolved_name,
                model_weight,
                clip_weight,
                False,
                0,
                lbw_a,
                lbw_b,
                "",
                lbw,
            )
        else:
            if lbw is not None:
                logging.warning("[Onyx] LBW syntax requested, but Inspire Pack is not installed. Falling back to the default LoRA loader.")
            model, clip = nodes.LoraLoader().load_lora(model, clip, resolved_name, model_weight, clip_weight)

    return model, clip, _encode_prompt_segments(clip, stripped_prompt)


def _tensor_check_image(image: torch.Tensor) -> None:
    if image.ndim != 4:
        raise ValueError(f"Expected NHWC tensor, but found {image.ndim} dimensions")
    if image.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Expected 1, 3 or 4 channels for image, but found {image.shape[-1]} channels")


def _tensor_check_mask(mask: torch.Tensor) -> None:
    if mask.ndim != 4:
        raise ValueError(f"Expected NHWC tensor, but found {mask.ndim} dimensions")
    if mask.shape[-1] != 1:
        raise ValueError(f"Expected 1 channel for mask, but found {mask.shape[-1]} channels")


def _crop_ndarray4(value, crop_region):
    x1, y1, x2, y2 = crop_region
    return value[:, y1:y2, x1:x2, :]


def _crop_ndarray3(value, crop_region):
    x1, y1, x2, y2 = crop_region
    return value[:, y1:y2, x1:x2]


def _to_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, Image.Image):
        return torch.from_numpy(np.array(value).astype(np.float32) / 255.0)
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    raise ValueError(f"Cannot convert {type(value)} to torch.Tensor")


def _tensor_to_pil(image: torch.Tensor) -> Image.Image:
    _tensor_check_image(image)
    return Image.fromarray(np.clip(255.0 * image.detach().cpu().numpy().squeeze(0), 0, 255).astype(np.uint8))


def _resize_mask(mask: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim == 4 and mask.shape[-1] == 1:
        mask = mask.squeeze(-1)
    resized = F.interpolate(mask.unsqueeze(1).float(), size=size, mode="bilinear", align_corners=False)
    return resized.squeeze(1)


def tensor_convert_rgba(image: torch.Tensor) -> torch.Tensor:
    _tensor_check_image(image)
    if image.shape[-1] == 4:
        return image.clone()
    if image.shape[-1] == 3:
        alpha = torch.ones((*image.shape[:-1], 1), dtype=image.dtype, device=image.device)
        return torch.cat((image, alpha), dim=-1)
    raise ValueError(f"illegal conversion (channels: {image.shape[-1]} -> 4)")


def tensor_convert_rgb(image: torch.Tensor) -> torch.Tensor:
    _tensor_check_image(image)
    if image.shape[-1] == 3:
        return image.clone()
    if image.shape[-1] == 4:
        return image[..., :3].clone()
    raise ValueError(f"illegal conversion (channels: {image.shape[-1]} -> 3)")


def tensor_resize(image: torch.Tensor, width: int, height: int) -> torch.Tensor:
    _tensor_check_image(image)
    if image.shape[-1] >= 3:
        scaled = []
        for single_image in image:
            pil_image = _tensor_to_pil(single_image.unsqueeze(0))
            scaled.append(torch.from_numpy(np.array(pil_image.resize((width, height), resample=_LANCZOS)).astype(np.float32) / 255.0))
        return torch.stack(scaled, dim=0)

    nchw = image.permute(0, 3, 1, 2)
    resized = F.interpolate(nchw.float(), size=(height, width), mode="bilinear", align_corners=False)
    return resized.permute(0, 2, 3, 1).to(dtype=image.dtype)


def tensor_get_size(image: torch.Tensor) -> tuple[int, int]:
    _tensor_check_image(image)
    return int(image.shape[2]), int(image.shape[1])


def tensor_putalpha(image: torch.Tensor, mask: torch.Tensor) -> None:
    _tensor_check_image(image)
    _tensor_check_mask(mask)
    image[..., -1] = mask[..., 0]


def tensor_paste(image1: torch.Tensor, image2: torch.Tensor, left_top: tuple[int, int], mask: torch.Tensor) -> None:
    _tensor_check_image(image1)
    _tensor_check_image(image2)
    _tensor_check_mask(mask)

    if image2.shape[1:3] != mask.shape[1:3]:
        mask = _resize_mask(mask.squeeze(-1), image2.shape[1:3]).unsqueeze(-1)

    x, y = left_top
    _, h1, w1, _ = image1.shape
    _, h2, w2, _ = image2.shape
    width = min(w1, x + w2) - x
    height = min(h1, y + h2) - y
    if width <= 0 or height <= 0:
        return

    mask = mask[:, :height, :width, :].to(image1.device, dtype=image1.dtype)
    region1 = image1[:, y : y + height, x : x + width, :]
    region2 = image2[:, :height, :width, :].to(image1.device, dtype=image1.dtype)
    image1[:, y : y + height, x : x + width, :] = (1.0 - mask) * region1 + mask * region2


def _gaussian_kernel(kernel_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(kernel_size, dtype=dtype, device=device) - (kernel_size - 1) / 2.0
    kernel_1d = torch.exp(-(coords.square()) / (2.0 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()
    return torch.outer(kernel_1d, kernel_1d).view(1, 1, kernel_size, kernel_size)


def tensor_gaussian_blur_mask(mask: Any, kernel_size: int, sigma: float = 10.0) -> torch.Tensor:
    mask = _to_tensor(mask)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(-1)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(-1)
    _tensor_check_mask(mask)

    if kernel_size <= 0:
        return mask

    kernel_size = kernel_size * 2 + 1
    shortest = min(mask.shape[1], mask.shape[2])
    if shortest <= kernel_size:
        kernel_size = int(shortest / 2)
        if kernel_size % 2 == 0:
            kernel_size += 1
        if kernel_size < 3:
            return mask

    prev_device = mask.device
    device = comfy.model_management.get_torch_device()
    nchw = mask.to(device=device, dtype=torch.float32).permute(0, 3, 1, 2)
    kernel = _gaussian_kernel(kernel_size, sigma, device, nchw.dtype)
    blurred = F.conv2d(nchw, kernel, padding=kernel_size // 2)
    return blurred.permute(0, 2, 3, 1).to(prev_device)


def to_latent_image(pixels: torch.Tensor, vae, vae_tiled_encode: bool = False):
    start = time.time()
    if vae_tiled_encode:
        encoded = nodes.VAEEncodeTiled().encode(vae, pixels, 512, overlap=64)[0]
        logging.info(f"[Onyx] VAE encoded (tiled) in {time.time() - start:.1f}s")
    else:
        encoded = nodes.VAEEncode().encode(vae, pixels)[0]
        logging.info(f"[Onyx] VAE encoded in {time.time() - start:.1f}s")
    return encoded


def calculate_sigmas(model, sampler_name: str, scheduler: str, steps: int) -> torch.Tensor:
    discard_penultimate_sigma = False
    if sampler_name in ["dpm_2", "dpm_2_ancestral", "uni_pc", "uni_pc_bh2"]:
        steps += 1
        discard_penultimate_sigma = True

    if scheduler.startswith("AYS"):
        sigmas = nodes.NODE_CLASS_MAPPINGS["AlignYourStepsScheduler"]().get_sigmas(scheduler[4:], steps, denoise=1.0)[0]
    elif scheduler.startswith("GITS[coeff="):
        sigmas = nodes.NODE_CLASS_MAPPINGS["GITSScheduler"]().execute(float(scheduler[11:-1]), steps, denoise=1.0)[0]
    elif scheduler == "LTXV[default]":
        sigmas = nodes.NODE_CLASS_MAPPINGS["LTXVScheduler"]().execute(20, 2.05, 0.95, True, 0.1)[0]
    elif scheduler.startswith("OSS"):
        sigmas = nodes.NODE_CLASS_MAPPINGS["OptimalStepsScheduler"]().execute(scheduler[4:], steps, denoise=1.0)[0]
    else:
        sigmas = samplers.calculate_sigmas(model.get_model_object("model_sampling"), scheduler, steps)

    if discard_penultimate_sigma:
        sigmas = torch.cat([sigmas[:-2], sigmas[-1:]])
    return sigmas


def _get_noise_sampler(x: torch.Tensor, cpu: bool, total_sigmas: torch.Tensor, **kwargs):
    extra_args = kwargs.get("extra_args", {})
    if "seed" in extra_args:
        sigma_min = total_sigmas[total_sigmas > 0].min()
        sigma_max = total_sigmas.max()
        return k_diffusion_sampling.BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=extra_args.get("seed"), cpu=cpu)
    return None


def _ksampler(sampler_name: str, total_sigmas: torch.Tensor):
    if sampler_name in ["dpmpp_sde", "dpmpp_sde_gpu", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_3m_sde", "dpmpp_3m_sde_gpu"]:
        if sampler_name == "dpmpp_sde":
            original_fn = k_diffusion_sampling.sample_dpmpp_sde
        elif sampler_name == "dpmpp_sde_gpu":
            original_fn = k_diffusion_sampling.sample_dpmpp_sde_gpu
        elif sampler_name == "dpmpp_2m_sde":
            original_fn = k_diffusion_sampling.sample_dpmpp_2m_sde
        elif sampler_name == "dpmpp_2m_sde_gpu":
            original_fn = k_diffusion_sampling.sample_dpmpp_2m_sde_gpu
        elif sampler_name == "dpmpp_3m_sde":
            original_fn = k_diffusion_sampling.sample_dpmpp_3m_sde
        else:
            original_fn = k_diffusion_sampling.sample_dpmpp_3m_sde_gpu

        def sampler_fn(model_inner, x, sigmas, **kwargs):
            if "noise_sampler" not in kwargs:
                kwargs["noise_sampler"] = _get_noise_sampler(x, "gpu" not in sampler_name, total_sigmas, **kwargs)
            return original_fn(model_inner, x, sigmas, **kwargs)

        return samplers.KSAMPLER(sampler_fn, {}, {})

    return comfy.samplers.sampler_object(sampler_name)


def sample_with_custom_noise(model, add_noise: bool, noise_seed: int, cfg: float, positive, negative, sampler, sigmas: torch.Tensor, latent_image, noise: torch.Tensor | None = None, callback=None):
    _ensure_sampler_support()

    latent = latent_image
    latent_samples = latent["samples"]
    if hasattr(comfy.sample, "fix_empty_latent_channels"):
        latent_samples = comfy.sample.fix_empty_latent_channels(model, latent_samples)

    output = latent.copy()
    output["samples"] = latent_samples
    if noise is None:
        noise = Noise_EmptyNoise().generate_noise(output) if not add_noise else Noise_RandomNoise(noise_seed).generate_noise(output)

    noise_mask = latent.get("noise_mask")
    x0_output: dict[str, torch.Tensor] = {}
    preview_callback = latent_preview.prepare_callback(model, sigmas.shape[-1] - 1, x0_output)
    if callback is not None:
        def touched_callback(step, x0, x, total_steps):
            callback(step, x0, x, total_steps)
            preview_callback(step, x0, x, total_steps)
    else:
        touched_callback = preview_callback

    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    device = model_management.get_torch_device()
    noise = noise.to(device)
    latent_samples = latent_samples.to(device)
    if noise_mask is not None:
        noise_mask = noise_mask.to(device)

    if negative != "NegativePlaceholder":
        samples = comfy.sample.sample_custom(model, noise, cfg, sampler, sigmas, positive, negative, latent_samples, noise_mask=noise_mask, callback=touched_callback, disable_pbar=disable_pbar, seed=noise_seed)
    else:
        guider = nodes_custom_sampler.Guider_Basic(model)
        positive = node_helpers.conditioning_set_values(positive, {"guidance": cfg})
        guider.set_conds(positive)
        samples = guider.sample(noise, latent_samples, sampler, sigmas, denoise_mask=noise_mask, callback=touched_callback, disable_pbar=disable_pbar, seed=noise_seed)

    samples = samples.to(comfy.model_management.intermediate_device())
    output["samples"] = samples
    if "x0" in x0_output:
        denoised = latent.copy()
        denoised["samples"] = model.model.process_latent_out(x0_output["x0"].cpu())
    else:
        denoised = output
    return output, denoised


def separated_sample(model, add_noise: bool, seed: int, steps: int, cfg: float, sampler_name: str, scheduler: str, positive, negative, latent_image, start_at_step: int | None, end_at_step: int | None, return_with_leftover_noise: bool, sigma_ratio: float = 1.0, sampler_opt=None, noise: torch.Tensor | None = None, callback=None, scheduler_func=None):
    total_sigmas = scheduler_func(model, sampler_name, steps) if scheduler_func is not None else calculate_sigmas(model, sampler_name if sampler_opt is None else "", scheduler, steps)
    sigmas = total_sigmas

    if end_at_step is not None and end_at_step < (len(total_sigmas) - 1):
        sigmas = total_sigmas[: end_at_step + 1]
        if not return_with_leftover_noise:
            sigmas[-1] = 0

    if start_at_step is not None:
        if start_at_step < (len(sigmas) - 1):
            sigmas = sigmas[start_at_step:] * sigma_ratio
        else:
            return latent_image if latent_image is not None else {"samples": torch.zeros_like(noise)}

    impact_sampler = sampler_opt if sampler_opt is not None else _ksampler(sampler_name, total_sigmas)
    if len(sigmas) == 0 or (len(sigmas) == 1 and sigmas[0] == 0):
        return latent_image

    result = sample_with_custom_noise(model, add_noise, seed, cfg, positive, negative, impact_sampler, sigmas, latent_image, noise=noise, callback=callback)
    return result[0] if return_with_leftover_noise else result[1]


def ksampler_wrapper(model, seed: int, steps: int, cfg: float, sampler_name: str, scheduler: str, positive, negative, latent_image, denoise: float, refiner_ratio=None, refiner_model=None, refiner_clip=None, refiner_positive=None, refiner_negative=None, sigma_factor: float = 1.0, noise: torch.Tensor | None = None, scheduler_func=None, sampler_opt=None):
    advanced_steps = math.floor(steps / denoise)
    start_at_step = advanced_steps - steps

    if refiner_ratio is None or refiner_model is None or refiner_clip is None or refiner_positive is None or refiner_negative is None:
        end_at_step = start_at_step + steps
        return separated_sample(model, True, seed, advanced_steps, cfg, sampler_name, scheduler, positive, negative, latent_image, start_at_step, end_at_step, False, sigma_ratio=sigma_factor, sampler_opt=sampler_opt, noise=noise, scheduler_func=scheduler_func)

    end_at_step = start_at_step + math.floor(steps * (1.0 - refiner_ratio))
    temp_latent = separated_sample(model, True, seed, advanced_steps, cfg, sampler_name, scheduler, positive, negative, latent_image, start_at_step, end_at_step, True, sigma_ratio=sigma_factor, sampler_opt=sampler_opt, noise=noise, scheduler_func=scheduler_func)

    if "noise_mask" in latent_image:
        compositor = nodes.NODE_CLASS_MAPPINGS["LatentCompositeMasked"]()
        temp_latent = compositor.composite(latent_image, temp_latent, 0, 0, False, latent_image["noise_mask"])[0]

    return separated_sample(refiner_model, False, seed, advanced_steps, cfg, sampler_name, scheduler, refiner_positive, refiner_negative, temp_latent, end_at_step, advanced_steps + 1, False, sigma_ratio=sigma_factor, sampler_opt=sampler_opt, scheduler_func=scheduler_func)


def segs_scale_match(segs, target_shape):
    height, width = segs[0]
    target_height = target_shape[1]
    target_width = target_shape[2]

    if (height == target_height and width == target_width) or height == 0 or width == 0:
        return segs

    ratio_h = target_height / height
    ratio_w = target_width / width
    new_segs = []
    for seg in segs[1]:
        cropped_image = seg.cropped_image
        cropped_mask = seg.cropped_mask
        x1, y1, x2, y2 = seg.crop_region
        bx1, by1, bx2, by2 = seg.bbox

        crop_region = int(x1 * ratio_w), int(y1 * ratio_w), int(x2 * ratio_h), int(y2 * ratio_h)
        bbox = int(bx1 * ratio_w), int(by1 * ratio_w), int(bx2 * ratio_h), int(by2 * ratio_h)
        new_width = crop_region[2] - crop_region[0]
        new_height = crop_region[3] - crop_region[1]

        mask_tensor = cropped_mask if isinstance(cropped_mask, torch.Tensor) else torch.from_numpy(cropped_mask)
        if mask_tensor.ndim == 3:
            resized_mask = F.interpolate(mask_tensor.unsqueeze(0).float(), size=(new_height, new_width), mode="bilinear", align_corners=False).squeeze(0)
        else:
            resized_mask = F.interpolate(mask_tensor.unsqueeze(0).unsqueeze(0).float(), size=(new_height, new_width), mode="bilinear", align_corners=False).squeeze(0).squeeze(0).numpy()

        if cropped_image is not None:
            image_tensor = cropped_image if isinstance(cropped_image, torch.Tensor) else torch.from_numpy(cropped_image)
            cropped_image = tensor_resize(image_tensor, new_width, new_height).numpy()

        new_segs.append(SEG(cropped_image, resized_mask, seg.confidence, crop_region, bbox, seg.label, seg.control_net_wrapper))
    return (target_height, target_width), new_segs


def crop_condition_mask(mask, image: torch.Tensor, crop_region):
    cond_scale = (mask.shape[1] / image.shape[1], mask.shape[2] / image.shape[2])
    mask_region = [round(value * cond_scale[index % 2]) for index, value in enumerate(crop_region)]
    return _crop_ndarray3(mask, mask_region)


def enhance_detail(image: torch.Tensor, model, clip, vae, guide_size: float, guide_size_for_bbox: bool, max_size: float, bbox, seed: int, steps: int, cfg: float, sampler_name: str, scheduler: str, positive, negative, denoise: float, noise_mask, force_inpaint: bool, wildcard_opt: str | None = None, wildcard_opt_concat_mode: str | None = None, detailer_hook=None, refiner_ratio=None, refiner_model=None, refiner_clip=None, refiner_positive=None, refiner_negative=None, control_net_wrapper=None, cycle: int = 1, inpaint_model: bool = False, noise_mask_feather: int = 0, scheduler_func=None, vae_tiled_encode: bool = False, vae_tiled_decode: bool = False):
    if noise_mask is not None:
        noise_mask = tensor_gaussian_blur_mask(noise_mask, noise_mask_feather).squeeze(3)
        if noise_mask_feather > 0 and "denoise_mask_function" not in model.model_options:
            model = nodes_differential_diffusion.DifferentialDiffusion().execute(model)[0]

    if wildcard_opt:
        model, clip, wildcard_positive = _process_prompt_with_loras(wildcard_opt, model, clip)
        if wildcard_opt_concat_mode == "concat":
            positive = nodes.ConditioningConcat().concat(positive, wildcard_positive)[0]
        else:
            positive = [wildcard_positive[0].copy()]
            if "pooled_output" in wildcard_positive[0][1]:
                positive[0][1]["pooled_output"] = wildcard_positive[0][1]["pooled_output"]
            elif "pooled_output" in positive[0][1]:
                del positive[0][1]["pooled_output"]

    crop_height = image.shape[1]
    crop_width = image.shape[2]
    bbox_height = bbox[3] - bbox[1]
    bbox_width = bbox[2] - bbox[0]
    if not force_inpaint and bbox_height >= guide_size and bbox_width >= guide_size:
        logging.info("[Onyx] Detailer segment skipped because the bbox is already larger than the guide size.")
        return None, None

    upscale = guide_size / min(bbox_width, bbox_height) if guide_size_for_bbox else guide_size / min(crop_width, crop_height)
    new_width = int(crop_width * upscale)
    new_height = int(crop_height * upscale)

    if "aitemplate_keep_loaded" in model.model_options:
        max_size = min(4096, max_size)
    if new_width > max_size or new_height > max_size:
        upscale *= max_size / max(new_width, new_height)
        new_width = int(crop_width * upscale)
        new_height = int(crop_height * upscale)

    if not force_inpaint:
        if upscale <= 1.0 or new_width == 0 or new_height == 0:
            return None, None
    elif upscale <= 1.0 or new_width == 0 or new_height == 0:
        upscale = 1.0
        new_width = crop_width
        new_height = crop_height

    if detailer_hook is not None:
        new_width, new_height = detailer_hook.touch_scaled_size(new_width, new_height)

    upscaled_image = tensor_resize(image, new_width, new_height)
    if detailer_hook is not None:
        upscaled_image = detailer_hook.post_upscale(upscaled_image, noise_mask)

    cnet_pils = None
    if control_net_wrapper is not None:
        positive, negative, cnet_pils = control_net_wrapper.apply(positive, negative, upscaled_image, noise_mask)
        model, cnet_pils2 = control_net_wrapper.doit_ipadapter(model)
        cnet_pils.extend(cnet_pils2)

    if detailer_hook is None or not detailer_hook.get_skip_sampling():
        if noise_mask is not None and inpaint_model:
            encode = nodes.InpaintModelConditioning().encode
            if "noise_mask" in inspect.signature(encode).parameters:
                positive, negative, latent_image = encode(positive, negative, upscaled_image, vae, mask=noise_mask, noise_mask=True)
            else:
                positive, negative, latent_image = encode(positive, negative, upscaled_image, vae, noise_mask)
        else:
            latent_image = to_latent_image(upscaled_image, vae, vae_tiled_encode=vae_tiled_encode)
            if noise_mask is not None:
                latent_image["noise_mask"] = noise_mask

        if detailer_hook is not None:
            latent_image = detailer_hook.post_encode(latent_image)

        refined_latent = latent_image
        sampler_opt = detailer_hook.get_custom_sampler() if detailer_hook is not None else None
        for cycle_index in range(cycle):
            if detailer_hook is not None:
                detailer_hook.set_steps((cycle_index, cycle))
                refined_latent = detailer_hook.cycle_latent(refined_latent)
                model2, seed2, steps2, cfg2, sampler_name2, scheduler2, positive2, negative2, _latent2, denoise2 = detailer_hook.pre_ksample(model, seed + cycle_index, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise)
                noise, is_touched = detailer_hook.get_custom_noise(seed + cycle_index, torch.zeros(latent_image["samples"].size()), is_touched=False)
                if not is_touched:
                    noise = None
            else:
                model2, seed2, steps2, cfg2, sampler_name2, scheduler2, positive2, negative2, _latent2, denoise2 = model, seed + cycle_index, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise
                noise = None

            refined_latent = ksampler_wrapper(model2, seed2, steps2, cfg2, sampler_name2, scheduler2, positive2, negative2, refined_latent, denoise2, refiner_ratio, refiner_model, refiner_clip, refiner_positive, refiner_negative, noise=noise, scheduler_func=scheduler_func, sampler_opt=sampler_opt)

        if detailer_hook is not None:
            refined_latent = detailer_hook.pre_decode(refined_latent)

        start = time.time()
        if vae_tiled_decode:
            refined_image = nodes.VAEDecodeTiled().decode(vae, refined_latent, 512)[0]
            logging.info(f"[Onyx] VAE decoded (tiled) in {time.time() - start:.1f}s")
        else:
            try:
                refined_image = vae.decode(refined_latent["samples"])
            except Exception:
                logging.warning(f"[Onyx] VAE decode failed after {time.time() - start:.1f}s, retrying with tiled decode.")
                refined_image = vae.decode_tiled(refined_latent["samples"], tile_x=64, tile_y=64)
            logging.info(f"[Onyx] VAE decoded in {time.time() - start:.1f}s")
    else:
        refined_image = upscaled_image

    if detailer_hook is not None:
        refined_image = detailer_hook.post_decode(refined_image)
    if len(refined_image.shape) == 5:
        refined_image = refined_image.squeeze(0)
    return tensor_resize(refined_image, crop_width, crop_height).cpu(), cnet_pils


class DetailerForEachCompat:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple]]:
        return {"required": {"scheduler": (get_scheduler_options(),)}}

    @staticmethod
    def do_detail(image: torch.Tensor, segs, model, clip, vae, guide_size: float, guide_size_for_bbox: bool, max_size: float, seed: int, steps: int, cfg: float, sampler_name: str, scheduler: str, positive, negative, denoise: float, feather: int, noise_mask: bool, force_inpaint: bool, wildcard_opt: str | None = None, detailer_hook=None, refiner_ratio=None, refiner_model=None, refiner_clip=None, refiner_positive=None, refiner_negative=None, cycle: int = 1, inpaint_model: bool = False, noise_mask_feather: int = 0, scheduler_func_opt=None, tiled_encode: bool = False, tiled_decode: bool = False):
        if len(image) > 1:
            raise Exception("[Onyx] Detailer (SEGS) does not allow image batches. Use Detailer (Batch SEGS) for batched inputs.")

        image = image.clone()
        enhanced_alpha_list: list[torch.Tensor] = []
        enhanced_list: list[torch.Tensor] = []
        cropped_list: list[torch.Tensor] = []
        cnet_pil_list: list[torch.Tensor] = []
        segs = segs_scale_match(segs, image.shape)
        new_segs = []

        wildcard_concat_mode = None
        if wildcard_opt is not None:
            if wildcard_opt.startswith("[CONCAT]"):
                wildcard_concat_mode = "concat"
                wildcard_opt = wildcard_opt[8:]
            wildcard_mode, wildcard_chooser = process_wildcard_for_segs(wildcard_opt)
        else:
            wildcard_mode, wildcard_chooser = None, None

        if wildcard_mode in ["ASC", "DSC", "ASC-SIZE", "DSC-SIZE"]:
            if wildcard_mode == "ASC":
                ordered_segs = sorted(segs[1], key=lambda seg: (seg.bbox[0], seg.bbox[1]))
            elif wildcard_mode == "DSC":
                ordered_segs = sorted(segs[1], key=lambda seg: (seg.bbox[0], seg.bbox[1]), reverse=True)
            elif wildcard_mode == "ASC-SIZE":
                ordered_segs = sorted(segs[1], key=lambda seg: (seg.bbox[2] - seg.bbox[0]) * (seg.bbox[3] - seg.bbox[1]))
            else:
                ordered_segs = sorted(segs[1], key=lambda seg: (seg.bbox[2] - seg.bbox[0]) * (seg.bbox[3] - seg.bbox[1]), reverse=True)
        else:
            ordered_segs = segs[1]

        if not (isinstance(model, str) and model == "DUMMY") and noise_mask_feather > 0 and "denoise_mask_function" not in model.model_options:
            model = nodes_differential_diffusion.DifferentialDiffusion().execute(model)[0]

        for index, seg in enumerate(ordered_segs):
            cropped_image = _to_tensor(_crop_ndarray4(image.cpu().numpy(), seg.crop_region))
            blurred_mask = tensor_gaussian_blur_mask(_to_tensor(seg.cropped_mask), feather)
            if bool((_to_tensor(seg.cropped_mask) == 0).all().item()):
                continue

            cropped_mask = seg.cropped_mask if noise_mask else None
            if wildcard_chooser is not None and wildcard_mode != "LAB":
                seg_seed, wildcard_item = wildcard_chooser.get(seg)
            elif wildcard_chooser is not None and wildcard_mode == "LAB":
                seg_seed, wildcard_item = None, wildcard_chooser.get(seg)
            else:
                seg_seed, wildcard_item = None, None
            seg_seed = seed + index if seg_seed is None else seg_seed

            cropped_positive = positive if isinstance(positive, str) else [[condition, {key: crop_condition_mask(value, image, seg.crop_region) if key == "mask" else value for key, value in details.items()}] for condition, details in positive]
            cropped_negative = negative if isinstance(negative, str) else [[condition, {key: crop_condition_mask(value, image, seg.crop_region) if key == "mask" else value for key, value in details.items()}] for condition, details in negative]

            if wildcard_item and wildcard_item.strip() == "[SKIP]":
                continue
            if wildcard_item and wildcard_item.strip() == "[STOP]":
                break

            original_cropped_image = cropped_image.clone()
            if not (isinstance(model, str) and model == "DUMMY"):
                enhanced_image, cnet_pils = enhance_detail(cropped_image, model, clip, vae, guide_size, guide_size_for_bbox, max_size, seg.bbox, seg_seed, steps, cfg, sampler_name, scheduler, cropped_positive, cropped_negative, denoise, cropped_mask, force_inpaint, wildcard_opt=wildcard_item, wildcard_opt_concat_mode=wildcard_concat_mode, detailer_hook=detailer_hook, refiner_ratio=refiner_ratio, refiner_model=refiner_model, refiner_clip=refiner_clip, refiner_positive=refiner_positive, refiner_negative=refiner_negative, control_net_wrapper=seg.control_net_wrapper, cycle=cycle, inpaint_model=inpaint_model, noise_mask_feather=noise_mask_feather, scheduler_func=scheduler_func_opt, vae_tiled_encode=tiled_encode, vae_tiled_decode=tiled_decode)
            else:
                enhanced_image = cropped_image
                cnet_pils = None

            if cnet_pils is not None:
                cnet_pil_list.extend(cnet_pils)

            if enhanced_image is not None:
                image = image.cpu()
                enhanced_image = enhanced_image.cpu()
                tensor_paste(image, enhanced_image, (seg.crop_region[0], seg.crop_region[1]), blurred_mask)
                enhanced_list.append(enhanced_image)
                if detailer_hook is not None:
                    image = detailer_hook.post_paste(image)

                enhanced_image_alpha = tensor_convert_rgba(enhanced_image)
                alpha_mask = tensor_resize(blurred_mask, *tensor_get_size(enhanced_image))
                tensor_putalpha(enhanced_image_alpha, alpha_mask)
                enhanced_alpha_list.append(enhanced_image_alpha)
                new_seg_image = enhanced_image.numpy()
            else:
                new_seg_image = None

            cropped_list.append(original_cropped_image)
            new_segs.append(SEG(new_seg_image, seg.cropped_mask, seg.confidence, seg.crop_region, seg.bbox, seg.label, seg.control_net_wrapper))

        image_tensor = tensor_convert_rgb(image)
        cropped_list.sort(key=lambda value: value.shape, reverse=True)
        enhanced_list.sort(key=lambda value: value.shape, reverse=True)
        enhanced_alpha_list.sort(key=lambda value: value.shape, reverse=True)
        return image_tensor, cropped_list, enhanced_list, enhanced_alpha_list, cnet_pil_list, (segs[0], new_segs)
