from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import comfy.model_management
import folder_paths
import torch

from .model_paths import add_model_folder_path_ext

_SAM3_ENV_VAR = "ONYX_SAM3_SRC_PATH"
_SAM3_BPE_FILE = "bpe_simple_vocab_16e6.txt.gz"
_DEFAULT_SAM3_MODEL = "sam3.pt"
_SAM3_CACHE: dict[tuple[str, str], "Sam3Runtime"] = {}
_SUPPORTED_SAM3_EXTENSIONS = getattr(
    folder_paths,
    "supported_pt_extensions",
    {".pt", ".pth", ".bin", ".ckpt", ".safetensors"},
)

add_model_folder_path_ext(
    "sam3",
    [os.path.join(folder_paths.models_dir, "sam3")],
    _SUPPORTED_SAM3_EXTENSIONS,
)


@dataclass(slots=True)
class Sam3Runtime:
    processor: Any
    device: torch.device
    device_str: str
    checkpoint_path: str
    model_name: str


def get_sam3_model_options() -> list[str]:
    model_names = list(folder_paths.get_filename_list("sam3"))
    return model_names or [_DEFAULT_SAM3_MODEL]


def resolve_sam3_device(device_choice: str) -> torch.device:
    selection = (device_choice or "Auto").strip().upper()
    auto_device = comfy.model_management.get_torch_device()
    if selection == "CPU":
        return torch.device("cpu")
    if selection == "GPU":
        if auto_device.type != "cuda":
            raise RuntimeError("GPU unavailable for SAM3. Choose CPU or Auto instead.")
        return torch.device("cuda")
    return torch.device("cuda" if auto_device.type == "cuda" else "cpu")


def load_sam3_runtime(model_name: str, device_choice: str) -> Sam3Runtime:
    torch_device = resolve_sam3_device(device_choice)
    device_str = "cuda" if torch_device.type == "cuda" else "cpu"
    checkpoint_path = resolve_sam3_checkpoint_path(model_name)
    cache_key = (_normalize_path(checkpoint_path), device_str)
    runtime = _SAM3_CACHE.get(cache_key)
    if runtime is not None:
        return runtime

    build_sam3_image_model, sam3_processor_cls, sam3_package = _import_sam3_runtime()
    bpe_path = resolve_sam3_bpe_path(sam3_package)
    model = build_sam3_image_model(
        bpe_path=bpe_path,
        device=device_str,
        eval_mode=True,
        checkpoint_path=checkpoint_path,
        load_from_HF=False,
        enable_segmentation=True,
        enable_inst_interactivity=False,
    )
    processor = sam3_processor_cls(model, device=device_str)
    runtime = Sam3Runtime(
        processor=processor,
        device=torch_device,
        device_str=device_str,
        checkpoint_path=checkpoint_path,
        model_name=model_name,
    )
    _SAM3_CACHE[cache_key] = runtime
    return runtime


def resolve_sam3_checkpoint_path(model_name: str) -> str:
    requested_name = (model_name or _DEFAULT_SAM3_MODEL).strip()
    if os.path.isabs(requested_name) and os.path.isfile(requested_name):
        return requested_name

    resolved = folder_paths.get_full_path("sam3", requested_name)
    if resolved and os.path.isfile(resolved):
        return resolved

    direct_path = Path(folder_paths.models_dir) / "sam3" / requested_name
    if direct_path.is_file():
        return str(direct_path)

    raise FileNotFoundError(
        "SAM3 checkpoint not found. Place the checkpoint in "
        f"'{Path(folder_paths.models_dir) / 'sam3'}' or provide an absolute path. Tried '{requested_name}'."
    )


def resolve_sam3_bpe_path(sam3_package: Any | None = None) -> str:
    candidate_paths: list[Path] = []
    if sam3_package is not None and getattr(sam3_package, "__file__", None):
        candidate_paths.append(Path(sam3_package.__file__).resolve().parent / "assets" / _SAM3_BPE_FILE)
    for package_dir in _candidate_package_dirs():
        candidate_paths.append(package_dir / "assets" / _SAM3_BPE_FILE)

    for path in _dedupe_paths(candidate_paths):
        if path.is_file():
            return str(path)

    raise FileNotFoundError(
        "SAM3 tokenizer assets are missing. Expected to find "
        f"'{_SAM3_BPE_FILE}' in the installed 'sam3' package, the source tree pointed to by '{_SAM3_ENV_VAR}', "
        f"or a real source checkout under '{Path(folder_paths.models_dir) / 'sam3'}'."
    )


def _candidate_package_dirs() -> list[Path]:
    package_dirs: list[Path] = []
    env_value = os.environ.get(_SAM3_ENV_VAR)
    if env_value:
        env_path = Path(env_value)
        package_dirs.append(env_path if env_path.name == "sam3" else env_path / "sam3")

    models_sam3_dir = Path(folder_paths.models_dir) / "sam3"
    if (models_sam3_dir / "__init__.py").is_file():
        package_dirs.append(models_sam3_dir)

    return _dedupe_paths(package_dirs)


def _import_sam3_runtime():
    last_error: Exception | None = None
    try:
        return _import_sam3_modules()
    except Exception as exc:
        last_error = exc

    for package_dir in _candidate_package_dirs():
        if package_dir.exists():
            sys_path = str(package_dir.parent)
            if sys_path not in sys.path:
                sys.path.insert(0, sys_path)

    try:
        return _import_sam3_modules()
    except Exception as exc:
        raise RuntimeError(
            "Unable to import the SAM3 runtime. Install the 'sam3' package, point '"
            f"{_SAM3_ENV_VAR}' at a SAM3 source checkout, or place a real SAM3 package checkout under "
            f"'{Path(folder_paths.models_dir) / 'sam3'}'."
        ) from (exc if exc is not None else last_error)


def _import_sam3_modules():
    build_sam3_image_model = importlib.import_module("sam3.model_builder").build_sam3_image_model
    sam3_processor_cls = importlib.import_module("sam3.model.sam3_image_processor").Sam3Processor
    sam3_package = importlib.import_module("sam3")
    return build_sam3_image_model, sam3_processor_cls, sam3_package


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        normalized = _normalize_path(path)
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(path)
    return deduped


def _normalize_path(path: str | Path) -> str:
    return os.path.normcase(os.path.abspath(str(path)))