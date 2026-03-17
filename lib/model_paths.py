from __future__ import annotations

import folder_paths


def add_model_folder_path_ext(
    folder_name: str,
    full_paths: list[str],
    extensions: set[str],
) -> None:
    if folder_name in folder_paths.folder_names_and_paths:
        return

    for full_path in full_paths:
        folder_paths.add_model_folder_path(folder_name, full_path)

    if folder_name in folder_paths.folder_names_and_paths:
        current_paths, current_extensions = folder_paths.folder_names_and_paths[folder_name]
        folder_paths.folder_names_and_paths[folder_name] = (
            current_paths,
            current_extensions | extensions,
        )
    else:
        folder_paths.folder_names_and_paths[folder_name] = (full_paths, extensions)