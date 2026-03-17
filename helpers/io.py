from __future__ import annotations

from typing import Callable

from comfy_api.latest import IO

ComboOptions = list[str] | list[int] | Callable[[], list[str] | list[int]]


class HashableOptionList(list):
    def __hash__(self) -> int:
        return hash(tuple(self))


def _resolve_combo_options(options: ComboOptions) -> HashableOptionList:
    return HashableOptionList(options() if callable(options) else options)


class ComboTypeInput(IO.Combo.Input):
    def __init__(
        self,
        options: ComboOptions,
        id: str,
        display_name: str = None,
        optional: bool = False,
        tooltip: str = None,
        lazy: bool = None,
    ):
        self._options = options
        super().__init__(
            id,
            options=_resolve_combo_options(options),
            display_name=display_name,
            optional=optional,
            tooltip=tooltip,
            lazy=lazy,
        )

    def get_io_type(self):
        return _resolve_combo_options(self._options)


class ComboTypeOutput(IO.Output):
    def __init__(
        self,
        options: ComboOptions,
        id: str = None,
        display_name: str = None,
        tooltip: str = None,
        is_output_list=False,
    ):  # type: ignore
        self.id = id
        self.display_name = display_name
        self.tooltip = tooltip
        self.is_output_list = is_output_list
        self.Parent = None  # type: ignore
        self._options = options

    def _resolve_options(self) -> list[str] | list[int]:
        return _resolve_combo_options(self._options)

    @property
    def io_type(self):
        return self._resolve_options()

    def get_io_type(self):
        return self._resolve_options()

    def as_dict(self):
        from comfy_api.internal import prune_dict

        return prune_dict(
            {
                "display_name": self.display_name,
                "tooltip": self.tooltip,
                "is_output_list": self.is_output_list,
            }
        )