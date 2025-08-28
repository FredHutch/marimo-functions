from typing import List
from marimo_functions import MarimoFunction
import marimo as mo


class PickFunction:
    functions: List[MarimoFunction]

    def __init__(self, functions: List[MarimoFunction]):
        self.functions = functions

    def prompt(self, label: str, value=None):
        return mo.ui.dropdown(
            label=label,
            options={func.name: func for func in self.functions},
            value=value
        )
