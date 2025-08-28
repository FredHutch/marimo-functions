import marimo as mo
import logging
from typing import TypedDict, NotRequired, Union, Dict

# Set up logging, printing to stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def _render_input(input: Dict[str, str]) -> mo.Html:
    """
    Return a marimo UI element, with the type of element
    defined by the 'type' key of the input dictionary.
    """
    try:
        ui_type = input["type"]
    except KeyError:
        logger.info(input)
        raise ValueError("Input dictionary must contain a 'type' key.")

    return getattr(
        mo.ui,
        ui_type
    )(
        **{
            kw: val
            for kw, val in input.items()
            if kw != "type"
        }
    )


class MarimoInput(TypedDict):
    type: str
    label: NotRequired[str]
    value: Union[str, int, float]
    start: NotRequired[float]
    stop: NotRequired[float]
    step: NotRequired[float]
    placeholder: NotRequired[str]
    kind: NotRequired[str]
    max_length: NotRequired[int]
    disabled: NotRequired[bool]
    debounce: NotRequired[float]
    full_width: NotRequired[bool]
    options: NotRequired[list[str]]


class MarimoFunction:
    """
    A class representing a Marimo function.
    A function has input arguments and a run() method.
    The input arguments are defined as an attribute, and the prompt()
    method is used to return the marimo UI elements which canbe used to prompt
    user input.
    """
    inputs: Dict[str, MarimoInput] = {}
    name: str
    description: str

    @classmethod
    def prompt(cls):
        """
        Returns the marimo UI elements for prompting user input.
        """
        return mo.md("\n".join([
            "- {" + kw + "}"
            for kw in cls.inputs.keys()
        ])).batch(**{
            kw: _render_input(input)
            for kw, input in cls.inputs.items()
        })

    @classmethod
    def run(cls, **kwargs):
        """Run the function using the inputs provided by the user."""
        pass
