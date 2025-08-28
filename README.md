# marimo-functions
Flexible interface for defining functions with marimo UI inputs 

### What is a "Marimo Function"

When running analysis in marimo, it is common to have a set of inputs
which are all mapped as inputs to a particular function.
The `marimo-functions` library provides a simple framework for
implementing this pattern of user interaction.

The `MarimoFunction` class has an `inputs` attribute which maps
keyword arguments to marimo UI elements.
The `type` attribute of each `input` specifies the UI element,
and all of the remaining kwargs are passed to the constructor.

The `run()` method then takes the kwargs provided by the user
and runs any method that consumes them appropriately.

For example:

```{python}

from marimo_functions import MarimoFunction
from sklearn.manifold import TSNE as SKTSNE


class TSNE(MarimoFunction):
    """
    A class representing a t-SNE (t-distributed Stochastic Neighbor Embedding) function.
    """
    inputs = {
        "n_components": {
            "type": "number",
            "label": "Number of components",
            "value": 2,
            "start": 1,
            "step": 1
        },
        "perplexity": {
            "type": "number",
            "label": "Perplexity",
            "value": 30,
            "start": 5,
            "stop": 50,
            "step": 1
        },
        "early_exaggeration": {
            "type": "number",
            "label": "Early exaggeration",
            "value": 12.0,
            "start": 1.0,
            "stop": 100.0,
            "step": 0.1
        },
        "random_state": {
            "type": "number",
            "label": "Random State",
            "value": 0
        }
    }
    name = "t-SNE"
    description = "T-distributed Stochastic Neighbor Embedding."

    @classmethod
    def run(cls, **kwargs) -> SKTSNE:
        return SKTSNE(**kwargs)

```

### Using a Function

To make use of a function in marimo, simply prompt the user for the inputs:

```{python}
# func: MarimoFunction
inputs = func.prompt()
inputs
```

> Note that by using `inputs` as the last line in the marimo cell, it is
> presented as a UI element to the user.

And then run the function using the values provided by the user:

```{python}
output = func.run(**inputs.value)
```

### Picking a Function

In addition to the single function, this library provides a mechanism
for selecting a function of interest from a dropdown menu.

```{python}
from marimo_functions import PickFunction

selected_function = (
    PickFunction(
        functions=[
            FunctionA,
            FunctionB,
            FunctionC,
            FunctionD,
            FunctionE
        ]
    )
    .prompt(
        label="Select a function",
        value="Function A"
    )
)
selected_function
```

The function selected by the user can be populated by the `value` attribute.

```{python}
func = selected_function.value

# Display the description of the function
mo.md(func.description)
```

## Interactive Example

To run a marimo notebook displaying this functionality:

1. Set up the local environment (`uv sync`)
2. Launch the notebook (`uv run marimo edit example.py`)
