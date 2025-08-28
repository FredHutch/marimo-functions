from marimo_functions import MarimoFunction
from sklearn.manifold import MDS as SKMDS


class MDS(MarimoFunction):
    """
    A class representing Multidimensional Scaling (MDS).
    """
    inputs = {
        "n_components": {
            "type": "number",
            "label": "Number of Components",
            "value": 2,
            "start": 1,
            "stop": 100,
            "step": 1
        },
        "metric": {
            "type": "dropdown",
            "label": "Metric",
            "value": True,
            "options": [True, False]
        },
        "max_iter": {
            "type": "number",
            "label": "Max Iterations",
            "value": 300,
            "start": 1,
            "step": 1
        },
        "eps": {
            "type": "number",
            "label": "Epsilon",
            "value": 1e-3,
            "start": 0.0,
            "step": 1e-4
        },
        "random_state": {
            "type": "number",
            "label": "Random State",
            "value": 0
        }
    }
    name = "MDS"
    description = "Multidimensional Scaling."

    @classmethod
    def run(cls, **kwargs) -> SKMDS:
        return SKMDS(**kwargs)
