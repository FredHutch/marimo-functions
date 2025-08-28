from marimo_functions import MarimoFunction
from sklearn.manifold import LocallyLinearEmbedding as SKLLE


class LocallyLinearEmbedding(MarimoFunction):
    """
    A class representing Locally Linear Embedding (LLE).
    """
    inputs = {
        "n_neighbors": {
            "type": "number",
            "label": "Number of Neighbors",
            "value": 5,
            "start": 1,
            "stop": 100,
            "step": 1
        },
        "n_components": {
            "type": "number",
            "label": "Number of Components",
            "value": 2,
            "start": 1,
            "stop": 100,
            "step": 1
        },
        "reg": {
            "type": "number",
            "label": "Regularization",
            "value": 1e-3,
            "start": 0.0,
            "step": 1e-4
        },
        "method": {
            "type": "dropdown",
            "label": "Method",
            "value": "standard",
            "options": ["standard", "modified", "hessian", "ltsa"]
        },
        "random_state": {
            "type": "number",
            "label": "Random State",
            "value": 0
        }
    }
    name = "Locally Linear Embedding"
    description = "Locally Linear Embedding (LLE)."

    @classmethod
    def run(cls, **kwargs) -> SKLLE:
        return SKLLE(**kwargs)
