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
