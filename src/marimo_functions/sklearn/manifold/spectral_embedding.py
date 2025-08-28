from marimo_functions import MarimoFunction
from sklearn.manifold import SpectralEmbedding as SKSpectralEmbedding


class SpectralEmbedding(MarimoFunction):
    """
    A class representing Spectral Embedding.
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
        "affinity": {
            "type": "dropdown",
            "label": "Affinity",
            "value": "nearest_neighbors",
            "options": ["nearest_neighbors", "rbf"]
        },
        "gamma": {
            "type": "number",
            "label": "Gamma (for rbf)",
            "value": None,
            "start": 0.0
        },
        "random_state": {
            "type": "number",
            "label": "Random State",
            "value": 0
        }
    }
    name = "Spectral Embedding"
    description = """
    Forms an affinity matrix given by the specified function and
    applies spectral decomposition to the corresponding graph laplacian.
    The resulting transformation is given by the value of the eigenvectors
    for each data point.
    """

    @classmethod
    def run(cls, **kwargs) -> SKSpectralEmbedding:
        return SKSpectralEmbedding(**kwargs)
