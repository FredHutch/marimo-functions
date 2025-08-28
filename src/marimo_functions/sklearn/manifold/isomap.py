from sklearn.manifold import Isomap as SKIsomap
from marimo_functions import MarimoFunction
import scipy.spatial.distance


class Isomap(MarimoFunction):

    inputs = {
        "n_components": {
            "type": "number",
            "label": "Number of Components",
            "value": 2,
            "start": 1,
            "stop": 100,
            "step": 1
        },
        "n_neighbors": {
            "type": "number",
            "label": "Number of Neighbors",
            "value": 5,
            "start": 1,
            "stop": 100,
            "step": 1
        },
        "radius": {
            "type": "number",
            "label": "Radius",
            "start": 0.0,
            "value": None
        },
        "eigen_solver": {
            "type": "dropdown",
            "value": "auto",
            "options": [
                "auto",
                "arpack",
                "lobpcg"
            ]
        },
        "tol": {
            "type": "number",
            "label": "Tolerance",
            "value": 0.0,
            "start": 0.0
        },
        "neighbors_algorithm": {
            "type": "dropdown",
            "label": "Neighbors Algorithm",
            "value": "auto",
            "options": [
                "auto",
                "ball_tree",
                "kd_tree",
                "brute"
            ]
        },
        "metric": {
            "type": "dropdown",
            "label": "Metric",
            "value": "euclidean",
            "options": scipy.spatial.distance.__all__ + ["precomputed"]
        }
    }
    name = "Isomap"
    description = "Non-linear dimensionality reduction through Isometric Mapping"

    @classmethod
    def run(cls, **kwargs):
        """Run the Isomap algorithm."""
        if kwargs.get("n_neighbors") is not None and kwargs.get("radius") is not None:
            kwargs.pop("radius")
        return SKIsomap(**kwargs)
