import marimo

__generated_with = "0.15.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Marimo Functions: Example

    The scikit-learn library provides multiple methods for embedding high dimensional datasets
    in lower-dimensional space.
    For each method, there are a number of inputs that the user can employ to tune the resolution
    of that lower-dimensional embedding.

    This example shows how the marimo_functions wrapper can be used to select a function of interest,
    and then also let the user populate the parameters for that particular method.
    """
    )
    return


@app.cell
def _():
    # Instead of importing the scikit-learn modules,
    # import the corresponding marimo_functions wrappers
    # which also include the UI elements needed
    from marimo_functions.sklearn.manifold.tsne import TSNE
    from marimo_functions.sklearn.manifold.isomap import Isomap
    from marimo_functions.sklearn.manifold.locally_linear_embedding import LocallyLinearEmbedding
    from marimo_functions.sklearn.manifold.mds import MDS
    from marimo_functions.sklearn.manifold.spectral_embedding import SpectralEmbedding

    from marimo_functions import PickFunction
    return (
        Isomap,
        LocallyLinearEmbedding,
        MDS,
        PickFunction,
        SpectralEmbedding,
        TSNE,
    )


@app.cell
def _():
    # Import some data to work with
    from sklearn import datasets
    iris = datasets.load_iris(as_frame=True)
    return (iris,)


@app.cell
def _(
    Isomap,
    LocallyLinearEmbedding,
    MDS,
    PickFunction,
    SpectralEmbedding,
    TSNE,
):
    # Define the list of functions to choose from and 
    # give the user a dropdown menu to select one
    selected_function = (
        PickFunction(
            functions=[
                TSNE,
                MDS,
                SpectralEmbedding,
                Isomap,
                LocallyLinearEmbedding
            ]
        )
        .prompt(
            label="Select a function for dimensionality reduction",
            value="t-SNE"
        )
    )
    selected_function
    return (selected_function,)


@app.cell
def _(mo, selected_function):
    # The model used for dimensionality reduction will be set by user input
    func = selected_function.value

    # Display the description of the function
    mo.md(func.description)
    return (func,)


@app.cell
def _(func):
    # Prompt for the inputs from the user
    inputs = func.prompt()
    inputs
    return (inputs,)


@app.cell
def _(func, inputs):
    # Build a TSNE object with those user-supplied parameters
    model = func.run(**inputs.value)
    return (model,)


@app.cell
def _(func, inputs, iris, model):
    # Run the t-SNE transformation and create a DataFrame with the coordinates, along with the target names
    import pandas as pd

    _target_names_dict = dict(zip(range(3), iris["target_names"]))
    coords = (
        pd.DataFrame(
            model.fit_transform(iris["data"]),
            columns=[f"{func.name} {i+1}" for i in range(inputs.value["n_components"])],
            index=iris["data"].index
        )
        .merge(
            iris["target"].replace(to_replace=_target_names_dict),
            left_index=True,
            right_index=True
        )
        .merge(
            iris["data"],
            left_index=True,
            right_index=True
        )
    )
    return (coords,)


@app.cell
def _():
    import plotly.express as px
    return (px,)


@app.cell
def _(coords, inputs, iris, mo, px):
    # Show a 2D scatter if the user selected 2 componants
    mo.stop(inputs.value["n_components"] != 2)
    px.scatter(
        data_frame=coords,
        x=coords.columns.values[0],
        y=coords.columns.values[1],
        color="target",
        template="simple_white",
        hover_data=iris["data"].columns
    )
    return


@app.cell
def _(coords, inputs, iris, mo, px):
    # Show a 3D scatter if the user selected 3 componants
    mo.stop(inputs.value["n_components"] != 3)
    px.scatter_3d(
        data_frame=coords,
        x=coords.columns.values[0],
        y=coords.columns.values[1],
        z=coords.columns.values[2],
        color="target",
        template="simple_white",
        hover_data=iris["data"].columns
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
