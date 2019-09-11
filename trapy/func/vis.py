from typing import Union

import numpy as np
import plotly.graph_objects as go


def plot_3d_plotly(
        x: Union[list, np.ndarray],
        y: Union[list, np.ndarray],
        z: Union[list, np.ndarray],
        v: Union[list, np.ndarray]
):
    data = [
        go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(
                size=5,
                color=v,  # set color to an array/list of desired values
                colorscale='Viridis',  # choose a colorscale
                opacity=1.0
            )
        )
    ]

    fig = go.Figure(data)

    return fig


if __name__ == '__main__':
    import itertools

    arr = np.linspace(0, 10, 11)
    arr = (arr[1:] + arr[:-1]) / 2
    arr = np.asarray(list(itertools.product(arr, arr)))
    xyz = np.zeros((np.shape(arr)[0] + 1, 3))
    xyz[0:np.shape(arr)[0], 0] = arr[:, 0]
    xyz[0:np.shape(arr)[0], 1] = arr[:, 1]
    xyz[-1, 2] = 10

    my_fig = plot_3d_plotly(xyz[:, 0], xyz[:, 1], xyz[:, 2], xyz[:, 0])
    my_fig.show()
