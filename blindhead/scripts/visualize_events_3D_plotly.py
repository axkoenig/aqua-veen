import plotly.graph_objects as go
import numpy as np


def create_event_scatter(events: np.ndarray):
    """
    Returns 3D scatter of the events.
    """
    rb = {0: "blue", -1: "blue", 1: "red"}
    colors = [rb[c] for c in events[:, 3].astype(np.int16)]
    tt = events[:, 2]
    tt = (tt - np.min(tt)) / (np.max(tt) - np.min(tt))
    sc = go.Scatter3d(
        x=events[:, 0],
        y=events[:, 1],
        z=tt,
        mode="markers",
        marker=dict(size=2, color=colors, opacity=0.8),
        showlegend=False,
    )
    return sc


if __name__ == '__main__':
    height, width = 720, 1280
    events = np.random.rand(100, 4) # (n x (y, x, t, p))
    events[:, 0] *= height
    events[:, 1] *= width

    cx1 = -0.2
    cy1 = -0.1
    cz1 = 0.1
    ex1 = 1.3
    ey1 = 1.85
    ez1 = 1.4

    camera = dict(
        up=dict(x=1, y=0, z=0),
        center=dict(x=cx1, y=cy1, z=cz1),
        eye=dict(x=ex1, y=ey1, z=ez1),
    )

    data = []
    data.append(create_event_scatter(events))
    fig = go.Figure(data=data)
    xy_ratio = width / height

    fig = go.Figure(data=data)
    xy_ratio = width / height

    fig.update_layout(
        scene=go.layout.Scene(
            xaxis=go.layout.scene.XAxis(title="height", range=[0, height]),
            yaxis=go.layout.scene.YAxis(title="width", range=[0, width]),
            zaxis=go.layout.scene.ZAxis(title="time", range=[0, 1], nticks=4),
            aspectratio=dict(x=1, y=xy_ratio, z=1.5)
        ),
        scene_camera=camera,
        title="3D events",
        width=900,
        height=700,
        margin=dict(l=30, r=30, b=30, t=30, pad=50),
    )

    fig.write_html('example.html')
