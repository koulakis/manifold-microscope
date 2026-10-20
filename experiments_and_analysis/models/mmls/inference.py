"""Projection using the unchanged MMLS neighbor/SVD implementation."""


def predict(projector, data):
    return projector.project(data.reshape(len(data), -1)).reshape(data.shape)
