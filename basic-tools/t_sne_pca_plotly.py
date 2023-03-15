# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 2023

@author: David

Practice t-SNE and maybe PCA dimensionality reduction using Plotly.

NOTE FOR FUTURE DAVID: UMAP STUFF IS IN dnn_noise_generation.py
YOU'RE WELCOME!
"""

#######################################################################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy
import itertools

import plotly.express as px
from sklearn.manifold import TSNE

#######################################################################

#Run text case from plotly to figure out what we're working with...

def test_function():
    df = px.data.iris()
    features = ["sepal_width", "sepal_length", "petal_width", "petal_length"]
    fig = px.scatter_matrix(df, dimensions=features, color="species")
    fig.show()

def test_tsne():
    df = px.data.iris()
    print(df)

    features = df.loc[:, :'petal_width']
    print(features)

    tsne = TSNE(n_components=2, random_state=0)
    projections = tsne.fit_transform(features)

    fig = px.scatter(projections, x=0, y=1, color=df.species, labels={'color': 'species'})
    #fig.show()

def test_tsne_3d():
    df = px.data.iris()

    features = df.loc[:, :'petal_width']

    tsne = TSNE(n_components=3, random_state=0)
    projections = tsne.fit_transform(features, )

    fig = px.scatter_3d(
        projections, x=0, y=1, z=2,
        color=df.species, labels={'color': 'species'})
    fig.update_traces(marker_size=8)
    fig.show()
    
#######################################################################

#test_function()
test_tsne()