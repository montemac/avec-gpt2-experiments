# %%
# Imports and setup
try:
    get_ipython().__class__.__name__
    is_ipython = True
except:
    is_ipython = False
if is_ipython:
    get_ipython().run_line_magic("reload_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")

import funcy as fn

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import einops
from fancy_einsum import einsum
from tqdm.auto import tqdm
import random
from pathlib import Path
import plotly as py
import plotly.subplots
import plotly.graph_objects as go
import plotly.express as px
from torch.utils.data import DataLoader
import warnings
import pickle
import datetime
import os

from jaxtyping import Float, Int
from typing import List, Union, Optional, Tuple
from functools import partial
import copy

import itertools
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import dataclasses
import datasets
from IPython.display import HTML
import prettytable
from ipywidgets import Output

from transformers import pipeline

import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)  # Hooking utilities
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache

from avec_gpt2 import xvector

# %%
# Load results
fn = 'sentiment_test_results/sentiment_test_results_20230317T234616.pkl'
with open(fn, 'rb') as fl:
    res = pickle.load(fl)
    results_normal = res['results_normal']
    results_patched = res['results_patched']
    metadata = res['metadata']

# %%
# Show some stats
#print(results.groupby('normal_sentiment_label').count()['prompt']/len(results))
#print(results.groupby('patched_sentiment_label').count()['prompt']/len(results))

# TODO: fix this to match new interface of results

results_patched['normal_sentiment_is_positive'] = results_normal['normal_sentiment_label'] == 'POSITIVE'
results_patched['patched_sentiment_is_positive'] = results_patched['patched_sentiment_label'] == 'POSITIVE'
results_patched['normal_loss'] = results_normal['normal_loss']

# Reduce over num_completions
grp = results_patched.groupby(['prompt', 'coeff'])
positive_fracs_rdu = grp[['normal_sentiment_is_positive', 'patched_sentiment_is_positive']].mean()
losses_rdu = grp[['normal_loss', 'patched_loss']].mean()

# plot_df = pd.concat([positive_fracs_rdu, losses_rdu], axis='columns').stack().reset_index()
# fig = px.line(plot_df, x='coeff', y=0, color='level_2', facet_col='prompt', facet_col_wrap=2)

# Plot!
prompts = np.unique(results_patched['prompt'])
coeffs = np.unique(results_patched['coeff'])
recipe_phrases = metadata['recipe_phrases']
num_cols = 2
subplot_titles = []
for prompt in prompts:
    subplot_titles.append(f'P(positive sentiment), "{prompt}"')
    subplot_titles.append(f'Unpatched model loss, "{prompt}"')
fig = py.subplots.make_subplots(rows=len(prompts), cols=num_cols, shared_xaxes=True,
    vertical_spacing=0.2/len(prompts),
    subplot_titles=subplot_titles)

clr_seq = py.colors.DEFAULT_PLOTLY_COLORS

for pp, prompt in enumerate(prompts):
    row = pp + 1
    fig.add_trace(go.Scatter(x=coeffs, y=positive_fracs_rdu.loc[prompt]['normal_sentiment_is_positive'], 
                             line_color=clr_seq[0], name='Normal', showlegend=(pp==0)), row=row, col=1)
    fig.add_trace(go.Scatter(x=coeffs, y=positive_fracs_rdu.loc[prompt]['patched_sentiment_is_positive'],
                             line_color=clr_seq[1], name='Patched', showlegend=(pp==0)), row=row, col=1)
    fig.add_trace(go.Scatter(x=coeffs, y=losses_rdu.loc[prompt]['normal_loss'],
                             line_color=clr_seq[0], showlegend=False), row=row, col=2)
    fig.add_trace(go.Scatter(x=coeffs, y=losses_rdu.loc[prompt]['patched_loss'],
                             line_color=clr_seq[1], showlegend=False), row=row, col=2)
    def axnum2str(num):
        return '' if num < 2 else f'{num}'
    fig.update_layout({
        f'yaxis{axnum2str(2*pp+1)}_range': [0,1],
        f'yaxis{axnum2str(2*pp+2)}_range': [losses_rdu.min(),losses_rdu.max()]})

fig.update_layout({
    'title': f'Sentiment and loss over coeffs and prompts with recipe: {recipe_phrases}',
    f'xaxis{2*len(prompts)-1}_title': "coeff",
    f'xaxis{2*len(prompts)}_title': "coeff",
    'height': len(prompts)*250})
fig.update_annotations(font_size=12)
fig.show()
