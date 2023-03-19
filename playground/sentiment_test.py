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

# We turn automatic differentiation off, to save GPU memory, as this notebook focuses on model inference not model training.
_ = torch.set_grad_enabled(False)


# %%
# Load model
device = "cuda:1" if torch.cuda.is_available() else "cpu"

model_name = "gpt2-xl"
print(f'Loading model {model_name}')
model = HookedTransformer.from_pretrained(model_name, device=device)


# %%
# Functions for sentiment analysis testing

sentiment_pipeline = pipeline("sentiment-analysis")

def get_sentiment(strs):
    return pd.DataFrame(sentiment_pipeline(list(strs)))

def add_sentiment_to_results(results):
    for col in results.columns:
        if 'completion' in col:
            sentiment_df = get_sentiment(results[col])
            base_name = col.split('_')[0]
            results[f'{base_name}_sentiment_label'] = sentiment_df['label']
            results[f'{base_name}_sentiment_score'] = sentiment_df['score']
    return results
    

# %%
num_completions = 100
num_normal_completions = 200
completion_length = 50

# prompts = [
#     'I feel',
#     'The best way to describe my feelings for you is',
#     "I'm not feeling good about you because",
#     "I'm feeling good about you because",
#     'The smell of coffee in the morning',
#     'Roses are red, violets are blue',
#     'Cilantro is',
#     'A cold shower',
#     'Really hot weather',
#     ]

# coeffs = np.arange(1,11,1)
# coeffs = np.concatenate([-coeffs, coeffs])

# recipe_phrases = ("Love", "Hate")
# layer = 15
# temperature = 1
# freq_penalty = 1
# top_p = .3

prompts = [
    'I feel',
    "Yesterday, my dog died. Today, I got denied for a raise. I'm feeling",
    "My day is going",
    'The smell of coffee in the morning',
    'My mood is best described as',
    ]

coeffs = np.array([1, 2, 5, 10, 100, 200, 500, 1000, 2000])
coeffs = np.concatenate([-coeffs, coeffs])

recipe_phrases = ("Happy", "")
layer = 20
temperature = 1
freq_penalty = 1
top_p = .3
random_seed = 42

# Run the experiment
results_normal_all = []
results_patched_all = []
for prompt in tqdm(prompts):
    # Get normal completions for this prompt
    normal_completions, normal_loss, _ = xvector.complete_prompt_normal(model, 
        prompt=[prompt]*num_normal_completions, random_seed=random_seed,
        completion_length=completion_length, temperature=temperature, freq_penalty=freq_penalty, top_p=top_p)
    results_normal_all.append(pd.DataFrame({
        'prompt': prompt,
        'normal_completion': normal_completions,
        'normal_loss': normal_loss}))

    # Get patched completions
    for coeff in tqdm(coeffs):
        # Get completions and associated losses
        results_this = xvector.complete_prompt_with_x_vector(model=model, recipe=[(recipe_phrases, coeff)], 
            prompt=[prompt]*num_completions, random_seed=random_seed,
            completion_length=completion_length, layer_num=layer, temperature=temperature, freq_penalty=freq_penalty, top_p=top_p)
        results_this['coeff'] = coeff
        results_patched_all.append(results_this)

results_normal = pd.concat(results_normal_all).reset_index()
results_patched = pd.concat(results_patched_all).reset_index()
metadata = {
    'recipe_phrases': recipe_phrases,
    'layer': layer,
    'temperature': temperature,
    'freq_penalty': freq_penalty,
    'top_p': top_p,
    'random_seed': random_seed}

# Add sentiment data
print('Adding sentiment...')
add_sentiment_to_results(results_normal)
add_sentiment_to_results(results_patched)
print('Done')

# Save results
utc = datetime.datetime.utcnow()
save_fn = utc.strftime('sentiment_test_results_%Y%m%dT%H%M%S.pkl')
with open(os.path.join('sentiment_test_results', save_fn), 'wb') as fl:
    pickle.dump({'results_normal': results_normal, 'results_patched': results_patched,
                 'metadata': metadata}, fl)

