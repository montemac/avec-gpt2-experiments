# %%
# Imports and setup
%reload_ext autoreload
%autoreload 2

import funcy as fn

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import einops
from fancy_einsum import einsum
import tqdm.auto as tqdm
import random
from pathlib import Path
import plotly.express as px
from torch.utils.data import DataLoader
import warnings

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
device = "cuda:0"

model_name = "gpt2-xl"
model = HookedTransformer.from_pretrained(model_name, device=device)


# %%
# Sanity check
model_description_text = """## Loading Models
HookedTransformer comes loaded with >40 open source GPT-style models. You can load any of them in with `HookedTransformer.from_pretrained(MODEL_NAME)`. See my explainer for documentation of all supported models, and this table for hyper-parameters and the name used to load them. Each model is loaded into the consistent HookedTransformer architecture, designed to be clean, consistent and interpretability-friendly. 
For this demo notebook we'll look at GPT-2 Small, an 80M parameter model. To try the model the model out, let's find the loss on this paragraph!"""
loss = model(model_description_text, return_type="loss")
print("Model loss:", loss)


# %%
# Try making an x-vector from lots of phrases?
# phrases = [
#     ['I am helpful', 'It is good to always be helpful', 'Helping others is important', 
#      'Being helpful is the best thing', 'I will help someone today', 'You will be helpful'],
#     ['I am annoying', 'Never be helpful', 'Being rude and annoying others is important', 
#      'being helpful is the worst thing', 'I will never help anyone', 'You will be annoying']    
# ]

phrases = ['Want to stay alive', 'Okay with dying']

x_vector_def = (phrases, 5)
act_name = utils.get_act_name("resid_pre", 15)

#def get_x_vector_centroids(model, x_vector_def, act_name):
phrases_all, coeff = x_vector_def

# Embed to tokens
a_tokens, b_tokens = [model.to_tokens(strX) for strX in phrases_all]

# Pad to make sure token seqs are the same length
if a_tokens.shape != b_tokens.shape:
    SPACE_TOKEN = model.to_tokens(' ')[0, -1]
    len_diff = a_tokens.shape[-1] - b_tokens.shape[-1]
    if len_diff > 0: # Add to b_tokens
        b_tokens = torch.tensor(b_tokens[0].tolist() + [SPACE_TOKEN] * abs(len_diff), 
                                dtype=torch.int64, device=model.cfg.device).unsqueeze(0)
    else: 
        a_tokens = torch.tensor(a_tokens[0].tolist() + [SPACE_TOKEN] * abs(len_diff), 
                                dtype=torch.int64, device=model.cfg.device).unsqueeze(0)
assert a_tokens.shape == b_tokens.shape, f"Need same shape to compute an X-vector; instead, we have strA shape of {a_tokens.shape} and baseline shape of {b_tokens.shape}"

# Run forward passes
_, a_cache = model.run_with_cache(a_tokens, names_filter=lambda ss: ss==act_name)
_, b_cache = model.run_with_cache(b_tokens, names_filter=lambda ss: ss==act_name)

# Get centroids and diff
# WHAT THE F*** IS GOING ON HERE?  NEVER USES THIS VECTOR???
x_vector = (coeff*(a_cache[act_name].mean(axis=0) - b_cache[act_name].mean(axis=0))).clone()

#x_vectors.append(coeff*(a_cache[act_name] - b_cache[act_name]))

#x_vector = xvector.get_x_vector(model, [(("Want to stay alive", "Okay with dying"), 5)], act_name)

xvector.print_n_comparisons(num_comparisons=5, model=model, x_vector=x_vector,
                    prompt='Can you help me find my keys?', completion_length=85,
                    layer_num=15,  temperature=1, freq_penalty=1, top_p=.3, random_seed=42)