# %%
# Imports and setup
%reload_ext autoreload
%autoreload 2

import torch
import numpy as np
import pandas as pd
from fancy_einsum import einsum
from tqdm.auto import tqdm
from pathlib import Path
import plotly as py
import plotly.subplots
import plotly.graph_objects as go
import plotly.express as px
import pickle
from time import perf_counter

from transformer_lens import HookedTransformer

from avec_gpt2 import xvector

# We turn automatic differentiation off, to save GPU memory, as this notebook focuses on model inference not model training.
_ = torch.set_grad_enabled(False)


# %%
# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "gpt2-xl"
model = HookedTransformer.from_pretrained(model_name, device=device)


# %%
# Time some batches

class catchtime:
    def __enter__(self):
        self.time = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.time = perf_counter() - self.time
        self.readout = f'Time: {self.time:.3f} seconds'
        print(self.readout)

num_compl_to_test = [300]
prompt = 'The smell of coffee in the morning'
coeff = 2
recipe_phrases = ("Love", "Hate")
layer = 15
temperature = 1
freq_penalty = 1
top_p = .3

for num_completions in num_compl_to_test:
    print(num_completions)
    with catchtime() as t:
        results = xvector.complete_prompt_with_x_vector(model=model, recipe=[(recipe_phrases, coeff)], 
            prompt=[prompt]*num_completions, 
            completion_length=25, layer_num=layer, temperature=temperature, freq_penalty=freq_penalty, top_p=top_p)
