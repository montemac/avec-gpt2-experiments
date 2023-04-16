"""Basic demonstration of sweeps and metrics operation."""

# %%
# Imports, etc.
import pickle
import datetime

import numpy as np
import torch

import plotly.express as px

from transformer_lens import HookedTransformer

from algebraic_value_editing import (
    sweeps,
    metrics,
    prompt_utils,
    completion_utils,
)

try:
    from IPython import get_ipython

    get_ipython().run_line_magic("reload_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")
except AttributeError:
    pass

# Disable gradients to save memory during inference
_ = torch.set_grad_enabled(False)

# %%
# Load a model
MODEL = HookedTransformer.from_pretrained(
    model_name="gpt2-xl", device="cpu"
).to("cuda:0")


# %%
# Generate a set of RichPrompts over a range of phrases, layers and
# coeffs
# TODO: need to find a way to add padding specifications to these sweep inputs
rich_prompts_df = sweeps.make_rich_prompts(
    [
        [
            ("I talk about weddings constantly  ", 1.0),
            ("I do not talk about weddings constantly", -1.0),
        ]
    ],
    [
        prompt_utils.get_block_name(block_num=num)
        for num in range(0, len(MODEL.blocks), 4)
    ],
    np.linspace(-5, 5, 21, endpoint=True),
)

# %%
# Populate a list of prompts to complete
prompts = [
    "I went up to my friend and said",
    "Frozen starts off with a scene about",
]

# %%
# Create metrics
metrics_dict = {
    "wedding_words": metrics.get_word_count_metric(
        [
            "wedding",
            "weddings",
            "wed",
            "marry",
            "married",
            "marriage",
            "bride",
            "groom",
            "honeymoon",
        ]
    ),
}


# %%
# Run the sweep of completions, or load from cache
cache_fn = datetime.datetime.utcnow().strftime(
    "sweeps_demo_cache_%Y%m%dT%H%M%S.pkl"
)

normal_df, patched_df = sweeps.sweep_over_prompts(
    MODEL,
    prompts,
    rich_prompts_df["rich_prompts"],
    num_normal_completions=100,
    num_patched_completions=100,
    seed=0,
    metrics_dict=metrics_dict,
    temperature=1,
    freq_penalty=1,
    top_p=0.3,
)
with open(cache_fn, "wb") as file:
    pickle.dump((normal_df, patched_df, rich_prompts_df), file)
