# %%
# Imports, etc
import pickle
import textwrap

import numpy as np
import pandas as pd
import scipy as sp
import torch
from tqdm.auto import tqdm
from IPython.display import display
import plotly.express as px
import plotly.graph_objects as go
import plotly as py
import plotly.subplots
import nltk
import nltk.data

from transformer_lens import HookedTransformer

from algebraic_value_editing import (
    hook_utils,
    prompt_utils,
    utils,
    completion_utils,
    metrics,
    sweeps,
)

utils.enable_ipython_reload()

# Disable gradients to save memory during inference
_ = torch.set_grad_enabled(False)


# %%
# Load a model
MODEL: HookedTransformer = HookedTransformer.from_pretrained(
    model_name="gpt2-xl", device="cpu"
).to("cuda:0")


# %%
# Test two options: text with string prompt prepended, text with
# spaces prepended and prompt added as an x-vector at layer 0.  I think
# they should result in identical model behavior?
text = " Frozen starts off with a scene about the wedding"

# Unmodified network with space-padded input
original_text = " " + text
print("Original tokens:")
print(MODEL.to_str_tokens(original_text))
print(MODEL.to_tokens(original_text))
original_logits, original_loss = MODEL.forward(
    original_text, return_type="both", loss_per_token=True
)

# Normal prompting
prompt_phrase = " weddings"
prompted_text = prompt_phrase + text
print("Prompting tokens:")
print(MODEL.to_str_tokens(prompted_text))
print(MODEL.to_tokens(prompted_text))
prompted_logits, prompted_loss = MODEL.forward(
    prompted_text, return_type="both", loss_per_token=True
)

# Space-padding and layer-0 injection
rich_prompts = [
    *prompt_utils.get_x_vector(
        prompt1=" weddings",
        prompt2="",
        coeff=1.0,
        act_name=6,
        model=MODEL,
        pad_method="tokens_right",
        custom_pad_id=MODEL.to_single_token(" "),
    ),
]
injected_text = text
print("Injection tokens:")
print(MODEL.to_str_tokens(injected_text))
print(MODEL.to_tokens(injected_text))
injected_logits, injected_loss = hook_utils.forward_with_rich_prompts(
    model=MODEL,
    rich_prompts=rich_prompts,
    input=injected_text,
    return_type="both",
    loss_per_token=True,
    injection_mode="pad",
)

# Check the results are equal
# print(
#     f"Prompted and injected logits match: {torch.allclose(prompted_logits, injected_logits)}"
# )

# Plot the losses at each token position
px.line(
    pd.concat(
        [
            pd.DataFrame(
                {
                    "loss": loss_tensor.detach().cpu().numpy().squeeze(),
                    "mode": mode,
                }
            )
            for mode, loss_tensor in {
                "original": original_loss,
                "prompted": prompted_loss,
                "injected": injected_loss,
            }.items()
        ]
    ).reset_index(names="pos"),
    x="pos",
    y="loss",
    color="mode",
).show()

# %%
# Try some completions with space padding
# (Update: doesn't seem to work that well)
# rich_prompts = [
#     *prompt_utils.get_x_vector(
#         prompt1=" love",
#         prompt2=" hate",
#         coeff=5,
#         act_name=6,
#         model=MODEL,
#         pad_method="tokens_right",
#         custom_pad_id=MODEL.to_single_token(" "),
#     ),
# ]

# completion_utils.print_n_comparisons(
#     model=MODEL,
#     # prompt="I hate you because",
#     prompt="  I hate you because",
#     tokens_to_generate=50,
#     rich_prompts=rich_prompts,
#     num_comparisons=7,
#     seed=0,
#     temperature=1,
#     freq_penalty=1,
#     top_p=0.3,
# )
