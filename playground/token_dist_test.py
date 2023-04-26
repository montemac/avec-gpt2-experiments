"""Notebook exploring the next-token distribution as an evaluation tool."""

# %%
# Imports, etc.
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

from transformer_lens import HookedTransformer

from algebraic_value_editing import (
    hook_utils,
    prompt_utils,
    completion_utils,
    utils,
    logits,
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
# Run a hacky sweep over coeffs


def logits_to_probs(logits):
    return torch.distributions.Categorical(logits=logits).probs


prompt = "Frozen starts off with a scene about the"
probs_normal = (
    logits_to_probs(MODEL.forward(prompt)[0, -1, :]).detach().cpu().numpy()
)

COEFFS = np.linspace(0, 20, 21)
results = []
for coeff in tqdm(COEFFS):
    # Specify prompts and generate associated hook functions
    rich_prompts = list(
        prompt_utils.get_x_vector(
            prompt1=" weddings",
            prompt2="",
            coeff=coeff,
            act_name=6,
            model=MODEL,
            pad_method="tokens_right",
            custom_pad_id=MODEL.to_single_token(" "),
        ),
    )
    hook_fns = hook_utils.hook_fns_from_rich_prompts(
        model=MODEL,
        rich_prompts=rich_prompts,
    )

    try:
        for act_name, hook_fn in hook_fns.items():
            MODEL.add_hook(act_name, hook_fn)
        probs_mod = (
            logits_to_probs(MODEL.forward(prompt)[0, -1, :])
            .detach()
            .cpu()
            .numpy()
        )
    except e:
        raise e
    finally:
        MODEL.remove_all_hook_fns()

    # Check some specific tokens
    token_to_check = (
        torch.flatten(MODEL.to_tokens(" wedding"))[-1].detach().cpu().numpy()
    )
    wedding_prob_diff = (
        probs_mod[token_to_check] - probs_normal[token_to_check]
    )
    # print(probs_normal[token_to_check], probs_mod[token_to_check])

    # Find the tokens with the highest diff and display them

    # HACK!
    if coeff == 1.0:
        logits.plot_changed_tokens(probs_normal, probs_mod, MODEL)

    # Check the KL divergence
    kl_div = sp.stats.entropy(probs_normal, probs_mod, base=2)

    # Store results
    results.append(
        {
            "coeff": coeff,
            "kl_div": kl_div,
            "wedding_prob_diff": wedding_prob_diff,
        }
    )

# %%
# Plot results

results_df = pd.DataFrame(results)

# plot_df = (
#     results_df[["kl_div", "wedding_prob_diff"]]
#     .stack()
#     .reset_index()
#     .rename(
#         {"level_0": "original_index", "level_1": "qty", 0: "value"},
#         axis="columns",
#     )
# ).join(results_df["coeff"], on="original_index")
# px.line(plot_df, x="coeff", y="value", color="qty").show()

fig = py.subplots.make_subplots(
    rows=2,
    cols=2,
    subplot_titles=[
        'Increase in prob of " wedding" token',
        "KL divergence of modified vs original dist",
    ],
)
fig.add_trace(
    go.Scatter(x=results_df["coeff"], y=results_df["wedding_prob_diff"]),
    row=1,
    col=1,
)
fig.update_layout(xaxis_title="coeff", yaxis_title='" wedding" prob incr')
fig.add_trace(
    go.Scatter(x=results_df["coeff"], y=results_df["kl_div"]),
    row=1,
    col=2,
)
fig.update_layout(xaxis2_title="coeff", yaxis2_title="KL divergence")
fig.add_trace(
    go.Scatter(
        x=results_df["wedding_prob_diff"],
        y=results_df["kl_div"],
        mode="lines+markers",
    ),
    row=2,
    col=1,
)
fig.update_layout(
    xaxis3_title='" wedding" prob incr', yaxis3_title="KL divergence"
)
fig.update_layout(height=600)
fig.show()


# %%
# Sanity check completions
rich_prompts = list(
    prompt_utils.get_x_vector(
        prompt1=" weddings",
        prompt2="",
        coeff=1.0,
        act_name=6,
        model=MODEL,
        pad_method="tokens_right",
        custom_pad_id=MODEL.to_single_token(" "),
    ),
)

completion_utils.print_n_comparisons(
    model=MODEL,
    prompt="Frozen starts off with a scene about",
    tokens_to_generate=50,
    rich_prompts=rich_prompts,
    num_comparisons=7,
    seed=0,
    temperature=1,
    freq_penalty=1,
    top_p=0.3,
)
