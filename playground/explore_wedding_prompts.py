# %%
# Imports, etc.
import pickle
import textwrap

import numpy as np
import pandas as pd
import torch
from IPython.display import display

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
).to("cuda:1")


# %%
# Generate a set of RichPrompts over a range of phrases, layers and
# coeffs.  The spaces are there to manually pad so that each pair of
# phrases has the same token length (a hack that I'll fix)
LABELLED_PHRASES = {
    "original": [
        ("I talk about weddings constantly  ", 1.0),
        ("I do not talk about weddings constantly", -1.0),
    ],
    'negative space-filled<br>"weddings" kept': [
        ("I talk about weddings constantly  ", 1.0),
        ("      weddings ", -1.0),
    ],
    "negative space-filled": [
        ("I talk about weddings constantly  ", 1.0),
        ("       ", -1.0),
    ],
    "weddings token in<br>same location": [
        ("I always talk about weddings", 1.0),
        ("I never talk about weddings", -1.0),
    ],
    "do not talk<br>about weddings": [
        ("I do not talk about weddings constantly", 1.0),
        ("       ", -1.0),
    ],
    "just weddings": [
        (" weddings", 1.0),
        (" ", -1.0),
    ],
}

COEFFS = np.array([-4, -1, 1, 4])

ACT_NAMES = [prompt_utils.get_block_name(block_num=num) for num in [6, 20]]

rich_prompts_df = sweeps.make_rich_prompts(
    LABELLED_PHRASES.values(),
    act_names=ACT_NAMES,
    coeffs=COEFFS,
)

rich_prompts_df["phrases_str"] = rich_prompts_df["phrases"].astype(str)
labels_by_phrase_str = pd.Series(
    LABELLED_PHRASES.keys(),
    index=[str(vv) for vv in LABELLED_PHRASES.values()],
    name="phrase_labels",
)
rich_prompts_df = rich_prompts_df.join(labels_by_phrase_str, on="phrases_str")

# Populate a list of prompts to complete
PROMPTS = [
    "Frozen starts off with a scene about",
]

WEDDING_WORDS = [
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

# Create metrics
metrics_dict = {
    "wedding_words": metrics.get_word_count_metric(WEDDING_WORDS),
}

# Print setup stuff, for prediction gathering
print("\nPHRASES (tokenized by model for clarity)")
with pd.option_context("display.width", 100):
    for label, phrases in LABELLED_PHRASES.items():
        print("\n" + label)
        print(
            pd.DataFrame(
                [
                    MODEL.to_str_tokens(phrase_text)
                    for phrase_text, coeff in phrases
                ],
                index=[coeff for phrase_text, coeff in phrases],
            )
        )
print("\n\nCOEFFS")
print(COEFFS)
print("\n\nACT_NAMES")
print(ACT_NAMES)
print("\n\nPROMPTS (tokenized by model for clarity)")
print([MODEL.to_str_tokens(prompt) for prompt in PROMPTS])
print("\n\nWEDDING_WORDS")
print(WEDDING_WORDS)


# %%
# Run the sweep of completions, or load from cache
CACHE_FN = "explore_wedding_prompts_cache.pkl"
try:
    with open(CACHE_FN, "rb") as file:
        normal_df, patched_df, rich_prompts_df = pickle.load(file)
except FileNotFoundError:
    normal_df, patched_df = sweeps.sweep_over_prompts(
        MODEL,
        PROMPTS,
        rich_prompts_df["rich_prompts"],
        num_normal_completions=100,
        num_patched_completions=100,
        seed=0,
        metrics_dict=metrics_dict,
        temperature=1,
        freq_penalty=1,
        top_p=0.3,
    )
    with open(CACHE_FN, "wb") as file:
        pickle.dump((normal_df, patched_df, rich_prompts_df), file)

# %%
# Analyze
# Reduce data
reduced_normal_df, reduced_patched_df = sweeps.reduce_sweep_results(
    normal_df, patched_df, rich_prompts_df
)


# Plot function
# def plot_col(
#     data,
#     col_to_plot,
#     title,
#     col_x="coeff",
#     baseline_data=None,
# ):
#     """Plot a column, with colors/facets/x set."""
#     fig = px.scatter(
#         data,
#         title=title,
#         y=col_to_plot,
#         x=col_x,
#         facet_col="phrase_labels",
#         facet_row="act_name",
#     )
#     if baseline_data is not None and col_to_plot in baseline_data:
#         fig.add_hline(
#             y=baseline_data.iloc[0][col_to_plot],
#             annotation_text="normal",
#             annotation_position="bottom left",
#         )


# Plot stuff
for col, title in [
    ("wedding_words_count", "Average wedding word count"),
    ("loss", "Average loss"),
]:
    fig = sweeps.plot_sweep_results(
        reduced_patched_df,
        col,
        title,
        col_x="coeff",
        col_color=None,
        col_facet_col="phrase_labels",
        col_facet_row="act_name",
        baseline_data=reduced_normal_df,
        px_func=px.scatter,
    )
    fig.add_hline(
        y=reduced_normal_df.iloc[0][col],
        annotation_text="normal",
        annotation_position="bottom left",
    )
    fig.update_annotations(font_size=12)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_layout(height=600)
    fig.show()


# %%
# # Generate some test completions to explore
# weddings_prompts = [
#     *prompt_utils.get_x_vector(
#         prompt1="I do not talk about weddings constantly",
#         prompt2="",
#         coeff=1,
#         act_name=6,
#         pad_method="tokens_right",
#         model=MODEL,
#         custom_pad_id=MODEL.to_single_token(" "),
#     )
# ]

# completion_utils.print_n_comparisons(
#     model=MODEL,
#     prompt="Frozen starts off with a scene about",
#     tokens_to_generate=50,
#     rich_prompts=weddings_prompts,
#     num_comparisons=7,
#     seed=0,
#     temperature=1,
#     freq_penalty=1,
#     top_p=0.3,
# )
