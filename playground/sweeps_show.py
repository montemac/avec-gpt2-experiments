# %%
# Imports, etc.
import pickle

import numpy as np

import plotly.express as px

try:
    from IPython import get_ipython

    get_ipython().run_line_magic("reload_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")
except AttributeError:
    pass

# %%
# Run the sweep of completions, or load from cache
CACHE_FN = "sweeps_demo_cache_20230416T053723.pkl"
with open(CACHE_FN, "rb") as file:
    normal_df, patched_df, rich_prompts_df = pickle.load(file)

# %%
# Visualize
# Reduce the patched DataFrame
reduced_df = patched_df.groupby(["prompts", "rich_prompt_index"]).mean(
    numeric_only=True
)
reduced_joined_df = reduced_df.join(
    rich_prompts_df, on="rich_prompt_index"
).reset_index()

# Reduce the normal DataFrame
reduced_normal_df = normal_df.groupby(["prompts"]).mean(numeric_only=True)


# Plot function
def plot_col(
    data,
    col_to_plot,
    title,
    col_x="coeff",
    col_color="act_name",
    baseline_data=None,
):
    """Plot a column, with colors/facets/x set."""
    fig = px.line(
        data,
        title=title,
        color=col_color,
        y=col_to_plot,
        x=col_x,
        facet_col="prompts",
    )
    if baseline_data is not None and col_to_plot in baseline_data:
        for ii, prompt in enumerate(baseline_data.index):
            fig.add_hline(
                y=baseline_data.loc[prompt][col_to_plot],
                row=1,
                col=ii + 1,
                annotation_text="normal",
                annotation_position="bottom left",
            )
    fig.show()


# plot_col(
#     reduced_joined_df, "wedding_words_count", "Average wedding word count"
# )
# plot_col(reduced_joined_df, "loss", "Average loss")

# Exlude the extreme coeffs, likely not that interesting
reduced_joined_filt_df = reduced_joined_df[
    (reduced_joined_df["coeff"] >= -4) & (reduced_joined_df["coeff"] <= 4)
]

# Plot
plot_col(
    reduced_joined_filt_df,
    "wedding_words_count",
    "Average wedding word count",
    baseline_data=reduced_normal_df,
)
plot_col(
    reduced_joined_filt_df,
    "loss",
    "Average loss",
    baseline_data=reduced_normal_df,
)
plot_col(
    reduced_joined_filt_df,
    "wedding_words_count",
    "Average wedding word count",
    col_x="act_name",
    col_color="coeff",
    baseline_data=reduced_normal_df,
)
plot_col(
    reduced_joined_filt_df,
    "loss",
    "Average loss",
    col_x="act_name",
    col_color="coeff",
    baseline_data=reduced_normal_df,
)
