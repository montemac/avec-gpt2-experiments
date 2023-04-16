# %%
# Imports and setup
from IPython import get_ipython

try:
    get_ipython().run_line_magic("reload_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")
except NameError:
    pass

from typing import List

import torch

from transformer_lens import HookedTransformer

from algebraic_value_editing import completions, hook_utils
from algebraic_value_editing.rich_prompts import RichPrompt

# We turn automatic differentiation off, to save GPU memory, as this notebook focuses on model inference not model training.
_ = torch.set_grad_enabled(False)


# %%
# Load model
# device = "cuda:0"
device = "cpu"

# model_name = "gpt2-xl"
model_name = "gpt2"
model = HookedTransformer.from_pretrained(model_name, device=device)


# %%
# Sanity check
model_description_text = """## Loading Models
HookedTransformer comes loaded with >40 open source GPT-style models.
You can load any of them in
with`HookedTransformer.from_pretrained(MODEL_NAME)`. See my explainer
for documentation of all supported models, and this table for
hyper-parameters and the name used to load them. Each model is loaded
into the consistent HookedTransformer architecture, designed to be
clean, consistent and interpretability-friendly. For this demo notebook
we'll look at GPT-2 Small, an 80M parameter model. To try the model the
model out, let's find the loss on this paragraph!"""
loss = model(model_description_text, return_type="loss")
print("Model loss:", loss)


# %%
# Test
target_activation_name: str = hook_utils.get_block_name(block_num=6)

rich_prompts: List[RichPrompt] = [
    RichPrompt(prompt="Love", coeff=10.0, act_name=target_activation_name)
]

completions.print_n_comparisons(
    prompt="Here's how I feel about you.",
    num_comparisons=15,
    model=model,
    rich_prompts=rich_prompts,
)
