# Copyright 2024 SLAPaper
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools as ft
import logging
import pathlib
import shutil
import typing as tg
import argparse

import gradio as gr
import torch
from modules import options, script_callbacks, scripts, shared
from modules.processing import StableDiffusionProcessing
from modules.ui_components import FormColumn, FormRow
from transformers import AutoModelForCausalLM, GPT2TokenizerFast
from transformers.generation.logits_process import LogitsProcessorList

# Constants
SEED_LIMIT_NUMPY = 2**32
NEG_INF = -8192.0

# Get model path
curr_path = pathlib.Path(__file__).parent.absolute()
model_path = curr_path / "models"
try:
    if cmd_opts.pe_model_path:
        model_path = cmd_opts.pe_model_path
except:
    pass
expansion_path = model_path / "expansion"

# Download models if necessary
def download_models():
    from torch.hub import download_url_to_file
    
    url = "https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_expansion.bin"
    file_name = "pytorch_model.bin"
    cached_file = expansion_path / file_name
    
    if not cached_file.exists():
        logging.info(f'Downloading: "{url}" to {cached_file}')
        download_url_to_file(url, str(cached_file), progress=True)


if not expansion_path.exists():
    shutil.copytree(curr_path / "expansion", expansion_path)
    download_models()

# Utility functions
def safe_str(x):
    x = str(x)
    for _ in range(16):
        x = x.replace("  ", " ")
    return x.strip(",. \r\n")

def set_seed(seed: int) -> None:
    import transformers
    seed = int(seed) % SEED_LIMIT_NUMPY
    transformers.set_seed(seed)

# Model Management
class ModelManagement:
    def __init__(self) -> None:
        try:
            # forge
            from ldm_patched.modules import model_management  
            self.load_device = model_management.text_encoder_device()
            self.offload_device = model_management.text_encoder_offload_device()

        except:
            # a1111
            from modules import devices
            self.offload_device = devices.cpu
            self.load_device = devices.get_optimal_device()

        logging.info(f"Prompt Expansion engine setting up for {self.load_device}.")

    def load(self, model: torch.nn.Module) -> torch.nn.Module:
        return model.to(self.load_device)

    def offload(self, model: torch.nn.Module) -> torch.nn.Module:
        return model.to(self.offload_device)

model_management = ModelManagement()

# Prompt Expansion Class
class PromptsExpansion:
    def __init__(self):
        self.tokenizer = GPT2TokenizerFast.from_pretrained(expansion_path, local_files_only=True)

        positive_words = (expansion_path / "positive.txt").read_text(encoding="utf8").splitlines()
        positive_words = ["Ġ" + x.lower() for x in positive_words if x != ""]

        self.logits_bias = torch.zeros((1, len(self.tokenizer.vocab)), dtype=torch.float32) + NEG_INF
        for k, v in self.tokenizer.vocab.items():
            if k in positive_words:
                self.logits_bias[0, v] = 0

        self.model = AutoModelForCausalLM.from_pretrained(expansion_path, local_files_only=True)
        self.model.eval()

    def logits_processor(self, input_ids, scores):
        with torch.inference_mode():
            assert scores.ndim == 2 and scores.shape[0] == 1
            self.logits_bias = self.logits_bias.to(scores)

            bias = self.logits_bias.clone()
            bias[0, input_ids[0].to(bias.device).long()] = NEG_INF
            bias[0, 11] = 0

            return scores + bias

    @ft.lru_cache(maxsize=1024)
    def __call__(self, prompt: str, seed: int, max_new_tokens: int, top_k: int = 100) -> str:
        if prompt == "":
            return ""

        prompt = safe_str(prompt) + ","
        set_seed(seed)

        with torch.inference_mode():
            tokenized_kwargs = self.tokenizer(prompt, return_tensors="pt")
            tokenized_kwargs.data["input_ids"] = tokenized_kwargs.data["input_ids"].to(model_management.load_device)
            tokenized_kwargs.data["attention_mask"] = tokenized_kwargs.data["attention_mask"].to(model_management.load_device)

            if max_new_tokens <= 0:
                current_token_length = int(tokenized_kwargs.data["input_ids"].shape[1])
                max_token_length = current_token_length + 75 - current_token_length % 75
                max_new_tokens = max_token_length - current_token_length

            model_management.load(self.model)
            features = self.model.generate(
                **tokenized_kwargs,
                top_k=int(top_k),
                max_new_tokens=max_new_tokens,
                do_sample=True,
                logits_processor=LogitsProcessorList([self.logits_processor]),
            )
            model_management.offload(self.model)

            response = self.tokenizer.batch_decode(features, skip_special_tokens=True)
            result = safe_str(response[0])

            return result


expansion = PromptsExpansion()

# Script Class
class PromptExpansion(scripts.Script):
    def title(self) -> str:
        return "Prompt Expansion 1.0"

    def show(self, is_img2img: bool):
        return scripts.AlwaysVisible

    def ui(self, is_img2img: bool) -> list:
        with gr.Group():
            with gr.Accordion("Fooocus V2", open=False):
                with FormRow():
                    with FormColumn(min_width=160):
                        is_enabled = gr.Checkbox(
                            value=False,
                            label="启用 Fooocus V2 模型",
                            info="使用额外的模型来扩展提示",
                        )

        return [is_enabled]

    def process(self, p: StableDiffusionProcessing, *args) -> None:
        is_enabled: bool = args[0]
        if not is_enabled:
            return

        opts = tg.cast(options.Options, shared.opts)
        max_new_tokens = 0
        if (
            "Fooocus_V2_Max_New_Tokens" in opts.data
            and opts.data["Fooocus_V2_Max_New_Tokens"] is not None
            and int(str(opts.data["Fooocus_V2_Max_New_Tokens"])) > 0
        ):
            max_new_tokens = int(str(opts.data["Fooocus_V2_Max_New_Tokens"]))

        for i, prompt in enumerate(p.all_prompts):
            p.all_prompts[i] = expansion(prompt, p.all_seeds[i], max_new_tokens)

        p.extra_generation_params["Prompt-Expansion"] = True
        p.extra_generation_params["Prompt-Expansion-Model"] = "Fooocus V2"

# UI Settings
def on_ui_settings():
    section = ("Prompt_Expansion", "Prompt Expansion")
    opts = tg.cast(options.Options, shared.opts)
    opts.add_option(
        "Fooocus_V2_Max_New_Tokens",
        shared.OptionInfo(
            default=0,
            label="Max new token length for Fooocus V2 (Set to 0 to fill up remaining tokens of 75*k)",
            component=gr.Slider,
            component_args={
                "minimum": 0,
                "maximum": 300,
                "step": 1,
            },
            section=section,
        ),
    )

script_callbacks.on_ui_settings(on_ui_settings)

# Preload (for command-line arguments)
def preload(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--pe-model-path",
        type=pathlib.Path,
        help="model path for prompt expansion",
    )