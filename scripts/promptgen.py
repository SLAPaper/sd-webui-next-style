import os
import subprocess
import time  # 引入time模块
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import gradio as gr
from modules import scripts

extension_path = scripts.basedir()
model_folder = ""
tokenizer = ""
model = ""
num_return_sequences = 1

def download_model():
    """自动下载模型文件夹"""
    model_dir = os.path.join(extension_path, "distilgpt2-stable-diffusion-v2")
    
    # 克隆模型仓库
    try:
        subprocess.run(
            ["git", "clone", "https://huggingface.co/FredZhang7/distilgpt2-stable-diffusion-v2", model_dir],
            check=True,
        )
        print("模型文件夹下载成功！")
    except subprocess.CalledProcessError:
        print("模型文件夹下载失败，请手动下载。")
        return False
    return True  # 返回True表示下载成功

def modelcheck():
    global model_folder
    global tokenizer
    global model
    global num_return_sequences

    model_dir = os.path.join(extension_path, "distilgpt2-stable-diffusion-v2")
    if not os.path.exists(model_dir):
        print("模型未找到，正在下载...")
        if not download_model():
            return False
        time.sleep(2)  # 延迟两秒以确保模型文件夹完全克隆

    # 确保模型文件夹路径正确
    model_folder = model_dir
    if not os.path.exists(os.path.join(model_folder, "model.safetensors")):
        print("模型文件丢失或不完整，请检查。")
        return False

    try:
        tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model = GPT2LMHeadModel.from_pretrained('distilgpt2')
        return True
    except Exception as e:
        print(f"加载模型时出错: {str(e)}")
        return False

def generate(prompt, temperature, top_k, style_max_length, repitition_penalty, usecomma):
    model_found = modelcheck()
    if model_found:
        input_ids = tokenizer(prompt, return_tensors='pt').input_ids
        if not usecomma:
            output = model.generate(
                input_ids,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                max_length=style_max_length,
                num_return_sequences=num_return_sequences,
                repetition_penalty=repitition_penalty,
                penalty_alpha=0.6,
                no_repeat_ngram_size=1,
                early_stopping=False
            )
        else:
            output = model.generate(
                input_ids,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                max_length=style_max_length,
                num_return_sequences=num_return_sequences,
                repetition_penalty=repitition_penalty,
                early_stopping=True
            )
        return tokenizer.decode(output[0], skip_special_tokens=True)
    else:
        gr.Warning("No Model Found. Check Command Console")
        return ""
