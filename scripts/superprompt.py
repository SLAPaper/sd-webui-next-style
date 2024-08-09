import os
import subprocess
import time  # 引入time模块
from transformers import T5Tokenizer, T5ForConditionalGeneration
import gradio as gr
from modules import scripts

extension_path = scripts.basedir()
model_folder = ""
tokenizer = ""
model = ""
num_return_sequences = 1

def download_model():
    """自动下载模型文件夹"""
    model_dir = os.path.join(extension_path, "superprompt-v1")
    
    # 克隆模型仓库
    try:
        subprocess.run(
            ["git", "clone", "https://huggingface.co/roborovski/superprompt-v1", model_dir],
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

    model_dir = os.path.join(extension_path, "superprompt-v1")
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
        tokenizer = T5Tokenizer.from_pretrained("roborovski/superprompt-v1")
        model = T5ForConditionalGeneration.from_pretrained("roborovski/superprompt-v1", device_map="auto")
        return True
    except Exception as e:
        print(f"加载模型时出错: {str(e)}")
        return False
        
def generate_super_prompt(prompt, max_new_tokens=77):
    model_found = modelcheck()
    if model_found:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
        outputs = model.generate(input_ids, max_new_tokens=max_new_tokens)
        generated_prompt = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_prompt
    else:
        gr.Warning("No Model Found. Check Command Console")
        return ""
