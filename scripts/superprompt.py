import os
import requests
import time  # 引入time模块
import torch
import transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration
import gradio as gr
from modules import scripts

extension_path = scripts.basedir()
model_folder = ""
tokenizer = ""
model = ""
num_return_sequences = 1
SEED_LIMIT_NUMPY = 2**32

def set_seed(seed: int) -> None:
    seed = int(seed) % SEED_LIMIT_NUMPY
    torch.manual_seed(seed)
    transformers.set_seed(seed)

def download_file(url, save_path):
    """从指定URL下载文件并保存到指定路径"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # 检查请求是否成功

        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 获取文件大小
        total_size = int(response.headers.get('content-length', 0))
        downloaded_size = 0

        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
                downloaded_size += len(chunk)
                # 输出下载进度
                print(f"下载进度: {downloaded_size}/{total_size} bytes", end='\r')

        print(f"文件 {os.path.basename(save_path)} 下载成功！")
    except requests.exceptions.RequestException as e:
        print(f"文件 {os.path.basename(save_path)} 下载失败：{e}")

def download_model():
    """创建文件夹并下载模型文件"""
    model_dir = os.path.join(extension_path, "superprompt-v1")
    
    # 创建文件夹
    os.makedirs(model_dir, exist_ok=True)

    # 文件列表及其对应的下载链接
    files_to_download = {
        "config.json": "https://huggingface.co/roborovski/superprompt-v1/resolve/main/config.json",
        "generation_config.json": "https://huggingface.co/roborovski/superprompt-v1/resolve/main/generation_config.json",
        "model.safetensors": "https://huggingface.co/roborovski/superprompt-v1/resolve/main/model.safetensors",
        "spiece.model": "https://huggingface.co/roborovski/superprompt-v1/resolve/main/spiece.model",
        "tokenizer.json": "https://huggingface.co/roborovski/superprompt-v1/resolve/main/tokenizer.json",
        "tokenizer_config.json": "https://huggingface.co/roborovski/superprompt-v1/resolve/main/tokenizer_config.json",
    }

    # 逐个下载文件
    for filename, url in files_to_download.items():
        save_path = os.path.join(model_dir, filename)
        if not os.path.isfile(save_path):  # 仅在文件不存在时下载
            download_file(url, save_path)
        else:
            print(f"文件 {filename} 已存在，跳过下载。")
    
    print("所有文件下载完成！")
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
        print("模型文件丢失或不完整，请手动下载https://huggingface.co/roborovski/superprompt-v1。")
        return False

    try:
        tokenizer = T5Tokenizer.from_pretrained("roborovski/superprompt-v1", legacy=False, clean_up_tokenization_spaces=False)
        model = T5ForConditionalGeneration.from_pretrained("roborovski/superprompt-v1", device_map="auto")
        return True
    except Exception as e:
        print(f"加载模型时出错: {str(e)}")
        return False
        
def generate_super_prompt(prompt, max_new_tokens=128, seed=123456):
    model_found = modelcheck()
    if model_found:
        # 设置种子值
        if seed is not None:
            set_seed(seed)  # 使用提供的种子设置函数

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True  # 启用采样
        )
        generated_prompt = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_prompt
    else:
        gr.Warning("未找到模型。请查看命令控制台信息")
        return ""
