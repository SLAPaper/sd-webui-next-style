import os
import requests
import time  # 引入time模块
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import gradio as gr
from modules import scripts

extension_path = scripts.basedir()
model_folder = ""
tokenizer = ""
model = ""
num_return_sequences = 1

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
    model_dir = os.path.join(extension_path, "distilgpt2-stable-diffusion-v2")
    
    # 创建文件夹
    os.makedirs(model_dir, exist_ok=True)

    # 文件列表及其对应的下载链接
    files_to_download = {
        "config.json": "https://huggingface.co/FredZhang7/distilgpt2-stable-diffusion-v2/resolve/main/config.json",
        "model.safetensors": "https://huggingface.co/FredZhang7/distilgpt2-stable-diffusion-v2/resolve/main/model.safetensors",
        #"pytorch_model.bin": "https://huggingface.co/FredZhang7/distilgpt2-stable-diffusion-v2/resolve/main/pytorch_model.bin",
        "tokenizer.json": "https://huggingface.co/FredZhang7/distilgpt2-stable-diffusion-v2/resolve/main/tokenizer.json",
        "training_args.bin": "https://huggingface.co/FredZhang7/distilgpt2-stable-diffusion-v2/resolve/main/training_args.bin",
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

    model_dir = os.path.join(extension_path, "distilgpt2-stable-diffusion-v2")
    if not os.path.exists(model_dir):
        print("模型未找到，正在下载...")
        if not download_model():
            return False
        time.sleep(2)  # 延迟两秒以确保模型文件夹完全克隆

    # 确保模型文件夹路径正确
    model_folder = model_dir
    if not os.path.exists(os.path.join(model_folder, "model.safetensors")):
        print("模型文件丢失或不完整，请手动下载https://huggingface.co/FredZhang7/distilgpt2-stable-diffusion-v2。")
        return False

    try:
        tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2', clean_up_tokenization_spaces=False)
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model = GPT2LMHeadModel.from_pretrained('distilgpt2')
        return True
    except Exception as e:
        print(f"加载模型时出错: {str(e)}")
        return False

def generate(prompt, temperature, top_k, style_max_length, repetition_penalty, usecomma):
    model_found = modelcheck()
    if model_found:
        input_ids = tokenizer(prompt, return_tensors='pt').input_ids
        # 设置 attention_mask，以指示哪些输入标记是有效的
        attention_mask = input_ids.ne(tokenizer.pad_token_id)
        if not usecomma:
            output = model.generate(
                input_ids,
                attention_mask=attention_mask,  # 添加 attention_mask
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                max_length=style_max_length,
                num_return_sequences=num_return_sequences,
                repetition_penalty=repetition_penalty,
                penalty_alpha=0.6,
                no_repeat_ngram_size=1,
                pad_token_id=tokenizer.eos_token_id,  # 设置 pad_token_id
                early_stopping=False
            )
        else:
            output = model.generate(
                input_ids,
                attention_mask=attention_mask,  # 添加 attention_mask
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                max_length=style_max_length,
                num_return_sequences=num_return_sequences,
                repetition_penalty=repetition_penalty,
                pad_token_id=tokenizer.eos_token_id,  # 设置 pad_token_id
                early_stopping=True
            )
        return tokenizer.decode(output[0], skip_special_tokens=True)
    else:
        gr.Warning("未找到模型。请查看命令控制台信息")
        return ""
