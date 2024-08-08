import os
from transformers import T5Tokenizer, T5ForConditionalGeneration
import gradio as gr
from modules import scripts

extension_path = scripts.basedir()
model_folder = ""
tokenizer = ""
model = ""
num_return_sequences =1
def modelcheck():
    global model_folder
    global tokenizer
    global model
    global num_return_sequences
    if os.path.exists(os.path.join(extension_path, "superprompt-v1")):
        model_folder = os.path.join(extension_path, "superprompt-v1")
        tokenizer = T5Tokenizer.from_pretrained("roborovski/superprompt-v1")
        model = T5ForConditionalGeneration.from_pretrained("roborovski/superprompt-v1", device_map="auto")
        return True
    else:
        print('\033[92m' + f"""No Model Found, Please run 'git clone https://huggingface.co/roborovski/superprompt-v1' to the {extension_path}""" + '\033[0m\n')
        return False

def generate_super_prompt(prompt,max_new_tokens=77):
    model_found = modelcheck()
    if model_found:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
        outputs = model.generate(input_ids, max_new_tokens=max_new_tokens)
        generated_prompt = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_prompt
    else:
        gr.Warning("No Model Found. Check Command Console")
        return ""
