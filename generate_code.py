import json
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import glob
import os
from PIL import Image
import numpy as np  # 必须导入numpy

# 配置模型路径和设备
MODEL_PATH = "/share/imagereward_work/xjj/database/model/cogvlm2-llama3-chat-19B"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TORCH_TYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

# 初始化tokenizer和model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=TORCH_TYPE, trust_remote_code=True).to(DEVICE).eval()

def generate_code(model, tokenizer, description):
    prompt = f"""
    Below is a description:
    {description}
    Write a python code that accomplishes the task described above. Only provide the code and ensure the code is within Python code block (```).
    """
    input_by_model = model.build_conversation_input_ids(
        tokenizer,
        query=prompt,
        history=[],  # 当前没有历史对话
        template_version='chat'
    )
    
    inputs = {
        'input_ids': input_by_model['input_ids'].unsqueeze(0).to(DEVICE),
        'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(DEVICE),
        'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(DEVICE),
    }
    
    gen_kwargs = {
        "max_new_tokens": 1024,
        "pad_token_id": 128002,  # Assuming this is the correct pad token ID
    }
    
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取 markdown 代码块中的代码
        code_start = generated_code.find("```")
        if code_start != -1:
            generated_code = generated_code[code_start + 3:]
            code_end = generated_code.find("```")
            if code_end != -1:
                generated_code = generated_code[:code_end]
        
        return generated_code.strip()

def process_parquet_file(input_file, results):
    # 读取 Parquet 文件
    df = pd.read_parquet(input_file)[:1]
    
    # 确保数据包含 'description' 和 'public_tests' 列
    if 'description' not in df.columns or 'public_tests' not in df.columns:
        raise ValueError("Parquet file must contain 'description' and 'public_tests' columns")

    # 生成代码
    for idx, (description, public_tests) in tqdm(enumerate(zip(df['description'], df['public_tests'])), desc=f"Processing {input_file}", total=len(df)):
        generated_code = generate_code(model, tokenizer, description)
        
        
        result = {
            'index': idx,
            'generated_code': generated_code,
            'public_tests_input': public_tests["input"].tolist(),
            'public_tests_output': public_tests["output"].tolist(),
        }
        results.append(result)

def process_all_parquet_files(input_dir, output_file):
    results = []

    # 查找所有以 'test' 开头的 Parquet 文件
    input_files = glob.glob(os.path.join(input_dir, 'test*.parquet'))

    for input_file in input_files[:1]:
        process_parquet_file(input_file, results)
        
    # 将结果保存到 JSONL 文件
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for result in results:
            json.dump(result, outfile)
            outfile.write('\n')

# 文件路径
input_dir = '/share/home/xujiajun/dataset/code_contests/data'
output_dir = '/share/home/xujiajun/RL'
output_file = os.path.join(output_dir, 'code.jsonl')

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 处理所有数据集
process_all_parquet_files(input_dir, output_file)