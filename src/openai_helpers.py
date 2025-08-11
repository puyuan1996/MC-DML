# src/openai_helpers.py

import time
from typing import Dict, List, Mapping, Optional
import openai
from openai import OpenAI
import tiktoken
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# --- 全局变量 ---

# OpenAI 客户端
# client = OpenAI() 
# Tiktoken 编码器
encoding = tiktoken.get_encoding("cl100k_base")

# Qwen 模型和分词器将在 init_qwen_model 中被加载
qwen_model = None
qwen_tokenizer = None
qwen_device = None


def init_qwen_model(model_path: str):
    """
    初始化并加载本地 Qwen 模型到全局变量中。
    此函数应在程序开始时调用一次。
    """
    global qwen_model, qwen_tokenizer, qwen_device
    
    if qwen_model is not None:
        print("Qwen model is already initialized.")
        return

    try:
        print(f"Initializing Qwen model from path: {model_path}...")
        qwen_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        qwen_tokenizer = AutoTokenizer.from_pretrained(model_path)
        qwen_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto").to(qwen_device)
        print(f"Qwen model loaded successfully on device: {qwen_device}")
    except Exception as e:
        print(f"Error initializing Qwen model: {e}")
        raise

def chat_completion_with_retries(
    model: str, 
    sys_prompt: str, 
    prompt: str, 
    llm_provider: str,
    max_retries: int = 5, 
    retry_interval_sec: int = 20, 
    **kwargs
) -> Mapping:
    """
    根据 llm_provider 的值，调用相应的模型（OpenAI或Qwen）并处理重试逻辑。

    Args:
        model (str): 模型名称。
        sys_prompt (str): 系统提示。
        prompt (str): 用户提示。
        llm_provider (str): LLM提供者，'openai' 或 'qwen'。
        max_retries (int): OpenAI API 失败时的最大重试次数。
        retry_interval_sec (int): 重试间隔。
        **kwargs: 传递给模型API的其他参数 (如 temperature, max_tokens)。

    Returns:
        一个与 OpenAI API 响应结构类似的字典。
    """
    if llm_provider == 'openai':
        # --- OpenAI API 调用逻辑 ---
        for n_attempts_remaining in range(max_retries, 0, -1):
            try:
                res = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    **kwargs
                )
                return res
            except (openai.RateLimitError, openai.APIError, openai.OpenAIError) as e:
                print(e)
                print(f"Hit openai.error exception. Waiting {retry_interval_sec} seconds for retry... ({n_attempts_remaining - 1} attempts remaining)", flush=True)
                time.sleep(retry_interval_sec)
        return {} # 重试失败后返回空字典

    elif llm_provider == 'qwen':
        # --- 本地 Qwen 模型调用逻辑 ---
        if qwen_model is None or qwen_tokenizer is None:
            raise RuntimeError("Qwen model is not initialized. Please call init_qwen_model() first.")
        
        try:
            # 从 kwargs 中提取生成参数
            gen_config = GenerationConfig(
                temperature=kwargs.get('temperature', 0.7),
                max_new_tokens=kwargs.get('max_tokens', 64),
                do_sample=True,
                pad_token_id=qwen_tokenizer.eos_token_id
            )
            
            messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}]
            text_prompt = qwen_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = qwen_tokenizer([text_prompt], return_tensors="pt").to(qwen_device)

            output = qwen_model.generate(**model_inputs, generation_config=gen_config)
            output_ids = output[0][len(model_inputs.input_ids[0]):].tolist()
            content = qwen_tokenizer.decode(output_ids, skip_special_tokens=True).strip()

            # 构造一个与OpenAI响应兼容的模拟对象
            # 注意：我们不生成 logprobs，所以这个字段会是空的
            class MockChoice:
                def __init__(self, content):
                    self.message = {'content': content}
                    self.logprobs = None # Qwen 本地调用不直接提供 logprobs

            class MockResponse:
                def __init__(self, content):
                    self.choices = [MockChoice(content)]
            
            return MockResponse(content)

        except Exception as e:
            print(f"Error during Qwen model generation: {e}")
            return {} # 发生错误时返回空字典
    else:
        raise ValueError(f"Unknown llm_provider: '{llm_provider}'. Must be 'openai' or 'qwen'.")


def truncate_text(text, max_tokens):
    """
    使用 tiktoken 将文本截断到指定的最大 token 数量。
    """
    tokens = encoding.encode(text)
    if len(tokens) > max_tokens:
        print(f"WARNING: Maximum token length exceeded ({len(tokens)} > {max_tokens})")
        tokens = tokens[:max_tokens]
        text = encoding.decode(tokens)
    return text