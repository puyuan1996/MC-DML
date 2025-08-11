# src/llm.py

import json
from typing import List, Dict
from collections import deque
import numpy as np
import re

# 导入经过改造的辅助函数
from .openai_helpers import chat_completion_with_retries, truncate_text, init_qwen_model
from .utils import softmax


class LLMAgent:
    """
    LLM Agent for selecting actions in a text-based adventure game.
    Supports both OpenAI and local Qwen models.
    """
    def __init__(self, args):
        # 新增 llm_provider 参数来决定使用哪个模型
        self.provider = args.llm_provider
        self.model = args.llm_model
        self.max_memory = args.max_memory
        self.llm_temperature = args.llm_temperature
        self.softmax_temperature = args.softmax_temperature

        # 如果选择 qwen，则初始化本地模型
        if self.provider == 'qwen':
            # qwen_model_path 需要在 args 中提供
            if not hasattr(args, 'qwen_model_path') or not args.qwen_model_path:
                raise ValueError("Argument 'qwen_model_path' is required when llm_provider is 'qwen'.")
            init_qwen_model(args.qwen_model_path)
        elif self.provider != 'openai':
            raise ValueError(f"Unsupported llm_provider: {self.provider}. Choose 'openai' or 'qwen'.")


    def _format_state(self, state_node):
        return f"PREV_STATE: {state_node.prev_state}\nACTION: {state_node.prev_action}\nCURRENT_STATE: {state_node.state}"


    def get_probs_prompts(self, state_node, memory):
        formatted_state = self._format_state(state_node)
        actions_str = [f"{i}: {a}" for i, a in enumerate(state_node.valid_actions)]
        formatted_actions = "\n".join(actions_str)

        sys_prompt = """You are a player in a text-based adventure game. Your task is to evaluate and select actions that are promising based on the given game state."""

        if memory:
            user_prompt = f"""Your memory of playing this game previously is: {memory}
            You are now facing the following state in the game:{formatted_state}
            Considering the current state and previous memories, please select the action most worth exploring from the following list: {formatted_actions}
            Respond by providing the index of the action only. Your response should be a single integer, without any extra formatting, spaces, punctuation, or text."""
        else:
            user_prompt = f"""You are now facing the following state in the game: {formatted_state}
            Considering the current state, please select the most promising action from the following list: {formatted_actions}
            Respond by providing the index of the action only. Your response should be a single integer, without any extra formatting, spaces, punctuation, or text."""       
        return sys_prompt, user_prompt
    

    def get_reflection_prompts(self, trajectory):
        text_trajectory = "\n".join(
            f"STEP {i}, STATE: {step['state']}, ACTION: {step['action']}"
            for i, step in enumerate(trajectory)
        )
        text_trajectory = truncate_text(text_trajectory, 5000)

        sys_prompt = """You will receive a log of unsuccessful gameplay from a text-based adventure game. Please identify the reasons for this game failure and provide a short suggestion for improving the game strategy next time. Do not summarize the gameplay trajectory; respond with your suggestion in a single sentence. For instance:  'Remember to light a lamp before entering dark areas to avoid being eaten by a grue. '"""
        user_prompt = f"""GAMEPLAY TRAJECTORY: \n{text_trajectory}"""
        return sys_prompt, user_prompt
    
    
    def get_action_probs(self, state_node, memory):
        sys_prompt, user_prompt = self.get_probs_prompts(state_node, memory)
        valid_labels = [str(i) for i in range(len(state_node.valid_actions))]

        # --- 修改点 1: 解决 Qwen temperature 为 0 的问题 ---
        # 如果是 qwen 且 temperature 为 0, 设置为一个极小的正数
        temp = self.llm_temperature
        if self.provider == 'qwen' and temp == 0:
            temp = 0.01

        # 调用通用的 chat_completion 函数，并传入 provider
        res = chat_completion_with_retries(
            model=self.model,
            sys_prompt=sys_prompt,
            prompt=user_prompt,
            llm_provider=self.provider, # 关键参数
            max_tokens=8, # 增加 token 长度以适应 Qwen
            temperature=temp, # 使用修正后的 temperature
            # OpenAI 特有参数，Qwen 会忽略它们
            logprobs=True if self.provider == 'openai' else None,
            top_logprobs=min(len(state_node.valid_actions), 20) if self.provider == 'openai' else None
        )
        
        if not res or not res.choices:
            print("WARNING: LLM call failed or returned empty response. Falling back to uniform probabilities.")
            text = np.random.choice(valid_labels)
            probs_list = [1.0 / len(valid_labels)] * len(valid_labels)
            return text, probs_list

        if self.provider == 'openai':
            text = res.choices[0].message.content.strip()

            # --- OpenAI: 使用 logprobs 计算概率分布 ---
            top_logprobs = res.choices[0].logprobs.content[0].top_logprobs
            
            action_log_dict = {}
            for logprob in top_logprobs:
                action_token = logprob.token.strip()
                action_logprob = logprob.logprob
                if action_token in valid_labels:
                    action_log_dict[action_token] = action_logprob

            logprobs_list = [action_log_dict.get(label, -10) for label in valid_labels] # 使用更低的默认值
            probs_list = softmax(logprobs_list, self.softmax_temperature)

        elif self.provider == 'qwen':
            text = res.choices[0].message["content"].strip()

            # --- Qwen: 根据模型输出的单个选择构造概率分布 ---
            # 清理模型输出，只保留数字
            numeric_part = re.search(r'\d+', text)
            if numeric_part:
                text = numeric_part.group(0)
            
            probs_list = [0.0] * len(valid_labels)
            if text in valid_labels:
                chosen_index = int(text)
                probs_list[chosen_index] = 1.0  # One-hot 概率分布
            else:
                # 如果模型输出无效，则使用均匀分布作为后备
                print(f"WARNING: Qwen output '{text}' is not a valid action index. Falling back to uniform distribution.")
                probs_list = [1.0 / len(valid_labels)] * len(valid_labels)
                # 随机选择一个有效动作作为文本输出
                text = np.random.choice(valid_labels)
        
        return text, probs_list


    def get_traj_reflection(self, trajectory: List[Dict]) -> str:
        # 此处变量名是 prompt, 而不是 user_prompt
        sys_prompt, prompt = self.get_reflection_prompts(trajectory)
        
        # --- 修改点 2: 同样解决 Qwen temperature 为 0 的问题 ---
        temp = self.llm_temperature
        if self.provider == 'qwen' and temp == 0:
            temp = 0.01

        # 调用时传入 provider
        res = chat_completion_with_retries(
            model=self.model,
            sys_prompt=sys_prompt,
            # --- 修改点 3: 修正 NameError ---
            # 将 user_prompt 改为 prompt
            prompt=prompt,
            llm_provider=self.provider, # 关键参数
            max_tokens=128, # 为反思提供更长的生成空间
            temperature=temp # 使用修正后的 temperature
        )

        if not res or not res.choices:
            print("WARNING: Reflection generation failed.")
            return "Failed to generate reflection."

        # 同样适配 qwen 的输出格式
        if self.provider == 'qwen':
            text = res.choices[0].message["content"]
        else: # openai
            text = res.choices[0].message.content
            
        print(f"Generated Reflection: {text}")
        return text