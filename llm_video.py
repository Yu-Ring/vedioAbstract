import openai
import os
from openai import OpenAI


# Step 1: 读取txt文件中的背景知识
def load_background_knowledge(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Step 2: 调用API进行问答交互
def ask_question(background_knowledge, question):

    client = OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        # api_key="",
        # base_url="https://api.chatanywhere.tech/v1",
    )



    # 构建prompt，将背景知识和问题结合
    prompt = f"以下是背景知识：\n{background_knowledge}\n\n根据这些背景知识，请回答以下问题：\n{question}"

    # # 调用OpenAI的GPT-3进行问答
    # response = openai.Completion.create(
    #     engine="text-davinci-003",  # 可以替换为你需要的模型
    #     prompt=prompt,
    #     max_tokens=150,
    #     temperature=0.7
    # )
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo", # 此处以qwen-plus为例，可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': prompt}],
        )
        
    print(completion.model_dump_json())
    

    # 返回模型的回答
    return completion.model_dump_json()

if __name__ == "__main__":
    # 1. 读取背景知识
    background_knowledge = load_background_knowledge('background.txt')

    # 2. 提出问题并获取答案
    question = "根据上述背景知识，分析视频的大纲"

    answer = ask_question(background_knowledge, question)
    print(f"问题：{question}")
    print(f"回答：{answer}")
