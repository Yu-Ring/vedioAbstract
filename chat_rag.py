# 导入必要的库和模块
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_community.docstore.document import Document
from langchain_community.vectorstores import FAISS
import faiss
from openai import OpenAI
import json

# 函数：处理通义千问的返回结果
def post_process_response(response):
    """处理通义千问的返回结果"""
    # 将返回的 JSON 格式的响应解析为 Python 字典
    response_data = json.loads(response)
    
    # 获取回答的内容部分
    content = response_data['choices'][0]['message']['content']
    
    # 清理内容，去掉 # 号和多余的换行符
    content = content.replace("#", "")  # 去掉 # 号
    content = content.replace("\n\n", "\n")  # 替换多余的换行
    content = content.replace("\n", "\n\n")  # 添加适当的换行
    
    return content


# API 的客户端
client = OpenAI(
    api_key="",
    base_url="https://api.chatanywhere.tech/v1",
)

# 函数：格式化大模型的返回结果
def query_tongyi_qianwen(query):
    """调用通义千问接口，获取答案"""
    # 调用通义千问的聊天接口，向其发送查询
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",  # 使用 qwen-plus 模型，按需选择模型
        messages=[  # 设置系统和用户消息
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': query}
        ]
    )
    
    # 返回通义千问的响应内容（JSON 格式）
    return completion.model_dump_json()

# 主函数：读取字幕文件、生成嵌入、构建索引、查询并获取答案
def main(input_filename, chunk_size=10):
    # 1. 读取字幕文件，每一行作为一个独立的文本
    # 打开字幕文件并按行读取，将每一行去除多余的空格并存入列表
    with open(input_filename, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    # 2. 将多行合并为一个块（chunk）
    # 将每个 chunk 的大小设置为 `chunk_size`，将文件内容分割成多个块
    chunks = []
    for i in range(0, len(lines), chunk_size):
        chunk = ' '.join(lines[i:i + chunk_size])  # 合并为一个块
        chunks.append(chunk)

    # 3. 将每个块转换为 Document 对象
    # 使用 langchain 的 Document 类将每个文本块封装为 Document 对象，便于后续索引和检索
    documents = [Document(page_content=chunk) for chunk in chunks]

    # 4. 使用 SentenceTransformer 创建嵌入模型
    # SentenceTransformer 是一个用于生成文本嵌入的模型，这里我们使用的是 `all-MiniLM-L6-v2`
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # 5. 生成每个文档的嵌入向量
    # 对每个文档生成其对应的嵌入向量，这些向量表示文档的语义信息
    embeddings = [model.encode(doc.page_content) for doc in documents]

    # 6. 构建 FAISS 索引
    # 使用 FAISS 库构建一个基于 L2 距离的索引，以便快速检索
    embedding_matrix = np.array(embeddings).astype("float32")
    dimension = embedding_matrix.shape[1]  # 获取嵌入向量的维度
    index = faiss.IndexFlatL2(dimension)  # 创建 L2 距离的平面索引
    index.add(embedding_matrix)  # 将所有文档的嵌入向量添加到索引中

    # 7. 构造文档存储（docstore）
    # 为了便于检索后返回具体的文档内容，创建一个字典存储每个文档
    docstore = {i: doc for i, doc in enumerate(documents)}

    # 8. 使用 FAISS 向量数据库初始化
    # 使用 FAISS 索引和文档存储构建一个 VectorStore 对象，用于检索
    faiss_index = FAISS(index=index, 
                        docstore=docstore, 
                        index_to_docstore_id={i: i for i in range(len(documents))},
                        embedding_function=model.encode)

    # 9. 构造检索器（Retriever）
    # 使用 FAISS 向量数据库创建检索器，便于后续查询
    retriever = faiss_index.as_retriever()

    # 10. 示例：检索与查询最相似的 10 个文档
    # 提供一个查询示例（例如，关于游戏客户端的基础模块），并计算该查询的嵌入
    query = "列出视频大纲"
    query_embedding = model.encode(query)  # 将查询转换为嵌入向量
    
    # 使用 FAISS 执行相似度搜索，查找与查询最相似的文档
    D, I = index.search(np.array([query_embedding], dtype='float32'), 10)  # 查找最相似的 10 个文档

    # 获取检索到的相关文档
    retrieved_docs = [documents[i] for i in I[0]]

    # 打印检索到的相关文本
    print("检索到的相关文本：")
    for doc in retrieved_docs:
        print("-", doc.page_content)

    # 基于检索到的相关文档，调用通义千问获取回答
    print("\n调用通义千问 API 获取答案：")
    response = query_tongyi_qianwen(query)  # 调用通义千问获取答案
    clean_response = post_process_response(response)  # 清理并格式化回答
    print("通义千问的回答:", clean_response)


# 程序入口，执行主要逻辑
if __name__ == "__main__":
    # 设置输入文件路径，并为输出文件设置文件名
    input_filename = 'video/B_HeiTianEr.srt'
    base, ext = os.path.splitext(input_filename)
    output_filename = f"{base}_clean.txt"
    
    # 调用主函数进行处理
    main(output_filename)
