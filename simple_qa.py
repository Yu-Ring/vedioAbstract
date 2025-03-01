from sentence_transformers import SentenceTransformer
# from langchain_community.docstore.document import Document
from langchain_core.documents import Document
# from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
import os
import faiss
import numpy as np

def main(input_filename):
    # 1. 读取字幕文件，每一行作为一个独立的文本
    with open(input_filename, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    # 2. 将每一行转换为 Document 对象（每个 Document 将对应一个向量）
    documents = [Document(page_content=line) for line in lines]

    # 3. 使用 SentenceTransformer 创建嵌入模型
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # 4. 生成每个文档的嵌入向量
    embeddings = [model.encode(doc.page_content) for doc in documents]

    # 5. 构建 FAISS 索引
    # 创建一个用于 FAISS 索引的 numpy 数组
    embedding_matrix = np.array(embeddings).astype("float32")

    # 通过 FAISS 创建索引
    dimension = embedding_matrix.shape[1]  # 向量的维度
    index = faiss.IndexFlatL2(dimension)  # L2距离的索引
    index.add(embedding_matrix)  # 将嵌入向量添加到索引中

    # 6. 构造文档存储（docstore）
    docstore = {i: doc for i, doc in enumerate(documents)}  # 将每个文档存入一个字典

    # 7. 使用 FAISS 向量数据库初始化
    faiss_index = FAISS(index=index, 
                        docstore=docstore, 
                        index_to_docstore_id={i: i for i in range(len(documents))},
                        embedding_function=model.encode)  # Pass the embedding function

    # 8. 构造检索器（Retriever）
    retriever = faiss_index.as_retriever()

    # 9. 示例：检索与查询最相似的 3 个文档
    query = "人类"
    query_embedding = model.encode(query)  # Convert the query to an embedding
    D, I = index.search(np.array([query_embedding], dtype='float32'), 3)  # Perform the similarity search

    # Get the retrieved documents based on the indices
    retrieved_docs = [documents[i] for i in I[0]]


    # 打印检索到的相关文本
    print("检索到的相关文本：")
    for doc in retrieved_docs:
        print("-", doc.page_content)


# 6. 循环进行问答
if __name__ == "__main__":
    input_filename = 'video/B_HeiTianEr.srt'
    # 生成输出文件名：在原文件名的基础上加上 _clean.txt
    base, ext = os.path.splitext(input_filename)
    output_filename = f"{base}_clean.txt"

    main(output_filename)

