import os
from datetime import datetime
from typing import Literal, List
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langgraph.graph import END, StateGraph, START
from langchain_chroma import Chroma
from pydantic import BaseModel, Field

# 載入環境變數
load_dotenv()

# 初始化工具與參數
web_search_tool = TavilySearchResults(k=3)
current_date = datetime.now().strftime("%Y-%m-%d")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

############################   1. Indexing (本地化檢索)   ##############################

CHROMA_PATH = "./chroma_db"

def get_retriever():
    embeddings = OpenAIEmbeddings()
    if os.path.exists(CHROMA_PATH):
        vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    else:
        loader = PyPDFLoader("./airway evaluation.pdf")
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=250, chunk_overlap=50)
        doc_splits = text_splitter.split_documents(docs)
        vectorstore = Chroma.from_documents(documents=doc_splits, embedding=embeddings, persist_directory=CHROMA_PATH)
    return vectorstore.as_retriever()

retriever = get_retriever()

############################   2. Components (LLM 鏈定義)   ##############################

# --- Router: 判斷問題類型 ---
class RouteQuery(BaseModel):
    datasource: Literal["vectorstore", "web_search"] = Field(
        description="醫療/插管/臨床指引相關問題選 'vectorstore'；一般常識/天氣/活動選 'web_search'"
    )

router_prompt = ChatPromptTemplate.from_messages([
    ("system", f"你是一位專業路由專家。今天是 {current_date}。根據問題性質，決定導向醫療知識庫(vectorstore)或網路搜尋(web_search)。"),
    ("human", "{question}")
])
router_chain = router_prompt | llm.with_structured_output(RouteQuery)

# --- Grader: 檢查文件相關性 ---
class GradeDocuments(BaseModel):
    binary_score: Literal["yes", "no"] = Field(description="文件是否與問題相關")

grade_prompt = ChatPromptTemplate.from_messages([
    ("system", "判斷提供的文件內容是否與使用者問題相關。只需回傳 'yes' 或 'no'。"),
    ("human", "文件: \n\n {document} \n\n 問題: {question}")
])
retrieval_grader = grade_prompt | llm.with_structured_output(GradeDocuments)

# --- Generator: 生成最終答案 ---
generate_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一位醫療助理。請僅根據提供的上下文回答。用繁體中文回覆。"),
    ("human", "上下文: \n\n {context} \n\n 問題: {question}")
])
rag_chain = generate_prompt | llm | StrOutputParser()

# --- Rewriter: 優化搜尋關鍵字 ---
rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", f"你是搜尋專家。今天是 {current_date}。請將問題改寫成更適合網路搜尋的繁體中文關鍵字。"),
    ("human", "原始問題: {question}")
])
question_rewriter = rewrite_prompt | llm | StrOutputParser()

# --- Hallucination & Answer Graders ---
class GradeHallucinations(BaseModel):
    binary_score: Literal["yes", "no"] = Field(description="答案是否基於事實")

hallucination_grader = ChatPromptTemplate.from_messages([
    ("system", "檢查 AI 的回答是否完全基於提供的文件內容。"),
    ("human", "文件: \n\n {documents} \n\n AI 回答: {generation}")
]) | llm.with_structured_output(GradeHallucinations)

class GradeAnswer(BaseModel):
    binary_score: Literal["yes", "no"] = Field(description="回答是否解決了問題")

answer_grader = ChatPromptTemplate.from_messages([
    ("system", "評估 AI 的回答是否真的解決了問題。"),
    ("human", "問題: {question} \n\n AI 回答: {generation}")
]) | llm.with_structured_output(GradeAnswer)

############################   3. Nodes & Logic (圖節點邏輯)   ##############################

class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[Document]
    datasource: str
    loop_step: int

def route_question(state):
    print("--- [Router] 判斷路徑 ---")
    res = router_chain.invoke({"question": state["question"]})
    return {
        "datasource": res.datasource,
        "loop_step": 0  # 在這裡重置，確保後續節點拿到的是 0
    }

def retrieve_document(state):
    print("--- [Retrieve] 檢索本地文件 ---")
    docs = retriever.invoke(state["question"])
    return {"documents": docs, "loop_step": state.get("loop_step", 0)}

def grade_documents(state):
    print("--- [Grade] 篩選相關文件 ---")
    valid_docs = [d for d in state["documents"] if retrieval_grader.invoke({"question": state["question"], "document": d.page_content}).binary_score == "yes"]
    return {"documents": valid_docs}

def generate(state):
    print("--- [Generate] 生成答案 ---")
    context = "\n\n".join([d.page_content for d in state["documents"]])
    gen = rag_chain.invoke({"context": context, "question": state["question"]})
    return {"generation": gen, "loop_step": state.get("loop_step", 0) + 1}

def transform_query(state):
    print("--- [Transform] 優化搜尋詞 ---")
    better_q = question_rewriter.invoke({"question": state["question"]})
    return {"question": better_q}

def web_search(state):
    print(f"--- [Web Search] 執行網路搜尋，關鍵字：{state['question']} ---")
    
    # 執行搜尋
    search_res = web_search_tool.invoke({"query": state["question"]})
    
    # 初始化文件清單
    documents = []
    
    # 檢查搜尋結果是否為清單
    if isinstance(search_res, list):
        for d in search_res:
            # 建立獨立的 Document 物件
            doc = Document(
                page_content=d.get("content", ""), 
                metadata={"source": d.get("url", "unknown")}
            )
            documents.append(doc)
            # 除錯列印：看看抓到了什麼標題或內容片段
            print(f"  > 找到來源: {d.get('url')[:50]}...")
    else:
        # 處理搜尋失敗或回傳字串的情況
        documents = [Document(page_content=str(search_res))]

    # 如果完全沒結果，也給一個保險
    if not documents:
        print("!!! [警告] 搜尋結果為空 !!!")
        documents = [Document(page_content="搜尋不到相關資訊。")]

    # 回傳時，記得維持 question 和 loop_step 的狀態
    return {
        "documents": documents, 
        "question": state["question"],
        "loop_step": state.get("loop_step", 0) 
    }

def handle_failure(state):
    return {"generation": "抱歉，目前找不到相關資訊。"}

# 決策函數
def decide_to_generate(state):
    if not state["documents"]:
        return "transform_query" if state.get("loop_step", 0) < 2 else "stop"
    return "generate"
def grade_generation_v_documents_and_question(state):
    print("--- [Check] 檢查幻覺與切題度 ---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    loop_step = state.get("loop_step", 0)  # 取得目前的次數

    doc_txt = "\n\n".join([d.page_content for d in documents])
    
    # 1. 檢查幻覺
    h_score = hallucination_grader.invoke({"documents": doc_txt, "generation": generation})
    if h_score.binary_score == "no":
        # 如果有幻覺，且還沒超過次數，回 generate 重新生一次
        return "not grounded" if loop_step < 3 else "max_retries"
    
    # 2. 檢查是否解決問題
    r_score = answer_grader.invoke({"question": question, "generation": generation})
    if r_score.binary_score == "yes":
        return "useful"
    else:
        # 如果答案沒用，且還沒超過次數，才回 transform_query
        print(f"--- [警告] 答案品質不佳，目前嘗試第 {loop_step} 次 ---")
        return "not useful" if loop_step < 3 else "max_retries"
    
############################   4. Graph Construction (建立工作流)   ##############################

workflow = StateGraph(GraphState)

# 新增節點
workflow.add_node("router", route_question)
workflow.add_node("retrieve_document", retrieve_document)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("transform_query", transform_query)
workflow.add_node("web_search", web_search)
workflow.add_node("handle_failure", handle_failure)

# 設定連線
workflow.set_entry_point("router")

workflow.add_conditional_edges(
    "router",
    lambda x: x["datasource"],
    {"vectorstore": "retrieve_document", "web_search": "transform_query"}
)

workflow.add_edge("retrieve_document", "grade_documents")

workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {"transform_query": "transform_query", "generate": "generate", "stop": "handle_failure"}
)

workflow.add_edge("transform_query", "web_search")
workflow.add_edge("web_search", "generate")
workflow.add_edge("handle_failure", END)

workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {"not grounded": "generate", "useful": END, "not useful": "transform_query","max_retries": "handle_failure"}
)

app = workflow.compile()

############################   5. Run (測試)   ##############################

if __name__ == "__main__":
    inputs = {"question": "今天高雄有什麼活動？", "loop_step": 0}
    for output in app.stream(inputs, config={"recursion_limit": 15}):
        for key, value in output.items():
            print(f"\n[Node]: {key}")
            if "generation" in value:
                print(f"結果：{value['generation']}")