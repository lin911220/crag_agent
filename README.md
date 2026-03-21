# CRAG Agent: 醫療指引與即時資訊智能助理 

本專案實作了一個具備自我修正能力的 **CRAG (Corrective Retrieval-Augmented Generation)** 代理人。系統能自動判斷問題類型，並在本地知識庫資料不足或不相關時，自動觸發網路搜尋以確保回答的準確率。

## 📖 系統核心功能

* **智能路由 (Router)**：自動過濾問題。涉及醫療、插管、臨床指引的問題會導向本地向量資料庫；一般常識或即時活動則導向網路搜尋。
* **本地專業知識**：內建「成人呼吸道評估與處理」指引。
    * **正常呼吸評估**：成人正常呼吸次數為每分鐘 **12-20 次** 。
    * **構造定義**：上呼吸道包含鼻及喉，下呼吸道包含氣管及肺 。
    * **處置建議**：建議每天飲水 **1500-2000 毫升** 以稀釋痰液（心腎疾病者需依醫囑調整）。
* **自我修正機制 (Corrective Logic)**：
    * **評分員 (Grader)**：自動篩選檢索結果，剔除不相關資訊。
    * **自動改寫 (Rewriter)**：若初始資料不足，系統會將問題優化為更適合搜尋引擎的關鍵字。
* **品質防護網**：包含幻覺檢查與效用評估，並透過 `loop_step` 限制嘗試次數（最高 3 次），防止 API 無限循環消耗。

---

## 開發環境與伺服器啟動步驟

本專案使用 `langgraph dev` 啟動本地 API 伺服器，作為 Agent 的運作引擎。

## 檔案結構
* `main.py`：定義 LangGraph 節點與 CRAG 核心邏輯。
* `langgraph.json`：配置 Agent 名稱為 `crag_agent`。
* `airway evaluation.pdf`：本地醫療指引資料來源。
* `requirements.txt`：專案依賴清單。

### 1. 安裝依賴套件
建議使用 Python 3.10 以上版本，並安裝 `langgraph-cli`：
```bash
pip install -r requirements.txt
pip install -U langgraph-cli
```

### 2. 設定環境變數 (`.env`)
在專案根目錄建立 `.env` 檔案，內容如下：
```env
# OpenAI API Key: 用於模型推理與向量嵌入
OPENAI_API_KEY=sk-your-openai-key-here

# Tavily API Key: 用於執行網路搜尋
TAVILY_API_KEY=tvly-your-tavily-key-here

# LangChain Tracing: 建議開啟以便於 LangSmith 觀察流程
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_your-key-here
LANGCHAIN_PROJECT=crag_agent
```

### 3. 啟動本地伺服器
執行以下指令啟動本地 Web 服務：
```bash
langgraph dev
```
啟動成功後，終端機會顯示：
* **Local API**: `http://localhost:8123` (你的 Agent 正在此本地網址運行)。
* **Studio 連結**: 點擊該連結可開啟視覺化介面。

### 4. 使用 LangGraph Studio 偵錯
* **觀察路徑**：輸入問題後，可看見 `router` 如何分流資料。
* **檢查狀態**：在右側側邊欄隨時查看 `loop_step` 的累加情況與 `documents` 的具體內容。
* **熱重載**：修改 `main.py` 存檔後，本地伺服器會自動更新邏輯，無需重啟。

---

## 2026-03-21 高雄實測範例

系統已成功串接 Tavily 搜尋，可處理即時資訊。今日（2026 年 3 月 21 日）高雄活動包含：
* **2026 大港開唱 (Megaport Festival)**：20 週年紀念活動，今日在駁二藝術特區正式展開。
* **內門宋江陣**：今日在內門紫竹寺舉辦開幕式，包含文武陣頭大賽。
* **戴佩妮演唱會**：晚上 19:30 於高雄巨蛋演出。

