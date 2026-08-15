# 💻 IT Infrastructure RAG Evaluation Assistant

A local **Retrieval-Augmented Generation (RAG)** system for answering IT infrastructure and server administration questions from a technical documentation knowledge base.

The project combines **local BGE embeddings**, **ChromaDB**, **BM25**, **Reciprocal Rank Fusion (RRF)**, and a **local Qwen2.5-0.5B-Instruct** model.

The system also includes **retrieval evaluation** and **runtime monitoring** to measure the performance of the RAG pipeline.

---

## 🎯 Project Goal

The goal of this project is to build and evaluate a practical RAG system specialized in IT infrastructure documentation.

Instead of relying only on an LLM's internal knowledge, the system retrieves relevant information from indexed technical documents and passes the retrieved context to a local language model.

The project focuses on two important RAG components:

1. **Retrieval quality**
2. **Answer generation**

---

# 🏗️ System Architecture

```text
                         User Question
                              │
                              ▼
                     ┌─────────────────┐
                     │   Streamlit UI  │
                     └────────┬────────┘
                              │
                              ▼
                     ┌─────────────────┐
                     │    Hybrid RAG   │
                     └────────┬────────┘
                              │
               ┌──────────────┴──────────────┐
               │                             │
               ▼                             ▼
       ┌───────────────┐             ┌───────────────┐
       │ Vector Search │             │  BM25 Search  │
       │     BGE       │             │   Keyword     │
       └───────┬───────┘             └───────┬───────┘
               │                             │
               └──────────────┬──────────────┘
                              │
                              ▼
                     ┌─────────────────┐
                     │      RRF        │
                     │ Rank Fusion     │
                     └────────┬────────┘
                              │
                              ▼
                     Retrieved Context
                              │
                              ▼
                     ┌─────────────────┐
                     │ Local Qwen LLM  │
                     │ Qwen2.5-0.5B    │
                     └────────┬────────┘
                              │
                              ▼
                       Final Answer
                              │
                       ┌──────┴──────┐
                       ▼             ▼
                    Sources       Metrics
```

---

# 🔎 RAG Pipeline

The system follows this pipeline:

```text
PDF Documents
      │
      ▼
Document Loading
      │
      ▼
Text Extraction
      │
      ▼
Chunking
      │
      ▼
BGE Embeddings
      │
      ▼
ChromaDB
      │
      ├───────────────┐
      ▼               ▼
 Vector Search      BM25
      │               │
      └───────┬───────┘
              ▼
             RRF
              │
              ▼
       Relevant Chunks
              │
              ▼
        Local Qwen
              │
              ▼
        Final Answer
```

---

# 🧠 Models

## Generation Model

The current generation model is:

```text
Qwen2.5-0.5B-Instruct
```

Local model path:

```text
D:\Models\Qwen2.5-0.5B-Instruct
```

The model is loaded locally using Hugging Face Transformers.

No external LLM API is required for the current generation pipeline.

---

## Embedding Model

The current embedding model is:

```text
BAAI/bge-small-en-v1.5
```

Local model path:

```text
D:\Models\bge-small-en-v1.5
```

The embedding model is used to create semantic representations of the documentation and user questions.

---

# 🗄️ Vector Database

The project uses:

```text
ChromaDB
```

Configuration:

```python
CHROMA_PATH = "chroma_db"

COLLECTION_NAME = "it_infrastructure"
```

The current ChromaDB installation used during testing contained:

```text
7,294 indexed chunks/documents
```

---

# 🔀 Hybrid Retrieval

The project combines two retrieval techniques.

## Vector Search

BGE embeddings are used to perform semantic similarity search.

This allows the system to retrieve content that is conceptually related to the question even when the exact wording differs.

---

## BM25

BM25 provides lexical keyword retrieval.

This is useful for technical terms such as:

```text
vSphere
vMotion
SYSVOL
VIOS
Active Directory
Windows Server
DHCP
```

---

## Reciprocal Rank Fusion

The vector and BM25 result lists are combined using **Reciprocal Rank Fusion (RRF)**.

The retrieval architecture is:

```text
Vector Search
      +
BM25 Search
      ↓
     RRF
      ↓
Combined Ranking
```

Current hybrid-search configuration:

```python
vector_k = 50
bm25_k = 50
final_k = 5
```

---

# 📚 Knowledge Base

The project uses technical PDF documentation as its knowledge base.

The indexed documentation covers areas including:

* VMware vSphere
* VMware vMotion
* VMware vSphere HA
* Microsoft Windows Server
* Windows administration
* Active Directory
* SYSVOL
* IBM PowerVM
* Virtual I/O Server (VIOS)

Example documents include:

```text
active_directory_operation_guide_part_1.pdf

IBM PowerVM.pdf

Microsoft Windows, Windows Server, Azure Stack Administrative Guide (22H2).pdf

Microsoft.pdf

vmware-vsphere-7-0.pdf
```

---

# 📊 Retrieval Evaluation

A dedicated evaluation script is included to compare:

```text
Vector Retrieval
        vs
Hybrid Retrieval
```

The evaluation uses:

* Hit@1
* Hit@3
* Hit@5
* Hit@10
* MRR

---

## Final Retrieval Results

The final evaluation produced:

| Metric | Vector |     Hybrid |  Difference |
| ------ | -----: | ---------: | ----------: |
| Hit@1  | 60.87% |     60.87% |      +0.00% |
| Hit@3  | 65.22% | **73.91%** |  **+8.70%** |
| Hit@5  | 69.57% | **73.91%** |  **+4.35%** |
| Hit@10 | 73.91% |     73.91% |      +0.00% |
| MRR    | 0.6467 | **0.6667** | **+0.0199** |

### Evaluation Result

```text
WINNER: Hybrid Retrieval
```

The largest improvement was observed at **Hit@3**:

```text
Vector : 65.22%

Hybrid : 73.91%

Gain   : +8.70%
```

MRR also improved:

```text
Vector : 0.6467

Hybrid : 0.6667

Gain   : +0.0199
```

These results indicate that the hybrid approach improved the ranking of relevant documents in the evaluation dataset.

---

# ⏱️ Runtime Monitoring

The RAG pipeline includes runtime monitoring.

The following metrics are measured:

* Retrieval time
* Generation time
* Total response time
* Candidate chunks
* Retrieved chunks
* Unique sources

Example:

```text
================================================================================
MONITORING
================================================================================
Retrieval time : 0.08s
Generation time: 6.82s
Total time     : 6.89s
Candidates     : 10
Chunks used    : 5
Unique sources : 5
================================================================================
```

The metrics are also returned from:

```python
HybridRAG.ask()
```

Example:

```python
{
    "retrieval_time": 0.08,
    "generation_time": 6.82,
    "total_time": 6.89,
    "candidate_chunks": 10,
    "retrieved_chunks": 5,
    "unique_sources": 5
}
```

---

# ⚡ Performance Observation

The monitoring results show a significant difference between retrieval and generation latency.

Typical observed values:

```text
Retrieval:
~0.08 - 0.20 seconds

Generation:
~6 - 8+ seconds
```

Therefore, in the current local configuration:

```text
Retrieval = Fast

Generation = Main latency bottleneck
```

The generation latency is primarily related to running the local Qwen model in the current environment.

---

# 🛡️ Grounded Generation

The Qwen generator uses a documentation-focused prompt.

The model is instructed to:

```text
1. Answer only from the provided documentation.
2. Do not use general knowledge.
3. Do not guess.
4. Do not invent definitions.
5. Do not add unsupported information.
6. Ignore irrelevant documents.
7. Keep the answer concise.
8. Return an "I don't know" response when the documentation
   does not support the answer.
```

The intended fallback response is:

```text
I don't know based on the indexed documentation.
```

---

# 🖥️ Streamlit Application

The project includes a Streamlit interface.

Run the application with:

```powershell
python -m streamlit run app.py
```

The interface displays:

* Application name
* Generation model
* Embedding model
* Number of indexed chunks
* Number of PDF documents
* Retrieval architecture
* Chat history
* Retrieved sources
* Source pages
* Runtime information

---

# 📁 Project Structure

```text
IT-Infrastructure-RAG-Evaluation-Assistant/
│
├── app.py
├── config.py
│
├── rag_hybrid.py
├── hybrid_search.py
├── vector_store.py
│
├── qwen_generator.py
├── gemini_generator.py
│
├── embeddings.py
├── document_loader.py
├── chunker.py
│
├── build_index.py
├── evaluate_retrieval.py
├── test_rag.py
├── test_batch.py
│
├── retrieval_evaluation.json
│
├── data/
│   └── *.pdf
│
├── chroma_db/
│
├── requirements.txt
│
└── README.md
```

---

# ⚙️ Configuration

The main configuration is located in:

```text
config.py
```

Current configuration:

```python
import os

from dotenv import load_dotenv

load_dotenv()


MODEL_NAME = r"D:\Models\Qwen2.5-0.5B-Instruct"

EMBEDDING_MODEL_PATH = r"D:\Models\bge-small-en-v1.5"

CHROMA_PATH = "chroma_db"

COLLECTION_NAME = "it_infrastructure"
```

---

# 🚀 Installation

## 1. Create Virtual Environment

```powershell
python -m venv .venv
```

## 2. Activate Environment

```powershell
.\.venv\Scripts\Activate.ps1
```

## 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

---

# 📥 Build the Knowledge Base

Place the PDF documents inside:

```text
data/
```

Then run:

```powershell
python build_index.py
```

The indexing process performs:

```text
PDF
 ↓
Text Extraction
 ↓
Chunking
 ↓
Embedding
 ↓
ChromaDB
```

---

# 🔍 Test Hybrid Search

To test retrieval independently:

```powershell
python -c "from hybrid_search import HybridSearch; h=HybridSearch(); print(h.search('What is vSphere HA?')[:5])"
```

---

# 🤖 Test Complete RAG

Run:

```powershell
python rag_hybrid.py
```

The test currently includes questions such as:

```text
What is vSphere HA?

What is VMware vMotion?

What is Windows Server management?

What is an authoritative restore of SYSVOL?

What is a Virtual I/O Server (VIOS)?
```

---

# 🧪 Evaluate Retrieval

Run the retrieval evaluation:

```powershell
python evaluate_retrieval.py
```

The evaluation compares vector retrieval with hybrid retrieval and saves the results to:

```text
retrieval_evaluation.json
```

---

# 🌐 Run the Web Application

Start Streamlit:

```powershell
python -m streamlit run app.py
```

The application can then be used to ask questions against the indexed IT documentation.

---

# 📝 Example Questions

## VMware

```text
What is vSphere HA?
```

```text
What is VMware vMotion?
```

## Microsoft Windows

```text
What is Windows Server management?
```

```text
What is Windows system administration?
```

## Active Directory

```text
What is an authoritative restore of SYSVOL?
```

## IBM PowerVM

```text
What is a Virtual I/O Server (VIOS)?
```

---

# 📖 Example RAG Flow

For a question such as:

```text
What is an authoritative restore of SYSVOL?
```

the system retrieves relevant Active Directory documentation.

The observed retrieval results included:

```text
active_directory_operation_guide_part_1.pdf
Page 27
Page 32
Page 34
```

For:

```text
What is a Virtual I/O Server (VIOS)?
```

the retrieval system identified:

```text
IBM PowerVM.pdf
```

with multiple relevant pages.

This demonstrates that the retrieval system can identify documentation specific to different infrastructure domains.

---

# ⚠️ Current Limitations

The project currently uses:

```text
Qwen2.5-0.5B-Instruct
```

The model is intentionally small enough to run locally on the current environment.

Because it is a small local language model, generation quality can still be limited.

Possible issues include:

* Incorrect interpretation of technical terminology
* Unsupported statements
* Incorrect abbreviation expansion
* Repetition
* Incomplete answers
* Combining information from unrelated context

Therefore, retrieval quality and generation quality are evaluated separately.

---

# 🔬 Retrieval vs Generation

A key design principle of this project is separating retrieval evaluation from generation evaluation.

## Retrieval

Retrieval asks:

> Did the system retrieve the correct documentation?

This is measured using:

```text
Hit@1
Hit@3
Hit@5
Hit@10
MRR
```

## Generation

Generation asks:

> Did the LLM produce an accurate answer from the retrieved evidence?

The current project has focused strongly on improving and measuring retrieval, while using a lightweight local model for generation.

---

# 🔐 Local Architecture

The current configuration is designed around local execution:

```text
                    Local Environment
                           │
             ┌─────────────┴─────────────┐
             │                           │
             ▼                           ▼
       Local BGE Model             Local Qwen Model
             │                           ▲
             ▼                           │
          ChromaDB                       │
             │                           │
             ▼                           │
        Vector Search                    │
             │                           │
             ├────── BM25 ──────┐        │
             │                  │        │
             └────── RRF ───────┘        │
                        │                │
                        └────────────────┘
```

No cloud LLM is required for the current generation pipeline.

---

# 🛠️ Technologies

| Technology             | Role                      |
| ---------------------- | ------------------------- |
| Python                 | Application development   |
| Streamlit              | User interface            |
| PyTorch                | Local model execution     |
| Transformers           | LLM loading and inference |
| Qwen2.5-0.5B-Instruct  | Text generation           |
| BGE-small-en-v1.5      | Embeddings                |
| ChromaDB               | Vector database           |
| BM25                   | Keyword retrieval         |
| RRF                    | Rank fusion               |
| PyPDF / PDF processing | Document ingestion        |

---

# 📌 Project Status

## Completed

* [x] PDF document ingestion
* [x] Text extraction
* [x] Document chunking
* [x] Local BGE embeddings
* [x] ChromaDB vector database
* [x] Vector search
* [x] BM25 search
* [x] Hybrid retrieval
* [x] Reciprocal Rank Fusion
* [x] Retrieval evaluation
* [x] Hit@K metrics
* [x] MRR metric
* [x] Local Qwen generation
* [x] Source attribution
* [x] Runtime monitoring
* [x] Streamlit interface
* [x] Local-only architecture
* [x] Vector vs Hybrid comparison

---

# 🔮 Future Improvements

Potential improvements include:

* Better relevance filtering before generation
* Better duplicate chunk removal
* Improved top-k selection
* Answer faithfulness evaluation
* Automated generation evaluation
* GPU acceleration
* Improved local inference performance
* Better monitoring dashboard
* Persistent evaluation history
* More comprehensive test questions
* Improved handling of unrelated retrieved documents

---

# 🎓 LLM Zoomcamp 2026

This project was developed as part of the **LLM Zoomcamp 2026** learning project.

The project applies RAG concepts to a practical IT infrastructure use case and focuses on:

```text
Document Retrieval
        +
Hybrid Search
        +
RAG
        +
LLM Generation
        +
Evaluation
        +
Monitoring
```

---

# 🏁 Conclusion

The **IT Infrastructure RAG Evaluation Assistant** demonstrates a complete local RAG pipeline for technical IT documentation.

The final architecture combines:

```text
Local BGE
    +
ChromaDB
    +
BM25
    +
RRF
    +
Local Qwen
    +
Streamlit
    +
Evaluation
    +
Monitoring
```

The retrieval evaluation demonstrates that the hybrid retrieval approach outperformed vector retrieval on the evaluated dataset:

```text
Hybrid Hit@3 : 73.91%
Hybrid MRR   : 0.6667
```

The project therefore provides both:

1. A working IT documentation assistant.
2. A measurable framework for evaluating and improving RAG retrieval performance.

---

## 👨‍💻 Project Name

**IT Infrastructure RAG Evaluation Assistant**

**LLM Zoomcamp 2026 Project**
