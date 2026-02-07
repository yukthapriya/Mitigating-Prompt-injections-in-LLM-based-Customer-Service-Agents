# Mitigating Prompt Injections in LLM-Based Customer Service Agents

A scalable, multi-agent defense system designed to protect enterprise LLM-powered customer service agents from prompt injection, data exfiltration, and privilege-escalation attacks.

## Overview
Large Language Models (LLMs) are increasingly used in customer service systems, but they remain vulnerable to prompt injection attacks that can override instructions or leak sensitive data.

This project presents a **defense-in-depth, multi-agent architecture** that combines:
- Deterministic routing
- Semantic security checks
- Distributed anomaly detection
- Output exfiltration guards

The system is designed for **real-time, enterprise-scale deployments** while maintaining low latency.

## Key Features
- Multi-agent security pipeline
- Semantic prompt injection detection using BERT
- Guardrails-based policy enforcement
- Local LLM inference (no data leakage)
- Distributed anomaly detection with Dask
- Streamlit dashboard for monitoring and human review
- Role-based access control for sensitive operations

## System Architecture
The system consists of the following agents:

### 1. PreFilter Agent
- Fast syntactic checks
- Unicode normalization
- Regex-based jailbreak detection

### 2. Router Agent
- Deterministic routing rules
- Role-based access control (RBAC)
- Early blocking of privileged commands

### 3. Security Agent
- Guardrails policy engine
- Fine-tuned DistilBERT classifier
- Semantic prompt injection detection

### 4. Query Agent
- Safe LLM interaction
- Prompt sanitization and templating
- Retrieval-augmented generation (RAG)

### 5. Output Guard
- Detects data exfiltration
- Redacts sensitive content
- Blocks unsafe responses

### 6. Distributed Monitoring Layer
- Dask-based anomaly detection
- Pattern tracking across large query volumes

## Tech Stack
- Python 3.11
- PyTorch (GPU support)
- LangChain
- Guardrails (NeMo Guardrails)
- DistilBERT (Transformers)
- Ollama (Llama-3.1-8B local model)
- Dask (distributed processing)
- Streamlit (admin dashboard)

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/llm-prompt-injection-defense.git
cd llm-prompt-injection-defense
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```

### 3. Install dependencies
```bash
pip install langchain guardrails-ai transformers streamlit ollama pandas numpy scikit-learn tqdm
pip install "dask[distributed]"
```

For GPU support:
```bash
pip install "torch torchvision torchaudio" --index-url https://download.pytorch.org/whl/cu121
```

## Running the System

### 1. Start the local LLM (Ollama)
```bash
ollama run llama3.1:8b
```

### 2. Launch the multi-agent backend
```bash
uvicorn app:app
```

### 3. Start the admin dashboard
```bash
streamlit run dashboard.py --server.port=8501
```

## Evaluation Results
- Tested on **50,000 augmented attack queries**
- **83.8% injection detection recall**
- **5–10% latency overhead**
- Designed for **10,000 queries per minute** environments

## Threat Model
The system defends against:
- Direct prompt injections
- Indirect/multi-turn injections
- Multilingual and encoded attacks
- Data exfiltration attempts
- Insider and privilege escalation attacks
- Sensitive information disclosure

## Project Structure
```
agents/
  ├── prefilter.py
  ├── router.py
  ├── security.py
  ├── query.py
  └── output_guard.py

app.py
dashboard.py
train_security_model.py
requirements.txt
```

## Research Questions
1. How can LLM security systems scale with low latency?
2. How can models generalize to unseen injection attacks?
3. How can enterprise AI systems remain transparent and trustworthy?

## Future Work
- Lower false-positive rates
- Adaptive online learning from flagged prompts
- Integration with enterprise identity systems
- Multimodal injection defenses

## Authors
- Yuktha Priya Masupalli  
- Sai Lahari Pathipati  
- Jagan Nookala  
Texas A&M University–San Antonio

## License
MIT License 
