# InquiroAI: Your Private, Local AI Assistant for Documents & Images

**Discover, Summarize, and Interact with Your Data—All On Your Machine!**

---

## What is InquiroAI?

**InquiroAI** is a next-generation, privacy-first desktop application that lets you upload documents and images, ask questions (by text or voice), and get instant, insightful answers—powered by local Large Language Models (LLMs) and vision models. No cloud. No data leaks. Just blazing-fast, intelligent Q&A and summarization, right on your computer.

---

## Why InquiroAI?

- **Privacy by Design:** All processing happens locally. Your files and questions never leave your device.
- **Multi-Modal:** Works with PDFs, text, Word docs, and images (including OCR).
- **Voice & Text:** Ask questions by typing or speaking—get answers in text or audio.
- **Customizable AI:** Choose your favorite LLMs and vision models via a simple config file.
- **Lightning Fast:** No waiting for cloud APIs. Get answers in seconds.

---

## How Does It Work?

1. **Upload Files:** Drag and drop or select documents/images to process.
2. **Choose Models:** Select from available text, image, and query models (Ollama, LLaMA, DeepSeek, LLava, etc.).
3. **Process:** Click "Process Documents" to embed and index your data.
4. **Ask Anything:** Type or speak your question. InquiroAI finds the answer using local LLMs and vision models.
5. **Get Answers:** Instantly see (and hear) summaries, answers, and sources—no internet required!

---

## Prerequisites

- **Python 3.10+** (for backend processing)
- **Node.js & npm** (for Electron desktop app)
- **Ollama** (for running local LLMs and vision models)
- **Windows, macOS, or Linux** (cross-platform support)
- **Install Required Python Packages:**  
  ```
  pip install -r requirements.txt
  ```
- **Install Ollama Models:**  
  Use `ollama pull <model-name>` for each model you want to use (see your `models.json`).

---

## Configuration

All model choices are managed in a single file:

```
C:\Users\<your-username>\Documents\InquiroAI\models.json
```

**Example:**
```json
{
  "model_categories": {
    "querying_models": ["llama3.1:8b", "deepseek-r1:7b"],
    "image_embedding_models": ["llava:13b", "llava-llama3:8b-v1.1-q4_0"],
    "text_embedding_models": ["all-minilm:latest", "nomic-embed-text:latest"]
  }
}
```
- **Tip:** The first model in each list is used as default.

---

## Getting Started

1. **Clone the Repo:**  
   `git clone https://github.com/yourusername/InquiroAI.git`
2. **Install Dependencies:**  
   - Python: `pip install -r requirements.txt`
   - Node: `npm install`
3. **Configure Models:**  
   Edit your `models.json` in Documents/InquiroAI.
4. **Run Ollama:**  
   Start Ollama and pull your models.
5. **Launch InquiroAI:**  
   - Backend: `python python_backend/inquiroAI.py`
   - Desktop App: `npm start` or run the Electron app.

---

## Use Cases

- **Research:** Instantly search and summarize academic papers.
- **Business:** Extract insights from contracts, reports, and presentations.
- **Personal:** Organize and query your notes, receipts, and images.
- **Developers:** Build on top of InquiroAI for custom local AI workflows.

---

## Under the Hood

- **Electron** for the desktop UI
- **Python** backend for document/image processing
- **LangChain** for LLM orchestration
- **Ollama** for running local LLMs and vision models
- **ChromaDB** for fast vector search
- **Speech Recognition & TTS** for voice interaction

---

## Try It Today!

**InquiroAI** is the future of private, local AI.  
No subscriptions. No cloud. Just your data, your questions, your answers.

---

**Ready to take control of your knowledge?**  
[Download InquiroAI](https://github.com/yourusername/InquiroAI) and start exploring your data—like never before.

---

*#AI #LLM #Privacy #DesktopApp #Ollama #LangChain #Electron #InquiroAI #Productivity #OpenSource*