## Setup

### 1. Install Dependencies

   **Python Version 3.12** 
   **Python Packages:**
   ```bash
   pip install chromadb langchain langchain_community langchain_core langchain_ollama PyMuPDF python-docx ollama openai-whisper
   ```
   **Ollama Version 0.9.0** 
   **Ollama Models:**
   For each model specified in your `models.json` (see Configuration section below), install it using:
   ```bash
   ollama pull <model-name>
   ```

### 2. Configuration
   Models are configured in `models.json`. This file is typically located at:
   `C:\Users\<your-username>\Documents\InquiroAI\models.json`

   The `models.json` file defines the following model categories:
   - `querying_models`: For text queries
   - `image_embedding_models`: For image processing
   - `text_embedding_models`: For text embeddings

   **Example:**
    ```json
    {
        "model_categories": {
        "querying_models": ["llama3.1:8b", "deepseek-r1:7b"],
        "image_embedding_models": ["llava-llama3:8b-v1.1-q4_0"],
        "text_embedding_models": ["all-minilm:latest"]
        }
    }
```
- **Tip:** The first model in each list is used as default.

