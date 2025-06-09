# CartaMind
CartaMind - Document Processing and Question Answering System

## Configuration

### Models Configuration

CartaMind uses a `models.json` file to configure which AI models are available for text processing, image processing, and querying. This file is located in:

- Windows: `%USERPROFILE%\Documents\CartaMind\models.json`
- macOS: `~/Documents/CartaMind/models.json`
- Linux: `~/Documents/CartaMind/models.json`

Example `models.json` structure:
```json
{
    "model_categories": {
        "querying_models": [            
            "llama2:13b",
            "llama3.1:8b",
            "deepseek-r1:7b"
        ],
        "image_embedding_models": [
            "llava:13b",
            "llava-llama3:8b-v1.1-q4_0"
        ],
        "text_embedding_models": [
            "all-minilm:latest",
            "nomic-embed-text:latest"
        ]
    }
}
```

You can customize this file to:
1. Add or remove models from any category
2. Change the order (first model in each category is used as default)
3. Add new compatible models as they become available
