# classifAI_API
A stripped-down, importable and extendable implementation of classifAI's fastAPI

---

### Local LLM Usage instructions:

install ollama;

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

set the ollama server running;

```bash
ollama serve
```

open a second terminal and download your embedding model from HuggingFace;
```bash
ollama pull <model_name - e.g. nomic-embed-text>
```

Note that the ollama server must be running (`ollama serve`) when the classifai_api server is being 
used, in order for them to talk to each other.