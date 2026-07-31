
# macOS / Linux
curl -fsSL <https://ollama.com/install.sh> | sh

# Windows: OllamaSetup.exe from ollama.com/download

# test
curl http://localhost:11434


# Q8_0 8-Bit-Integer 
ollama run llama3.2:1b

# 4-Bit — halb so groß, etwas ungenauer, schneller auf schwacher Hardware
ollama run llama3.2:1b-instruct-q4_K_M

# volle Präzision (fp16) — größer, minimal bessere Qualität
ollama run llama3.2:1b-instruct-fp16


ollama run hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:IQ3_M
