"""Model providers. Built-in presets are OpenAI-compatible endpoints; `mock` is offline."""

PRESETS: dict[str, dict] = {
    "ollama": {"base_url": "http://localhost:11434/v1"},
    "vllm": {"base_url": "http://localhost:8000/v1"},
    "lmstudio": {"base_url": "http://localhost:1234/v1"},
    "openrouter": {"base_url": "https://openrouter.ai/api/v1", "api_key_env": "OPENROUTER_API_KEY"},
    "together": {"base_url": "https://api.together.xyz/v1", "api_key_env": "TOGETHER_API_KEY"},
    "groq": {"base_url": "https://api.groq.com/openai/v1", "api_key_env": "GROQ_API_KEY"},
}
