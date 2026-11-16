# tests/test_factory.py
from core.llm_factory import get_chat_model

def test_models():
    print("Testing Gemini via LangChain...")
    gemini = get_chat_model(provider="gemini")
    res1 = gemini.invoke("Respond with one word: 'Ready'.")
    print("Gemini response:", res1.content)

    print("\nTesting OpenRouter via LangChain...")
    openrouter = get_chat_model(provider="openrouter")
    res2 = openrouter.invoke("Respond with one word: 'Ready'.")
    print("OpenRouter response:", res2.content)

if __name__ == "__main__":
    test_models()