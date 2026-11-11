from lex_secrets import Secret
from lex_settings import Settings


def test_secret_repr_hides_value() -> None:
    secret = Secret("super-secret")
    assert "super-secret" not in repr(secret)
    assert "super-secret" not in str(secret)
    assert secret.get() == "super-secret"


def test_settings_prefer_llm_names() -> None:
    settings = Settings.from_environ(
        {
            "DATA_DIR": "kb",
            "LLM_BASE_URL": "http://proxy",
            "OPENAI_BASE_URL": "http://other",
            "LLM_API_KEY": "k1",
            "OPENAI_API_KEY": "k2",
            "LLM_MODEL": "deepseek",
            "EMB_MODEL": "bge-m3",
            "RERANKER_MODEL": "qwen-rerank",
            "TEMPERATURE": "0.2",
            "MAX_TOKENS": "256",
            "MAX_WORKERS": "2",
            "TOP_K": "5",
            "RERANKER_TOP_N": "3",
        }
    )
    assert settings.data_dir.as_posix() == "kb"
    assert settings.llm_base_url == "http://proxy"
    assert settings.llm_api_key.get() == "k1"
    assert settings.llm_model == "deepseek"
    assert settings.embedding_model == "bge-m3"
    assert settings.reranker_model == "qwen-rerank"
    assert settings.max_workers == 2
    assert "k1" not in repr(settings)
