
import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Settings:
    """
    Central application configuration.
    """

    openai_api_key: str = field(repr=False)

    embedding_model: str = "text-embedding-3-small"
    generation_model: str = "gpt-5-mini"

    faiss_index_path: str = "enterprise_index.faiss"
    vectors_path: str = "enterprise_vectors.pkl"

    retrieval_top_k: int = 3
    similarity_threshold: float = 0.0

    allowed_origins: tuple[str, ...] = (
        "https://healthsportvoyageai.com",
        "https://www.healthsportvoyageai.com",
        "https://sergvision.github.io",
    )


def get_settings() -> Settings:
    """
    Load and validate application settings
    from environment variables.
    """

    api_key = os.getenv("OPENAI_API_KEY", "").strip()

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY environment variable is not configured."
        )

    top_k = int(os.getenv("RETRIEVAL_TOP_K", "3"))
    threshold = float(
        os.getenv("SIMILARITY_THRESHOLD", "0.0")
    )

    if top_k < 1:
        raise ValueError("RETRIEVAL_TOP_K must be at least 1.")

    if not -1.0 <= threshold <= 1.0:
        raise ValueError(
            "SIMILARITY_THRESHOLD must be between -1.0 and 1.0."
        )

    origins = os.getenv("ALLOWED_ORIGINS")

    allowed_origins = (
        tuple(
            origin.strip()
            for origin in origins.split(",")
            if origin.strip()
        )
        if origins
        else Settings.allowed_origins
    )

    return Settings(
        openai_api_key=api_key,
        embedding_model=os.getenv(
            "EMBEDDING_MODEL",
            "text-embedding-3-small"
        ),
        generation_model=os.getenv(
            "GENERATION_MODEL",
            "gpt-5-mini"
        ),
        faiss_index_path=os.getenv(
            "FAISS_INDEX_PATH",
            "enterprise_index.faiss"
        ),
        vectors_path=os.getenv(
            "VECTORS_PATH",
            "enterprise_vectors.pkl"
        ),
        retrieval_top_k=top_k,
        similarity_threshold=threshold,
        allowed_origins=allowed_origins,
    )
