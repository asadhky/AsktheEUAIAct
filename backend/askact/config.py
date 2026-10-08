"""Application settings.

Every setting comes from an environment variable (or a local `.env`, which is gitignored).
This is the only module that reads the environment; everything else receives a `Settings`.
`.env.example` documents each variable, and a test keeps the two in sync.
"""

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

# The original Official Journal text of Regulation (EU) 2024/1689, not a consolidated version.
DEFAULT_SOURCE_URL = "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=OJ:L_202401689"


class ModelPrice(BaseModel):
    """USD per million tokens. Real prices are the operator's responsibility; the repo has none."""

    input_per_mtok: float = Field(ge=0)
    output_per_mtok: float = Field(ge=0)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        # A blank `NAME=` means "unset". That lets `.env.example` list every variable,
        # including ones that have no default, without breaking typed fields.
        env_ignore_empty=True,
        # `.env` also holds NEXT_PUBLIC_API_URL, which only the Next.js build reads.
        extra="ignore",
    )

    # --- Ingestion ---
    source_url: str = DEFAULT_SOURCE_URL
    # Used when the fetch fails. A different name from the cached download, so a fetch
    # can never overwrite a manually downloaded copy.
    source_fallback_path: Path = Path("data/raw/ai-act-oj-2024-1689.manual.html")
    # A starting value: a character count is only a proxy for the embedder's token limit.
    chunk_max_chars: int = Field(1800, gt=0)

    # --- Retrieval ---
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    embedder: Literal["real", "stub"] = "real"  # "stub" for tests and CI: no model download
    reranker: Literal["real", "stub"] = "real"
    reranker_enabled: bool = False
    top_k: int = Field(5, gt=0)

    # --- Generation ---
    # No honest default exists: they are tuned on the dev split. Until then the gate is unusable.
    relevance_threshold_cosine: float | None = None  # bm25, dense, hybrid
    relevance_threshold_rerank: float | None = None  # hybrid+rerank
    # Validated where it is used (the provider factory), so an unsupported value gets a
    # clear "LLM unavailable" error instead of failing every import of the settings.
    llm_provider: str | None = None
    llm_model: str | None = None
    llm_api_key: SecretStr | None = None  # SecretStr keeps the key out of repr() and logs
    llm_base_url: str | None = None
    # JSON, e.g. {"model-name": {"input_per_mtok": 0.0, "output_per_mtok": 0.0}}
    llm_price_table: dict[str, ModelPrice] = Field(default_factory=dict)
    max_output_tokens: int = Field(500, gt=0)

    # --- Public-demo safety limits ---
    max_question_chars: int = Field(500, gt=0)
    rate_limit_per_minute: int = Field(10, gt=0)
    trusted_proxy_header: str | None = None
    daily_spend_cap_usd: float = Field(1.00, ge=0)
    spend_db_path: Path = Path("/var/lib/askact/spend.sqlite3")
    # NoDecode: take the raw string, so the .env value can be a plain comma-separated list.
    cors_origins: Annotated[list[str], NoDecode] = ["http://localhost:3000"]

    # --- Evaluation and logging ---
    eval_max_llm_calls: int = Field(250, ge=0)
    log_questions: bool = False

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value
