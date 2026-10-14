"""Application settings.

All settings are read from environment variables (or a .env file).
Nested settings use `__` as the delimiter, e.g. `DB__HOST=localhost`.
"""

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class DBSettings(BaseModel):
    host: str = "localhost"
    port: int = 5432
    user: str = "rag"
    password: str = "rag"
    name: str = "rag"

    @property
    def dsn(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


class LLMSettings(BaseModel):
    """Any OpenAI-compatible endpoint: OpenAI, Azure OpenAI (via proxy),
    Groq, Ollama (http://localhost:11434/v1), vLLM, LiteLLM gateway.

    Defaults to Groq's free tier, since that's this project's default setup
    (see README's "LLM" section) -- override via LLM__* env vars for any
    other OpenAI-compatible provider.
    """

    base_url: str = "https://api.groq.com/openai/v1"
    api_key: str = "unset"
    # "api_key" sends api_key as the bearer token. "azure_managed_identity"
    # ignores api_key and sends a short-lived Microsoft Entra ID token fetched
    # from the platform (see auth.py), for Azure OpenAI with key access disabled.
    auth: str = "api_key"
    azure_token_resource: str = "https://cognitiveservices.azure.com"
    # Was llama-3.3-70b-versatile until Groq decommissioned it (the whole Llama
    # chat lineup went; /v1/models now answers 404 model_not_found for it). A
    # hosted model can't be pinned the way a package can, so this default WILL
    # rot again -- when it does, `GET {base_url}/models` says what's still
    # offered. Chosen over openai/gpt-oss-*, which emit reasoning tokens and so
    # cost ~3x per call; the Ragas judge runs on this same model, and a full
    # judged run already strains the 100K-token daily cap (see README).
    # NOTE: eval/baseline.json was measured on the old model. Scores are not
    # comparable across this swap -- the baseline must be re-promoted.
    model: str = "qwen/qwen3.8-27b"
    temperature: float = 0.0
    timeout_s: float = 60.0
    # Declared output ceiling. Not optional in practice: providers reserve
    # per-minute output budget against a request's *expected* output, and with
    # no max_tokens they reserve the model's default (2048 on Groq). That alone
    # exceeds Groq's free-tier 1000 output-tokens-per-minute cap, so every
    # /query is rejected 429 "Request too large" no matter how idle the account
    # is. Answers here run ~30 tokens, so 512 is generous and still fits.
    max_tokens: int = 512
    min_interval_s: float = 0.0  # throttle between requests, e.g. to stay under a TPM cap
    # Sent only when set. reasoning_effort is for reasoning models such as
    # gpt-oss; json_mode asks the provider for a JSON object (response_format).
    reasoning_effort: str | None = None
    json_mode: bool = False


class EmbeddingSettings(BaseModel):
    """Defaults to local fastembed, since Groq (the default LLM provider
    above) has no embeddings endpoint."""

    provider: str = "fastembed"  # "openai" | "fastembed" | "hash" (tests only)
    base_url: str = "https://api.openai.com/v1"
    api_key: str = "unset"
    model: str = "text-embedding-3-small"
    dimension: int = 1536
    # fastembed only: path to a pre-populated model cache. The container image
    # bakes the model in and sets this, so startup needs no HuggingFace fetch.
    cache_dir: str | None = None


class LangfuseSettings(BaseModel):
    enabled: bool = False
    host: str = "http://localhost:3000"
    public_key: str = ""
    secret_key: str = ""


class JudgeSettings(LLMSettings):
    """The answer-correctness judge (judge.py): an LLM endpoint of its own.

    Defaults to a different model family from the answering model, so the
    judge is not grading its own family's answers, and on Groq a different
    model also draws on a separate daily token quota. An unset api_key falls
    back to LLM__API_KEY (see judge_llm_settings).
    """

    model: str = "openai/gpt-oss-120b"
    # gpt-oss spends output tokens on reasoning before the verdict; low effort
    # keeps that short, and the ceiling leaves room for it.
    reasoning_effort: str | None = "low"
    json_mode: bool = True
    max_tokens: int = 1024
    min_interval_s: float = 2.0


class RerankSettings(BaseModel):
    """Optional cross-encoder reranking of the hybrid-retrieval pool (rerank.py)."""

    enabled: bool = False
    model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    # How many RRF-fused chunks the cross-encoder reorders before the top_k cut.
    candidates: int = 20
    cache_dir: str | None = None


class TelemetrySettings(BaseModel):
    """OpenTelemetry traces and metrics, exported over OTLP/HTTP."""

    enabled: bool = False
    # Collector base URL, e.g. http://localhost:4318. Left unset, the exporters
    # use the standard OTEL_EXPORTER_OTLP_* environment variables instead.
    otlp_endpoint: str | None = None
    service_name: str = "evalgate-rag"
    export_interval_s: int = 15


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_nested_delimiter="__", extra="ignore")

    db: DBSettings = DBSettings()
    llm: LLMSettings = LLMSettings()
    embedding: EmbeddingSettings = EmbeddingSettings()
    langfuse: LangfuseSettings = LangfuseSettings()
    telemetry: TelemetrySettings = TelemetrySettings()
    rerank: RerankSettings = RerankSettings()
    judge: JudgeSettings = JudgeSettings()

    retrieval_top_k: int = 4
    rrf_k: int = 60  # reciprocal-rank-fusion constant


def get_settings() -> Settings:
    return Settings()


def judge_llm_settings(cfg: Settings) -> LLMSettings:
    """The judge's endpoint settings, borrowing the main LLM key when the
    judge has none of its own."""
    if cfg.judge.api_key == "unset":
        return cfg.judge.model_copy(update={"api_key": cfg.llm.api_key})
    return cfg.judge
