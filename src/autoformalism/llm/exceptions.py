"""LLM boundary failures."""


class LLMError(Exception):
    """Base error for structured LLM calls."""


class LLMProviderError(LLMError):
    """Provider failure with an explicit retry classification."""

    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


class LLMResponseError(LLMError):
    """A response could not be validated against the required schema."""


class LLMCacheError(LLMError):
    """A cache entry is malformed or cannot be persisted."""

