"""
Ollama LLM wrapper for local model execution.

This module provides OllamaLLM and OllamaEmbeddings classes that interface
with locally running Ollama models, separated from the project's starter code.
"""
from typing import Dict, List, Optional, Union
from websocietysimulator.llm.llm import LLMBase

try:
    import ollama
except Exception as _e:
    raise RuntimeError(
        "The 'ollama' Python package is required for OllamaLLM. "
        "Install it with `pip install ollama`. "
        f"Original error: {_e}"
    ) from _e


class OllamaLLM(LLMBase):
    """Ollama wrapper which uses the installed `ollama` Python API only.

    This class assumes the `ollama` Python package is installed and will
    raise helpful errors if required entrypoints aren't present. It does not
    attempt to call the `ollama` CLI.
    """

    def __init__(self, model: str = "mistral", timeout: int = 60):
        super().__init__(model)
        self.timeout = timeout

    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert list of message dicts to a single prompt string."""
        if isinstance(messages, list):
            return "\n".join([m.get("content", "") for m in messages])
        return str(messages)

    def __call__(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 500,
        stop_strs: Optional[List[str]] = None,
        n: int = 1
    ) -> Union[str, List[str]]:
        """Call Ollama to generate a response.
        
        Args:
            messages: List of input messages with role and content
            model: Optional model override
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum tokens in response
            stop_strs: Optional list of stop strings (not currently used)
            n: Number of responses to generate
            
        Returns:
            Generated text response(s)
        """
        prompt = self._messages_to_prompt(messages)
        model_name = model or self.model

        result = ollama.generate(model_name, prompt)

        output = getattr(result, "response", None)
        if output is None:
            raise RuntimeError("ollama.generate returned no 'response' field")
        
        output_text = str(output).strip()
        
        if n == 1:
            return output_text
        return [output_text] * n

    def get_embedding_model(self, embed_model: Optional[str] = None):
        """Return an OllamaEmbeddings instance.

        Args:
            embed_model: Optional separate embedding model name to use with Ollama
                        (e.g. 'all-minilm'). Defaults to 'all-minilm'.
        
        Returns:
            OllamaEmbeddings instance configured for this model
        """
        return OllamaEmbeddings(
            model=self.model,
            embed_model=embed_model or 'all-minilm',
            timeout=self.timeout
        )


class OllamaEmbeddings:
    """Lightweight embeddings wrapper for Ollama.

    This implementation uses the ollama Python API to generate embeddings
    for text documents and queries.
    """

    def __init__(
        self,
        model: str = "mistral",
        embed_model: str = "all-minilm",
        timeout: int = 60
    ):
        """Initialize Ollama embeddings wrapper.

        Args:
            model: Primary model name (used for reference)
            embed_model: Explicit embedding model name to use with Ollama
            timeout: Subprocess timeout (not currently used)
        """
        self.model = model
        self.timeout = timeout
        self.embed_model = embed_model or model

    def _embed_single(self, text: str) -> List[float]:
        """Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        result = ollama.embed(self.embed_model, [text])
        emb = getattr(result, "embeddings", None)
        
        if not emb or not isinstance(emb, list):
            raise RuntimeError("ollama.embed did not return a list of embeddings")
        
        return emb[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple documents.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        for text in texts:
            embeddings.append(self._embed_single(text))
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a query text.
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        return self._embed_single(text)
