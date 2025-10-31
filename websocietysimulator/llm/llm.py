from typing import Dict, List, Optional, Union
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from .infinigence_embeddings import InfinigenceEmbeddings
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import json
import logging
import subprocess
logger = logging.getLogger("websocietysimulator")

class LLMBase:
    def __init__(self, model: str = "qwen2.5-72b-instruct"):
        """
        Initialize LLM base class
        
        Args:
            model: Model name, defaults to deepseek-chat
        """
        self.model = model
        
    def __call__(self, messages: List[Dict[str, str]], model: Optional[str] = None, temperature: float = 0.0, max_tokens: int = 500, stop_strs: Optional[List[str]] = None, n: int = 1) -> Union[str, List[str]]:
        """
        Call LLM to get response
        
        Args:
            messages: List of input messages, each message is a dict containing role and content
            model: Optional model override
            max_tokens: Maximum tokens in response, defaults to 500
            stop_strs: Optional list of stop strings
            n: Number of responses to generate, defaults to 1
            
        Returns:
            Union[str, List[str]]: Response text from LLM, either a single string or list of strings
        """
        raise NotImplementedError("Subclasses need to implement this method")
    
    def get_embedding_model(self):
        """
        Get the embedding model for text embeddings
        
        Returns:
            OpenAIEmbeddings: An instance of OpenAI's text embedding model
        """
        raise NotImplementedError("Subclasses need to implement this method")

class InfinigenceLLM(LLMBase):
    def __init__(self, api_key: str, model: str = "qwen2.5-72b-instruct"):
        """
        Initialize Deepseek LLM
        
        Args:
            api_key: Deepseek API key
            model: Model name, defaults to qwen2.5-72b-instruct
        """
        super().__init__(model)
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://cloud.infini-ai.com/maas/v1"
        )
        self.embedding_model = InfinigenceEmbeddings(api_key=api_key)
        
    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=10, max=300),  # 等待时间从10秒开始，指数增长，最长300秒
        stop=stop_after_attempt(10)  # 最多重试10次
    )
    def __call__(self, messages: List[Dict[str, str]], model: Optional[str] = None, temperature: float = 0.0, max_tokens: int = 500, stop_strs: Optional[List[str]] = None, n: int = 1) -> Union[str, List[str]]:
        """
        Call Infinigence AI API to get response with rate limit handling
        
        Args:
            messages: List of input messages, each message is a dict containing role and content
            model: Optional model override
            max_tokens: Maximum tokens in response, defaults to 500
            stop_strs: Optional list of stop strings
            n: Number of responses to generate, defaults to 1
            
        Returns:
            Union[str, List[str]]: Response text from LLM, either a single string or list of strings
        """
        try:
            response = self.client.chat.completions.create(
                model=model or self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop_strs,
                n=n,
            )
            
            if n == 1:
                return response.choices[0].message.content
            else:
                return [choice.message.content for choice in response.choices]
        except Exception as e:
            if "429" in str(e):
                logger.warning("Rate limit exceeded")
            else:
                logger.error(f"Other LLM Error: {e}")
            raise e
    
    def get_embedding_model(self):
        return self.embedding_model

class OpenAILLM(LLMBase):
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        """
        Initialize OpenAI LLM
        
        Args:
            api_key: OpenAI API key
            model: Model name, defaults to gpt-3.5-turbo
        """
        super().__init__(model)
        self.client = OpenAI(api_key=api_key)
        self.embedding_model = OpenAIEmbeddings(api_key=api_key)
        
    def __call__(self, messages: List[Dict[str, str]], model: Optional[str] = None, temperature: float = 0.0, max_tokens: int = 500, stop_strs: Optional[List[str]] = None, n: int = 1) -> Union[str, List[str]]:
        """
        Call OpenAI API to get response
        
        Args:
            messages: List of input messages, each message is a dict containing role and content
            model: Optional model override
            max_tokens: Maximum tokens in response, defaults to 500
            stop_strs: Optional list of stop strings
            n: Number of responses to generate, defaults to 1
            
        Returns:
            Union[str, List[str]]: Response text from LLM, either a single string or list of strings
        """
        response = self.client.chat.completions.create(
            model=model or self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop_strs,
            n=n
        )
        
        if n == 1:
            return response.choices[0].message.content
        else:
            return [choice.message.content for choice in response.choices]
    
    def get_embedding_model(self):
        return self.embedding_model 


class OllamaLLM(LLMBase):
    """Simple local Ollama wrapper using the `ollama` CLI.

    This implementation invokes the `ollama generate <model> "prompt"`
    CLI command. It is intentionally lightweight for quick prototyping.

    Requirements:
    - `ollama` CLI must be installed and available on PATH.
    - A local Ollama model with the provided name must be downloaded/available.
    """

    def __init__(self, model: str = "llama2", timeout: int = 60):
        super().__init__(model)
        self.timeout = timeout

    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        # Convert list of message dicts to a single prompt string
        if isinstance(messages, list):
            return "\n".join([m.get("content", "") for m in messages])
        return str(messages)

    def __call__(self, messages: List[Dict[str, str]], model: Optional[str] = None, temperature: float = 0.0, max_tokens: int = 500, stop_strs: Optional[List[str]] = None, n: int = 1) -> Union[str, List[str]]:
        prompt = self._messages_to_prompt(messages)
        model_name = model or self.model

        # Try commonly-used Ollama CLI subcommands in order of likelihood.
        # Newer Ollama uses `ollama run <model> <prompt>`. Older examples may use
        # `ollama generate` which some versions don't support. Try `run` first,
        # then fall back to `generate` if `run` fails with unknown-command.
        tried = []
        for subcmd in ("run", "generate"):
            cmd = ["ollama", subcmd, model_name, prompt]
            tried.append(" ".join(cmd))
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=self.timeout, check=True)
                output = result.stdout.strip()
                if n == 1:
                    return output
                else:
                    return [output] * n
            except subprocess.CalledProcessError as e:
                stderr = (e.stderr or "").lower()
                # If this subcommand isn't supported, try the next one.
                if "unknown command" in stderr or "is not a valid command" in stderr:
                    logger.debug(f"ollama subcommand '{subcmd}' not supported, trying next (stderr: {e.stderr})")
                    continue
                # Otherwise it's a real failure; raise with context.
                logger.error(f"Ollama {subcmd} failed (exit {e.returncode}): {e.stderr}")
                raise
            except Exception as e:
                logger.error(f"Ollama invocation error with subcommand '{subcmd}': {e}")
                raise

        # If we reached here, none of the tried subcommands worked.
        logger.error(f"Ollama invocation failed; tried commands: {tried}")
        raise RuntimeError(f"Ollama CLI did not accept any tested subcommands. Tried: {tried}")

    def get_embedding_model(self, embed_model: Optional[str] = None):
        """Return an OllamaEmbeddings instance.

        Args:
            embed_model: optional separate embedding model name to use with Ollama (e.g. a small embed model).
        """
        return OllamaEmbeddings(model=self.model, embed_model='all-minilm', timeout=self.timeout)


class OllamaEmbeddings:
    """Lightweight embeddings wrapper that uses the `ollama embed` CLI.

    Notes:
    - This implementation calls `ollama embed <model> <text>` once per document.
    - It attempts to parse JSON output. If the CLI returns a bare list of floats
      (e.g., `[0.1, 0.2, ...]`) that will be used directly. For batch usage this
      is intentionally simple; we can optimize to a single batched CLI call
      if needed and supported by your local Ollama version.
    """

    def __init__(self, model: str = "mistral", embed_model: Optional[str] = None, timeout: int = 60):
        """Lightweight embeddings wrapper.

        Args:
            model: primary model name (used for reference)
            embed_model: explicit embedding model name to call with ollama (if different)
            timeout: subprocess timeout
        """
        self.model = model
        self.timeout = timeout
        # embed_model explicitly chooses the model used for embedding commands
        self.embed_model = embed_model or model

    def _embed_single(self, text: str):
        try:
            import ollama
        except Exception as e:
            raise RuntimeError(
                "The 'ollama' Python package is required for OllamaEmbeddings but is not available. "
                "Install it with `pip install ollama` or use a different embedding backend (sentence-transformers)."
            ) from e

        # Call the Python API and expect an object with an `embeddings` property.
        try:
            # ollama.embed(model, inputs) typically accepts a list of strings and
            # returns an object with .embeddings (list of vectors)
            if not hasattr(ollama, "embed"):
                raise RuntimeError("Installed 'ollama' package does not expose 'embed' API. Upgrade ollama or use another embedder.")

            result = ollama.embed(self.embed_model, [text])

            # Prefer attribute-style access
            if hasattr(result, "embeddings"):
                emb = result.embeddings
                if isinstance(emb, list) and len(emb) > 0:
                    return emb[0]

            # Fallback: dict-like structure
            if isinstance(result, dict):
                if "embeddings" in result and isinstance(result["embeddings"], list) and result["embeddings"]:
                    return result["embeddings"][0]
                if "data" in result and isinstance(result["data"], list) and result["data"]:
                    first = result["data"][0]
                    if isinstance(first, dict) and "embedding" in first:
                        return first["embedding"]
                    if isinstance(first, list):
                        return first

            raise RuntimeError("Unable to extract embedding vector from ollama.embed() result. Ensure the embed model returns numeric embeddings via the ollama Python API.")

        except Exception as e:
            raise RuntimeError("Failed to get embeddings from ollama Python API. Ensure 'ollama' is installed and the specified embed model returns numeric embeddings via the Python API.") from e

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for t in texts:
            embeddings.append(self._embed_single(t))
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        return self._embed_single(text)
