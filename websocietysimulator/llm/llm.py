from typing import Dict, List, Optional, Union
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from .infinigence_embeddings import InfinigenceEmbeddings
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import json
import logging
try:
    import ollama
except Exception as _e:
    # Fail fast: this module is intended to use the ollama Python API only.
    raise RuntimeError("The 'ollama' Python package is required for OllamaLLM. Install it with `pip install ollama`. Original error: {0}".format(_e)) from _e
import logging
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
    """Ollama wrapper which uses the installed `ollama` Python API only.

    This class assumes the `ollama` Python package is installed and will
    raise helpful errors if required entrypoints aren't present. It does not
    attempt to call the `ollama` CLI.
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

        # Call the single chosen API method directly (based on local probe):
        # use `ollama.generate(model, prompt)` and parse its `.response` field.
        result = ollama.generate(model_name, prompt)

        # Normalize result to a string
        output = None
        # Parse the GenerateResponse: expect a `.response` attribute/string.
        # This intentionally does not implement alternate parsing or fallbacks.
        output = getattr(result, "response")
        if output is None:
            # If the response field is unexpectedly None, raise an informative error
            raise RuntimeError("ollama.generate returned no 'response' field")
        # Some responses may include leading whitespace; normalize.
        output_text = str(output).strip()
        if n == 1:
            return output_text
        return [output_text] * n

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

    def __init__(self, model: str = "mistral", embed_model: str = "all-minilm", timeout: int = 60):
        """Lightweight embeddings wrapper.

        Args:
            model: primary model name (used for reference)
            embed_model: explicit embedding model name to call with ollama (if different). Defaults to `'all-minilm'`.
            timeout: subprocess timeout
        """
        self.model = model
        self.timeout = timeout
        # embed_model explicitly chooses the model used for embedding commands
        self.embed_model = embed_model or model

    def _embed_single(self, text: str):
        # Call the chosen embedding API directly and parse the result in one way.
        # Based on the probe, use `ollama.embed(model, [text])` and expect a
        # response object with an `.embeddings` attribute containing a list of vectors.
        result = ollama.embed(self.embed_model, [text])
        emb = getattr(result, "embeddings")
        if not emb or not isinstance(emb, list):
            raise RuntimeError("ollama.embed did not return a list of embeddings")
        return emb[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for t in texts:
            embeddings.append(self._embed_single(t))
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        return self._embed_single(text)
