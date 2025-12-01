from websocietysimulator.llm.llm import LLMBase
from typing import Dict, List, Optional, Union
from google import genai
from google.genai import types
from langchain_google_genai import GoogleGenerativeAIEmbeddings


class GoogleLLM(LLMBase):
    def __init__(self, api_key: str = None, model: str = "gemini-2.5-flash"):
        """
        Initialize OpenAI LLM
        
        Args:
            api_key: Google AI Studio API key. If unspecified, the following defaults will be used: client api key = env.GEMINI_API_KEY, embedding api key = env.GOOGLE_API_KEY (These two keys should be the same.)
            model: Model name, defaults to gemini-2.5-flash
        """
        super().__init__(model)
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)

    def __call__(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 500,
        stop_strs: Optional[List[str]] = None,
        n: int = 1,
    ) -> Union[str, List[str]]:
        """
        Call Gemini API to get response

        Args:
            messages: List of input messages, each message is a dict containing role and content
            model: Optional model override
            max_tokens: Maximum tokens in response, defaults to 500
            stop_strs: Optional list of stop strings
            n: Number of responses to generate, defaults to 1

        Returns:
            Union[str, List[str]]: Response text from LLM, either a single string or list of strings
        """
        contents = []
        instructions = []
        # Reformat OpenAI messages into Gemini format
        for d in messages:
            if d["role"] == "user":
                contents.append(d["content"])
            else:
                instructions.append(d["content"])

        response = self.client.models.generate_content(
            model=model or self.model,
            contents=' '.join(contents),
            config=types.GenerateContentConfig(
                stop_sequences=stop_strs,
                temperature=temperature,
                max_output_tokens=max_tokens,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
                system_instruction=' '.join(instructions)
            )
        )

        if n == 1:
            return response.text
        else:
            print("WARN: A really lazy implementation of the argument `n` where we just return n copies of a single inference.")
            return [response.text] * n

    def get_embedding_model(self):
        return self.embedding_model 
