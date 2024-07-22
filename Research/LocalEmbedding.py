from transformers import AutoTokenizer, AutoModel
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Field
import torch
import tiktoken


class LocalEmbeddings(BaseModel, Embeddings):
    """
    Local embedding models using a PyTorch model from Hugging Face.
    """

    model_name: str = Field(..., description="The name of the Hugging Face model to use.")
    model: AutoModel = None
    tokenizer: AutoTokenizer = None

    class Config:
        arbitrary_types_allowed = True  # Allow arbitrary types

    def __init__(self, **data):
        super().__init__(**data)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.eval()  # Set the model to evaluation mode

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text using the local Hugging Face model."""
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
        return embeddings

    def embed_documents(self, texts: list[str], **kwargs) -> list[list[float]]:
        """Embed a list of documents using the local model."""
        embeddings = []
        for text in texts:
            embeddings.append(self.embed_text(text))
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query using the local model."""
        return self.embed_text(text)

    def __call__(self, texts: list[str], **kwargs) -> list[list[float]]:
        """Allows the instance to be callable, conforming to expected API in Chroma."""
        return self.embed_documents(texts, **kwargs)