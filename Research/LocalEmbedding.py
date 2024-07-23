from transformers import AutoModel, AutoTokenizer
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Field
import torch
from typing import Any, List, Optional
from sentence_transformers import SentenceTransformer

class LocalEmbeddings(BaseModel, Embeddings):
    """
    Local embedding models using a PyTorch model from Hugging Face or SentenceTransformer.
    """

    model_path: str = Field(..., description="Path to the local model directory.")
    tokenizer_path: Optional[str] = Field(None, description="Path to the local tokenizer directory")
    model: Optional[Any] = None
    tokenizer: Optional[Any] = None

    class Config:
        arbitrary_types_allowed = True  # Allow arbitrary types

    def __init__(self, **data):
        super().__init__(**data)
        if not self.tokenizer_path:
            self.model = SentenceTransformer(self.model_path)
        else:
            self.model = AutoModel.from_pretrained(self.model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
            self.model.eval()  # Set the model to evaluation mode

    def embed_text(self, text: str) -> List[float]:
        """Embed a single text using the local model."""
        if isinstance(self.model, SentenceTransformer):
            embeddings = self.model.encode(text, normalize_embeddings=True).tolist()
        else:
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
            tokens_tensor = inputs["input_ids"]

            with torch.no_grad():
                outputs = self.model(input_ids=tokens_tensor)
            embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
        return embeddings

    def embed_documents(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Embed a list of documents using the local model."""
        embeddings = [self.embed_text(text) for text in texts if text]
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query using the local model."""
        return self.embed_text(text)

    def __call__(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Allows the instance to be callable, conforming to expected API in Chroma."""
        return self.embed_documents(texts, **kwargs)
