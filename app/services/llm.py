from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.core.config import get_settings

settings = get_settings()


class EmbeddingService:
    def __init__(self) -> None:
        self.model = OpenAIEmbeddings(model=settings.embedding_model, api_key=settings.openai_api_key)

    def embed(self, text: str) -> list[float]:
        return self.model.embed_query(text)


class ChatService:
    def __init__(self) -> None:
        self.chat = ChatOpenAI(model=settings.chat_model, temperature=0, api_key=settings.openai_api_key)

    def invoke(self, prompt: str) -> str:
        return self.chat.invoke(prompt).content
