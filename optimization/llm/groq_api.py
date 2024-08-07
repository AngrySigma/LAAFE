import hydra
from hydra.utils import instantiate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq


class GroqChatBot:
    def __init__(
        self,
        groq_api_key: str,
        model_name: str = "llama3-70b-8192",
        temperature: float = 0,
    ) -> None:
        self.client = ChatGroq(
            temperature=temperature,
            model_name=model_name,
            groq_api_key=groq_api_key,
        )
        self.model = model_name

    def get_completion(
        self,
        chat_messages,
    ) -> str:
        """
        Get completion from groq chatbot
        """
        chat_messages = (
            chat_messages if isinstance(chat_messages, list) else ([chat_messages])
        )
        response = self.client.invoke(chat_messages).content
        return response if isinstance(response, str) else ""


@hydra.main(
    version_base="1.2",
    config_path="D:/PhD/LAAFE/cfg",
    config_name="cfg",
)
def main(cfg):
    llm = instantiate(cfg.llm[cfg.llm.llm_backend])
    messages = [
        SystemMessage(
            content="Answer with Hello Back on any Hello World message."
            "For any other message, answer with Failed",
            role="system",
        ),
        HumanMessage(content="Hello World", role="user"),
    ]
    result = llm.get_completion(messages)
    print(result)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
