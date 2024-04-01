from dataclasses import dataclass

from openai import OpenAI


@dataclass
class ChatMessage:
    content: str
    role: str = "user"

    def __str__(self):
        return self.content


class ChatBot:
    def __init__(self, api_key, api_organization, model="gpt-3.5-turbo"):
        self.client = OpenAI(api_key=api_key, organization=api_organization)
        self.model = model

    def get_completion(
        self,
        chat_messages: ChatMessage | list[ChatMessage],
        max_tokens=150,
        stop=None,
        temperature=1,
        n=1,
    ):
        """
        Get completion from openai chatbot
        """
        chat_messages = (
            [chat_messages] if isinstance(chat_messages, ChatMessage) else chat_messages
        )
        messages: list[dict[str, str]] = [
            {"role": message.role, "content": message.content}
            for message in chat_messages
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            n=n,
            stop=stop,
            temperature=temperature,
        )
        return response.choices[0].message.content
