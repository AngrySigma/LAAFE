from dataclasses import dataclass

from openai import OpenAI


@dataclass
class ChatMessage:
    content: str
    role: str = "user"

    def __str__(self):
        return self.content


class MessageHistory:
    def __init__(self, messages: list[ChatMessage] | None = None):
        self.messages = messages if messages is not None else []

    def add_message(self, content: str, role: str = "user"):
        self.messages.append(ChatMessage(role=role, content=content))

    def add_pipeline_evaluation(self, pipeline: str, metrics: float):
        self[-2].content += f"\nPipeline: {pipeline}, Metrics: {metrics}"

    def __str__(self):
        return "\n".join([str(message) for message in self.messages])

    def __add__(self, other):
        return MessageHistory(self.messages + other.messages)

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, item):
        return self.messages[item]

    def __iter__(self):
        return iter(self.messages)


class ChatBot:
    def __init__(self, api_key, api_organization, model="gpt-3.5-turbo"):
        self.client = OpenAI(api_key=api_key, organization=api_organization)
        self.model = model

    def get_completion(
        self,
        messages: ChatMessage | list[ChatMessage],
        max_tokens=150,
        stop=None,
        temperature=1,
        n=1,
    ):
        """
        Get completion from openai chatbot
        """
        messages = [messages] if isinstance(messages, ChatMessage) else messages
        messages = [
            {"role": message.role, "content": message.content} for message in messages
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
