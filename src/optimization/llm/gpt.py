import os
from dataclasses import dataclass

from openai import OpenAI


@dataclass
class ChatMessage:
    role: str
    content: str

    def __str__(self):
        return f"{self.role}: {self.content}"


class ChatBot:
    def __init__(self, api_key, api_organization, model="gpt-3.5-turbo"):
        self.client = OpenAI(api_key=api_key, organization=api_organization)
        self.model = model

    def get_completion(
            self, messages: ChatMessage, max_tokens=150, stop=None,
            temperature=1, n=1
    ):
        """
        Get completion from openai chatbot
        """
        messages = [
            {"role": message.role, "content": message.content} for message in
            messages
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
