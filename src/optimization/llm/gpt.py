import os
from dataclasses import dataclass

import openai
from dotenv import load_dotenv


@dataclass
class ChatMessage:
    role: str
    content: str

    def __str__(self):
        return f"{self.role}: {self.content}"


class ChatBot:
    def __init__(self, model="gpt-3.5-turbo"):
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.organization = os.getenv("OPENAI_API_ORGANIZATION")
        openai.api_key = self.api_key
        openai.organization = self.organization
        self.model = model

    def get_completion(
        self, messages: ChatMessage, max_tokens=150, stop=None, temperature=1, n=1
    ):
        """
        Get completion from openai chatbot
        """
        messages = [
            {"role": message.role, "content": message.content} for message in messages
        ]
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            n=n,
            stop=stop,
            temperature=temperature,
        )