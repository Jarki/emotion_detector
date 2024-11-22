import logging

import openai

from . import prompts
from . import enums


logger = logging.getLogger(__name__)

class ResponseScorer:
    def __init__(self, api_key: str, model: str="gpt-4o", promptset: prompts.ResponseScorerPrompts|None = None) -> None:
        self.client = openai.Client(api_key=api_key)
        self.model = model

        if promptset is not None:
            self.promptset = promptset
        else:
            self.promptset = prompts.default_promptset

    def score_sarcasm(self, answer: str, question: str|None = None) -> enums.ClassScore:
        return self._send_and_parse_response(self.promptset.sarcasm, self._form_request(answer, question))

    def score_anger(self, answer: str, question: str|None = None) -> enums.ClassScore:
        return self._send_and_parse_response(self.promptset.anger, self._form_request(answer, question))

    def _form_request(self, answer: str, question: str|None) -> str:
        request = ""
        if question is not None:
            request = f"Q: {question}\n"
        
        request += f"A: {answer}"
        return request

    def _send_and_parse_response(self, system_prompt: str, user_prompt: str) -> enums.ClassScore:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system", "content": system_prompt
                },
                {
                    "role": "user", "content": user_prompt
                }
            ]
        )
        logger.debug(f"Used {completion.usage.total_tokens} tokens")
        return self._parse_response(completion.choices[0].message.content)
    
    def _parse_response(self, message: str) -> enums.ClassScore:
        message = message.strip().upper()
        
        if message in enums.ClassScore.__members__:
            return enums.ClassScore[message]
        
        return enums.ClassScore.INCONCLUSIVE