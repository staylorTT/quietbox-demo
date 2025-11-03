import os
import json
import jwt
from openai import OpenAI




class ResponderTenstorrent:
    def __init__(self, model_name="meta-llama/Llama-3.1-8B-Instruct",
                 base_url="http://localhost:8000/v1", max_new_tokens=128):
        self.model_name = model_name
        jwt_secret = os.getenv("JWT_SECRET")
        if jwt_secret is None:
            raise ValueError("User must set JWT_SECRET")
        json_payload = json.loads("{\"team_id\": \"tenstorrent\", \"token_id\": \"debug-test\"}")
        openai_api_key = jwt.encode(json_payload, jwt_secret, algorithm="HS256")
        self.client = OpenAI(
            base_url=base_url,
            api_key=openai_api_key,
        )

    def respond(self, user_text):
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "developer", "content": "You are a concise, helpful assistant."},
                {"role": "user", "content": user_text},
            ]
        )

        # extract completion content
        reply = completion.choices[0].message.content
        return reply
