import http.client
import json


class InterfaceAPI:
    def __init__(self, api_endpoint, api_key, model_LLM, debug_mode):
        # Parse endpoint into host and base path
        api_endpoint = api_endpoint.replace("https://", "").replace("http://", "")

        if "/" in api_endpoint:
            parts = api_endpoint.split("/", 1)
            self.host = parts[0]
            self.base_path = "/" + parts[1]
        else:
            self.host = api_endpoint
            self.base_path = "/v1"  # Default for OpenAI

        self.api_key = api_key
        self.model_LLM = model_LLM
        self.debug_mode = debug_mode
        self.n_trial = 5

    def get_response(self, prompt_content):
        payload_explanation = json.dumps(
            {
                "model": self.model_LLM,
                "messages": [{"role": "user", "content": prompt_content}],
            }
        )

        headers = {
            "Authorization": "Bearer " + self.api_key,
            "User-Agent": "Apifox/1.0.0 (https://apifox.com)",
            "Content-Type": "application/json",
            "x-api2d-no-cache": "1",
        }

        response = None
        n_trial = 0

        while n_trial < self.n_trial:
            n_trial += 1
            try:
                conn = http.client.HTTPSConnection(self.host)
                full_path = self.base_path + "/chat/completions"

                if self.debug_mode:
                    print(f"API Request to: {self.host}{full_path}")

                conn.request("POST", full_path, payload_explanation, headers)
                res = conn.getresponse()
                data = res.read()
                json_data = json.loads(data)
                response = json_data["choices"][0]["message"]["content"]
                break
            except Exception as e:
                if self.debug_mode:
                    print(f"Error in API (attempt {n_trial}/{self.n_trial}): {e}")
                continue

        return response
