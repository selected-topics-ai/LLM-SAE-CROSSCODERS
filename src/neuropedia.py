import json
import requests


def ask_model(prompt: str,
            model_id: str,
            layer: str,
            feature_index: int,
            steer_coef: int,
            temperature: float=0.5,
            n_tokens: int=16,
            frequency_penalty: float=1.0,
            seed: int=16,
            strength_multiplier=1) -> None:

    FEATURE = {
        "modelId": model_id,
        "layer": layer,
        "index": feature_index,
        "strength": 5
    }

    # make the request
    url = "https://www.neuronpedia.org/api/steer"
    data = {
        "prompt": prompt,
        "modelId": model_id,
        "features": [FEATURE],
        "temperature": temperature,
        "n_tokens": n_tokens,
        "freq_penalty": frequency_penalty,
        "seed": seed,
        "strength_multiplier": strength_multiplier,
    }
    headers = {"Content-Type": "application/json"}

    # send request
    response = requests.post(url, json=data, headers=headers)
    json_response = response.json()
    formatted_response = json.dumps(json_response, indent=4)
    print(formatted_response)


if __name__ == "__main__":

    print(ask_model(
        prompt="What is 1/2 + 1/3? Return answer in format ANSWER: (<answer>)",
        model_id="deepseek-r1-distill-llama-8b",
        layer="15-llamascope-slimpj-res-32k",
        feature_index=30939,
        n_tokens=128,
        steer_coef=5,
    ))