import os
import time
import openai
from utils.utils import set_environment_variable


@set_environment_variable
def run_llm(
    prompt,
    temperature=0.7,
    max_tokens=256,
    opeani_api_keys=None,
    engine="gpt-3.5-turbo",
    stop="\n",
    stream=False,
):
    if "llama" in engine.lower():
        openai.api_key = "EMPTY"
        openai.api_base = "http://localhost:8000/v1"  # your local llama server port
        engine = openai.Model.list()["data"][0]["id"]
    else:
        if opeani_api_keys is None:
            opeani_api_keys = os.environ["OPENAI_API_KEY"]
        openai.api_key = opeani_api_keys

    messages = [
        {"role": "system", "content": "You are an AI assistant that answers complex questions."}
    ]
    message_prompt = {"role": "user", "content": prompt}
    messages.append(message_prompt)
    f = 0
    while f == 0:
        try:
            response = openai.ChatCompletion.create(
                model=engine,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop,
                stream=stream,
            )
            if not stream:
                result = response["choices"][0]["message"]["content"]
            else:
                result = ""
                for i in response:
                    try:
                        result += i["choices"][0]["delta"]["content"]
                    except Exception as e:
                        break

            f = 1
        except Exception as e:
            print(e)
            if "maximum context length" in str(e):
                messages[-1]["content"] = messages[-1]["content"][4000:]
            print("openai error, retry")
            time.sleep(2)
    return result
