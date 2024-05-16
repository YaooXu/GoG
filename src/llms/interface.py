import os
import time
import openai
from sklearn import logger
import tiktoken

encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")


def run_llm(
    prompt,
    temperature=0.7,
    max_tokens=256,
    opeani_api_keys=None,
    engine="gpt-3.5-turbo-0613",
    stop="\n",
    stream=False,
):
    if "llama" in engine.lower():
        openai.api_key = "EMPTY"
        openai.api_base = "http://localhost:18001/v1"  # your local llama server port
        engine = openai.Model.list()["data"][0]["id"]
        print(engine)
    else:
        if opeani_api_keys is None:
            opeani_api_keys = os.environ["OPENAI_API_KEY"]
        # openai.api_key = opeani_api_keys
        # openai.proxy = {
        #     "http": "socks5h://127.0.0.1:11300",
        #     "https": "socks5h://127.0.0.1:11300",
        # }

    messages = [
        {"role": "system", "content": "You are an AI assistant that answers complex questions."}
    ]
    message_prompt = {"role": "user", "content": prompt}
    messages.append(message_prompt)
    f = 0
    while f == 0:
        try:
            if len(encoding.encode(prompt)) >= 4096:
                raise RuntimeError("maximum context length of prompt")
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
            if "maximum context length" in str(e):
                logger.error(f"{e}")
                return None
            logger.error(f"{e}, openai error, retry")
            time.sleep(2)
    return result
