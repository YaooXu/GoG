import os
import time
import openai
from sklearn import logger
import tiktoken
import httpx
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")


def run_llm(
    prompt,
    temperature=0.7,
    max_tokens=256,
    opeani_api_keys=None,
    engine="gpt-3.5-turbo-0613",
    stop="\n",
    stream=False,
    n=1,
):
    client = openai.OpenAI(
        base_url=os.environ['base_url'],
        api_key=os.environ['opeani_api_keys'],
        http_client=httpx.Client(proxies=os.environ['custom_proxy']) if 'custom_proxy' in os.environ else None,
    )

    messages = [
        {"role": "system", "content": "You are an AI assistant that answers complex questions."}
    ]
    message_prompt = {"role": "user", "content": prompt}
    messages.append(message_prompt)

    if stop and type(stop) is str:
        stop = [stop]

    f = 0
    while f == 0:
        try:
            if len(encoding.encode(prompt)) >= 4096:
                raise RuntimeError("maximum context length of prompt")
            response = client.chat.completions.create(
                model=engine,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop,
                stream=stream,
                n=n,
            )
            results = [response.choices[i].message.content for i in range(n)]

            if stop:
                stop = stop[0]
                for i, result in enumerate(results): 
                    if stop in result:
                        result = result.split(stop)[0]
                    results[i] = result
                    
            f = 1
        except Exception as e:
            if "maximum context length" in str(e):
                logger.error(f"{e}")
                return None
            logger.error(f"{e}, openai error, retry")
            time.sleep(5)
    if n == 1:
        return results[0]
    else:
        return results
