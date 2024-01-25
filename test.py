import os
from datasets import load_dataset


os.environ["http_proxy"] = "socks5h://210.75.240.139:11300"
os.environ["https_proxy"] = "socks5h://210.75.240.139:11300"

dataset = load_dataset("drt/complex_web_questions", 'complexwebquestions_test')