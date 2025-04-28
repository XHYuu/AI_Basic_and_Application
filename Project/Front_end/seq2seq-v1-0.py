import os
from datetime import datetime
from pathlib import Path

from openai import OpenAI

KIMI_API_KEY = "sk-rgxAPpDWCjSjvhdwvr7khgETRTzKWFqDZGS1FbFx8l11BVoq"

client = OpenAI(
    api_key=KIMI_API_KEY,
    base_url="https://api.moonshot.cn/v1",
)

completion = client.chat.completions.create(
    model="moonshot-v1-8k",
    messages=[
        {"role": "system",
         "content": "你要扮演一名医疗助手，你需要根据患者的症状描述提供回答，包括但不限于：病情诊断、追问更多细节、提供健康建议。请用英文回答。"},
        {"role": "user",
         "content": "I am experiencing symptoms of mild fever, headache, and body aches. I recently took a trip to an area with high risk of Lyme disease. Should I start taking antiboitics?"}
    ],
    temperature=0.3,
)
answer = completion.choices[0].message.content
answer = answer.replace("I'm not a doctor, but ", "")

print(answer)
# dir_path = Path('output')
# dir_path.mkdir(exist_ok=True)
#
# # 通过 API 我们获得了 Kimi 大模型给予我们的回复消息（role=assistant）
# with open(dir_path / 'kimi-answers-history.txt', 'a', encoding='utf8') as f:
#     f.write(f'''>>>>> {datetime.now().isoformat()}
# {answer}
# <<<<<
#
# ''')
