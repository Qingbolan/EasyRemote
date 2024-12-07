from transformers import pipeline
from easyremote import ComputeNode

node = ComputeNode(
    vps_address="127.0.0.1:8080",
    node_id="ml-processor"
)

# 加载模型
sentiment_analyzer = pipeline("sentiment-analysis")
text_generator = pipeline("text-generation")

@node.register
def analyze_sentiment(text: str) -> dict:
    """情感分析"""
    result = sentiment_analyzer(text)
    return result[0]

@node.register(stream=True)
def generate_text(prompt: str, max_length: int = 100):
    """流式文本生成"""
    generation = text_generator(
        prompt,
        max_length=max_length,
        num_return_sequences=1,
        pad_token_id=50256
    )
    
    # 模拟流式输出
    text = generation[0]['generated_text']
    words = text.split()
    
    for i in range(0, len(words), 3):
        chunk = " ".join(words[i:i+3])
        yield chunk

if __name__ == "__main__":
    node.serve()
