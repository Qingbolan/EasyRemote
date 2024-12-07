from easyremote import easyremote
import numpy as np
from PIL import Image

# 处理单个图片
@easyremote()
def process_image(image: Image.Image) -> Image.Image:
    return image.resize((100, 100))

# 处理视频帧
@easyremote()
def process_video_frames(frames: np.ndarray):
    # frames shape: [frames, height, width, channels]
    for i in range(frames.shape[0]):
        yield frames[i]  # 逐帧处理

# 处理复杂数据结构
@easyremote()
def process_data(data: dict) -> dict:
    return {
        'processed': True,
        'data': data
    }

# 使用示例
if __name__ == '__main__':
    # 图片处理
    img = Image.open('test.jpg')
    result = process_image(img)
    result.save('processed.jpg')

    # 视频处理
    video = np.random.rand(10, 480, 640, 3)
    for frame in process_video_frames(video):
        print(frame.shape)

    # 复杂数据
    data = {
        'image': img,
        'array': np.array([1, 2, 3]),
        'text': 'hello'
    }
    result = process_data(data)
    print(result)