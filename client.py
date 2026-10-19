import requests
import base64
from PIL import Image
import io

# Flask API地址
API_URL = "http://localhost:5001/generate"

# 请求参数
params = {
    "lam_abs": 722,
    "lam_em": 754,
    "eps": 109647
}

try:
    # 发送POST请求
    response = requests.post(API_URL, json=params)
    
    if response.status_code == 200:
        result = response.json()
        
        # 打印返回的SMILES
        print("收到的SMILES:", result["smiles"])
        
        # 解码base64图片数据
        image_data = base64.b64decode(result["image"])
        image = Image.open(io.BytesIO(image_data))
        
        # 显示图片
        image.show()
        
        # 或者保存图片到本地
        with open("generated_image.png", "wb") as f:
            f.write(image_data)
            print("图片已保存为 generated_image.png")
    else:
        print("请求失败，状态码:", response.status_code)
        print("错误信息:", response.json()["error"])

except requests.exceptions.ConnectionError:
    print("连接失败！请确保Flask服务正在运行")