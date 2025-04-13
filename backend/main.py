# backend/main.py
import os
import sys
import time
import torch
from torchvision.transforms.functional import to_pil_image
from flask import Flask, request, jsonify
from flask_cors import CORS

# 添加项目根目录到 sys.path，确保能够导入 configs 和 models
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from models.generator import Generator
from configs.config import latent_dim
from backend.utils.device import get_device

app = Flask(__name__, static_folder="../static")
CORS(app)

# 实例化生成器（条件生成器）
generator = Generator(latent_dim, n_classes=7)

device = get_device()
generator.to(device)

# 这里请替换成你训练好的权重文件路径
# 示例：
# weight_path = os.path.join(os.path.dirname(__file__), "generator_weights.pth")
# generator.load_state_dict(torch.load(weight_path, map_location=device))
# generator.eval()

# 条件映射（全转为小写）
char_type_mapping = {
    "monster": 0,
    "human": 1,
    "animal": 2,
}
color_theme_mapping = {
    "random": 0,
    "red": 1,
    "blue": 2,
    "green": 3,
}

def one_hot(index, num_classes):
    vec = torch.zeros(1, num_classes, device=device)
    vec[0, index] = 1.0
    return vec

def generate_image(seed, condition):
    if seed is not None and str(seed).strip() != "" and str(seed).lower() != "random":
        torch.manual_seed(int(seed))
    with torch.no_grad():
        noise = torch.randn((1, latent_dim), device=device)
        # 注意这里调用生成器时传入 condition 参数
        generated_tensor = generator(noise, condition)
        image = to_pil_image(generated_tensor.squeeze(0).cpu())
        time.sleep(1)  # 可选：模拟延时，方便前端显示加载提示
    return image

@app.route("/api/generate-character", methods=["POST"])
def api_generate_character():
    try:
        data = request.get_json()
        seed = data.get("seed")
        char_type_str = data.get("character_type", "monster").lower()
        color_theme_str = data.get("color_theme", "random").lower()
        print(f"Received: seed={seed}, character_type={char_type_str}, color_theme={color_theme_str}")
        
        # 将条件转换为 one-hot 编码
        char_index = char_type_mapping.get(char_type_str, 0)
        color_index = color_theme_mapping.get(color_theme_str, 0)
        char_one_hot = one_hot(char_index, 3)   # 3类：monster, human, animal
        color_one_hot = one_hot(color_index, 4)   # 4类：random, red, blue, green
        condition = torch.cat((char_one_hot, color_one_hot), dim=1)  # 总维度 7
        
        image = generate_image(seed, condition)
        
        # 保存图片到 static/generated 文件夹
        generated_dir = os.path.join(app.static_folder, "generated")
        os.makedirs(generated_dir, exist_ok=True)
        filename = "character_generated.png"  # 如需防止覆盖，可加入时间戳
        file_path = os.path.join(generated_dir, filename)
        image.save(file_path, format="PNG")
        
        # 构造图片 URL
        image_url = f"http://127.0.0.1:5000/static/generated/{filename}"
        print("生成图片URL:", image_url)
        
        return jsonify({"image_url": image_url, "message": "Generation successful"}), 200
    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
