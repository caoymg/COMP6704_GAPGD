import torch
from PIL import Image
import torchvision.transforms as transforms
import os
import pandas as pd



seed = 23
torch.manual_seed(seed)

# num = 129
num = 100
imagenet_df = pd.read_csv("/path/imagenet1000.csv")


for sigma_noise in [16/255]:

    for attack_type in ["comp6704_gaussian_alpha03"]:

        print(f"=========== attack_type: {attack_type}, sigma_noise:{sigma_noise} =========== ")
        
        for index, row in imagenet_df.iterrows():
            
            if index < num:
                image_name = row['Image Names']
                target_text = row['Target Text']

                adv_image_file = f"/path/{attack_type}/{attack_type}_{image_name}.png"

                # 读取图片并转换为tensor
                image = Image.open(adv_image_file).convert('RGB')
                image_tensor = transforms.ToTensor()(image)  # [0,1]范围的tensor
                
                # 生成高斯噪声
                noise = torch.randn_like(image_tensor) * sigma_noise
                
                # 添加噪声并裁剪到有效范围
                noisy_tensor = torch.clamp(image_tensor + noise, 0.0, 1.0)
                
                # 转换回PIL图像并保存
                noisy_image = transforms.ToPILImage()(noisy_tensor)
                
                # 确保目录存在并保存
                output_dir = "/path/gaussian_noise"
                os.makedirs(output_dir, exist_ok=True)
                output_path = f"{output_dir}/{attack_type}_{sigma_noise}_{image_name}.png"
                noisy_image.save(output_path)
                
                # print(f"已保存: {output_path}")


