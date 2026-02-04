import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import os
import random
import numpy as np

from PIL import Image, ImageDraw

from datasets import load_from_disk

from .trainer import OminiModel, get_config, train
from ..pipeline.flux_omini import Condition, convert_to_condition, generate


# class ImageConditionDataset(Dataset):
#     def __init__(
#         self,
#         base_dataset,
#         condition_size=(512, 512),
#         target_size=(512, 512),
#         condition_type: str = "canny",
#         drop_text_prob: float = 0.1,
#         drop_image_prob: float = 0.1,
#         return_pil_image: bool = False,
#         position_scale=1.0,
#     ):
#         self.base_dataset = base_dataset
#         self.condition_size = condition_size
#         self.target_size = target_size
#         self.condition_type = condition_type
#         self.drop_text_prob = drop_text_prob
#         self.drop_image_prob = drop_image_prob
#         self.return_pil_image = return_pil_image
#         self.position_scale = position_scale
#
#         self.to_tensor = T.ToTensor()
#
#     def __len__(self):
#         return len(self.base_dataset)
#
#     def __get_condition__(self, image, condition_type):
#         condition_size = self.condition_size
#         position_delta = np.array([0, 0])
#         if condition_type in ["canny", "coloring", "deblurring", "depth"]:
#             image, kwargs = image.resize(condition_size), {}
#             if condition_type == "deblurring":
#                 blur_radius = random.randint(1, 10)
#                 kwargs["blur_radius"] = blur_radius
#             condition_img = convert_to_condition(condition_type, image, **kwargs)
#         elif condition_type == "depth_pred":
#             depth_img = convert_to_condition("depth", image)
#             condition_img = image.resize(condition_size)
#             image = depth_img.resize(condition_size)
#         elif condition_type == "fill":
#             condition_img = image.resize(condition_size).convert("RGB")
#             w, h = image.size
#             x1, x2 = sorted([random.randint(0, w), random.randint(0, w)])
#             y1, y2 = sorted([random.randint(0, h), random.randint(0, h)])
#             mask = Image.new("L", image.size, 0)
#             draw = ImageDraw.Draw(mask)
#             draw.rectangle([x1, y1, x2, y2], fill=255)
#             if random.random() > 0.5:
#                 mask = Image.eval(mask, lambda a: 255 - a)
#             condition_img = Image.composite(
#                 image, Image.new("RGB", image.size, (0, 0, 0)), mask
#             )
#         elif condition_type == "sr":
#             condition_img = image.resize(condition_size)
#             position_delta = np.array([0, -condition_size[0] // 16])
#         else:
#             raise ValueError(f"Condition type {condition_type} is not  implemented.")
#         return condition_img, position_delta
#
#     def __getitem__(self, idx):
#         image = self.base_dataset[idx]["jpg"]
#         image = image.resize(self.target_size).convert("RGB")
#         item = self.base_dataset[idx]
#         if "json" in item and "prompt" in item["json"]:
#             description = item["json"]["prompt"]
#         elif "caption" in item:  # 有些数据集叫 caption
#             description = item["caption"]
#         else:
#             # 如果都没有，给一个默认提示词
#             # description = "colorize the high quality image"
#             description = ""
#
#         condition_size = self.condition_size
#         position_scale = self.position_scale
#
#         condition_img, position_delta = self.__get_condition__(
#             image, self.condition_type
#         )
#
#         # Randomly drop text or image (for training)
#         drop_text = random.random() < self.drop_text_prob
#         drop_image = random.random() < self.drop_image_prob
#
#         if drop_text:
#             description = ""
#         if drop_image:
#             condition_img = Image.new("RGB", condition_size, (0, 0, 0))
#
#         return {
#             "image": self.to_tensor(image),
#             "condition_0": self.to_tensor(condition_img),
#             "condition_type_0": self.condition_type,
#             "position_delta_0": position_delta,
#             "description": description,
#             "idx": idx,
#             **({"pil_image": [image, condition_img]} if self.return_pil_image else {}),
#             **({"position_scale_0": position_scale} if position_scale != 1.0 else {}),
#         }
class ImageConditionDataset(Dataset):
    def __init__(
            self,
            base_dataset,
            condition_size=(512, 512),
            target_size=(512, 512),
            condition_type: str = "coloring",  # 默认改为 coloring
            drop_text_prob: float = 0.1,
            drop_image_prob: float = 0.1,
            return_pil_image: bool = False,
            position_scale=1.0,
            # 新增参数：训练模式配置
            training_mode: str = "single",  # "mixed" or "single"
            color_hint_prob: float = 0.3,  # color_hint 概率
            gray_no_text_prob: float = 0.4,  # 无description灰度图概率
            gray_with_text_prob: float = 0.3,  # 有description灰度图概率
            # color_hint 参数
            num_patches_range: tuple = (1, 10),
            patch_size_range: tuple = (8, 16),
    ):
        self.base_dataset = base_dataset
        self.condition_size = condition_size
        self.target_size = target_size
        self.condition_type = condition_type
        self.drop_text_prob = drop_text_prob
        self.drop_image_prob = drop_image_prob
        self.return_pil_image = return_pil_image
        self.position_scale = position_scale

        # 新增属性
        self.training_mode = training_mode
        self.color_hint_prob = color_hint_prob
        self.gray_no_text_prob = gray_no_text_prob
        self.gray_with_text_prob = gray_with_text_prob
        self.num_patches_range = num_patches_range
        self.patch_size_range = patch_size_range

        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.base_dataset)

    def __get_color_hint__(self, image, variant="auto"):
        """生成带有颜色提示的灰度图 - 多种变体"""
        rgb_img = image.convert("RGB")
        gray_img = rgb_img.convert("L").convert("RGB")

        width, height = rgb_img.size

        gray_array = np.array(gray_img)
        rgb_array = np.array(rgb_img)

        if variant == "auto":
            variant = random.choice(["patches", "circles", "scattered"])

        if variant == "patches":
            # 方形patch
            num_patches = random.randint(*self.num_patches_range)
            for _ in range(num_patches):
                patch_size = random.randint(*self.patch_size_range)
                if width > patch_size and height > patch_size:
                    x = random.randint(0, width - patch_size)
                    y = random.randint(0, height - patch_size)
                    gray_array[y:y + patch_size, x:x + patch_size] = rgb_array[y:y + patch_size, x:x + patch_size]

        elif variant == "circles":
            # 圆形区域
            num_circles = random.randint(1, 10)
            for _ in range(num_circles):
                cx, cy = random.randint(0, width), random.randint(0, height)
                radius = random.randint(8, 16)
                y_coords, x_coords = np.ogrid[:height, :width]
                mask = (x_coords - cx) ** 2 + (y_coords - cy) ** 2 <= radius ** 2
                gray_array[mask] = rgb_array[mask]

        elif variant == "scattered":
            # 散点
            num_points = random.randint(20, 50)
            for _ in range(num_points):
                x, y = random.randint(0, width - 1), random.randint(0, height - 1)
                r = random.randint(2, 8)
                y1, y2 = max(0, y - r), min(height, y + r)
                x1, x2 = max(0, x - r), min(width, x + r)
                gray_array[y1:y2, x1:x2] = rgb_array[y1:y2, x1:x2]

        return Image.fromarray(gray_array)

    def __get_condition__(self, image, condition_type):
        condition_size = self.condition_size
        position_delta = np.array([0, 0])

        if condition_type == "color_hint":
            # 新增：color_hint 条件
            image = image.resize(condition_size)
            condition_img = self.__get_color_hint__(image)

        elif condition_type in ["canny", "coloring", "deblurring", "depth"]:
            image, kwargs = image.resize(condition_size), {}
            if condition_type == "deblurring":
                blur_radius = random.randint(1, 10)
                kwargs["blur_radius"] = blur_radius
            condition_img = convert_to_condition(condition_type, image, **kwargs)

        elif condition_type == "depth_pred":
            depth_img = convert_to_condition("depth", image)
            condition_img = image.resize(condition_size)
            image = depth_img.resize(condition_size)

        elif condition_type == "fill":
            condition_img = image.resize(condition_size).convert("RGB")
            w, h = image.size
            x1, x2 = sorted([random.randint(0, w), random.randint(0, w)])
            y1, y2 = sorted([random.randint(0, h), random.randint(0, h)])
            mask = Image.new("L", image.size, 0)
            draw = ImageDraw.Draw(mask)
            draw.rectangle([x1, y1, x2, y2], fill=255)
            if random.random() > 0.5:
                mask = Image.eval(mask, lambda a: 255 - a)
            condition_img = Image.composite(
                image, Image.new("RGB", image.size, (0, 0, 0)), mask
            )

        elif condition_type == "sr":
            condition_img = image.resize(condition_size)
            position_delta = np.array([0, -condition_size[0] // 16])

        else:
            raise ValueError(f"Condition type {condition_type} is not implemented.")

        return condition_img, position_delta

    def _sample_condition_type_and_text(self, original_description):
        """
        根据概率采样条件类型和是否保留文本描述
        - 50%: color_hint, 无description
        - 30%: coloring (灰度图), 无description
        - 20%: coloring (灰度图), 有description
        """
        rand = random.random()

        if rand < self.color_hint_prob:
            # 50%: color_hint，无description
            return "color_hint", ""
        elif rand < self.color_hint_prob + self.gray_no_text_prob:
            # 30%: 灰度图，无description
            return "coloring", ""
        else:
            # 20%: 灰度图，有description
            return "coloring", original_description

    def __getitem__(self, idx):
        image = self.base_dataset[idx]["jpg"]
        image = image.resize(self.target_size).convert("RGB")
        item = self.base_dataset[idx]

        # 获取原始描述
        if "json" in item and "prompt" in item["json"]:
            original_description = item["json"]["prompt"]
        elif "caption" in item:
            original_description = item["caption"]
        else:
            original_description = ""

        condition_size = self.condition_size
        position_scale = self.position_scale

        # 根据训练模式选择条件类型
        if self.training_mode == "mixed":
            # 混合训练模式：按概率采样
            condition_type, description = self._sample_condition_type_and_text(
                original_description
            )
        else:
            # 单一模式：使用指定的 condition_type
            condition_type = self.condition_type
            description = original_description

            # 原有的随机drop逻辑
            if random.random() < self.drop_text_prob:
                description = ""

        # 生成条件图像
        condition_img, position_delta = self.__get_condition__(
            image, condition_type
        )

        # 混合模式下不使用原有的drop逻辑，单一模式下保留
        if self.training_mode != "mixed":
            if random.random() < self.drop_image_prob:
                condition_img = Image.new("RGB", condition_size, (0, 0, 0))

        return {
            "image": self.to_tensor(image),
            "condition_0": self.to_tensor(condition_img),
            "condition_type_0": condition_type,  # 使用实际的condition_type
            "position_delta_0": position_delta,
            "description": description,
            "idx": idx,
            **({"pil_image": [image, condition_img]} if self.return_pil_image else {}),
            **({"position_scale_0": position_scale} if position_scale != 1.0 else {}),
        }


@torch.no_grad()
def test_function(model, save_path, file_name):
    condition_size = model.training_config["dataset"]["condition_size"]
    target_size = model.training_config["dataset"]["target_size"]

    position_delta = model.training_config["dataset"].get("position_delta", [0, 0])
    position_scale = model.training_config["dataset"].get("position_scale", 1.0)

    adapter = model.adapter_names[2]
    condition_type = model.training_config["condition_type"]
    test_list = []

    if condition_type in ["canny", "coloring", "deblurring", "depth"]:
        image = Image.open("assets/vase_hq.jpg")
        image = image.resize(condition_size)
        condition_img = convert_to_condition(condition_type, image, 5)
        condition = Condition(condition_img, adapter, position_delta, position_scale)
        test_list.append((condition, "A beautiful vase on a table."))
    elif condition_type == "depth_pred":
        image = Image.open("assets/vase_hq.jpg")
        image = image.resize(condition_size)
        condition = Condition(image, adapter, position_delta, position_scale)
        test_list.append((condition, "A beautiful vase on a table."))
    elif condition_type == "fill":
        condition_img = (
            Image.open("./assets/vase_hq.jpg").resize(condition_size).convert("RGB")
        )
        mask = Image.new("L", condition_img.size, 0)
        draw = ImageDraw.Draw(mask)
        a = condition_img.size[0] // 4
        b = a * 3
        draw.rectangle([a, a, b, b], fill=255)
        condition_img = Image.composite(
            condition_img, Image.new("RGB", condition_img.size, (0, 0, 0)), mask
        )
        condition = Condition(condition, adapter, position_delta, position_scale)
        test_list.append((condition, "A beautiful vase on a table."))
    elif condition_type == "super_resolution":
        image = Image.open("assets/vase_hq.jpg")
        image = image.resize(condition_size)
        condition = Condition(image, adapter, position_delta, position_scale)
        test_list.append((condition, "A beautiful vase on a table."))
    else:
        raise NotImplementedError
    os.makedirs(save_path, exist_ok=True)
    for i, (condition, prompt) in enumerate(test_list):
        generator = torch.Generator(device=model.device)
        generator.manual_seed(42)

        res = generate(
            model.flux_pipe,
            prompt=prompt,
            conditions=[condition],
            height=target_size[1],
            width=target_size[0],
            generator=generator,
            num_inference_steps = 1,
            model_config=model.model_config,
            kv_cache=model.model_config.get("independent_condition", False),
        )
        file_path = os.path.join(save_path, f"{file_name}_{condition_type}_{i}.jpg")
        res.images[0].save(file_path)


def main():
    # Initialize
    config = get_config()
    training_config = config["train"]
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    print("local_rank:", local_rank)
    # Load dataset text-to-image-2M
    # dataset = load_dataset(
    #     "webdataset",
    #     data_files={"train": training_config["dataset"]["urls"]},
    #     split="train",
    #     cache_dir="cache/t2i2m",
    #     num_proc=32,
    # )
    dataset = load_from_disk("/root/autodl-fs")
    # Initialize custom dataset
    # dataset = ImageConditionDataset(
    #     dataset,
    #     condition_size=training_config["dataset"]["condition_size"],
    #     target_size=training_config["dataset"]["target_size"],
    #     condition_type=training_config["condition_type"],
    #     drop_text_prob=training_config["dataset"]["drop_text_prob"],
    #     drop_image_prob=training_config["dataset"]["drop_image_prob"],
    #     position_scale=training_config["dataset"].get("position_scale", 1.0),
    # )
    # 创建混合训练的数据集
    dataset = ImageConditionDataset(
        base_dataset=dataset,
        condition_size=(512, 512),
        target_size=(512, 512),
        training_mode="single",  # 启用混合模式
        # 概率配置 (总和应为1.0)
        color_hint_prob=0.5,  # 50% color_hint 无文本
        gray_no_text_prob=0.3,  # 30% 灰度图 无文本
        gray_with_text_prob=0.2,  # 20% 灰度图 有文本
        # color_hint 参数
        num_patches_range=(1, 10),
        patch_size_range=(8, 16),
    )

    # print(f"【】最终 device: cuda:{local_rank}")
    # Initialize model
    trainable_model = OminiModel(
        flux_pipe_id=config["flux_path"],
        lora_config=training_config["lora_config"],
        lora_path=training_config.get("lora_path", None),
        time_layers_path=training_config.get("time_layers_path", None),
        device=f"cuda:{local_rank}",
        dtype=getattr(torch, config["dtype"]),
        optimizer_config=training_config["optimizer"],
        model_config=config.get("model", {}),
        gradient_checkpointing=training_config.get("gradient_checkpointing", False),
    )

    train(dataset, trainable_model, config, test_function)


if __name__ == "__main__":
    main()
