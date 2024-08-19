import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, ViTFeatureExtractor


class Flickr8KDataset(Dataset):
    def __init__(self, images_dir, dataset_file, vit_feature_extractor,vitmodel, split='train', transform=None):
        """
        Args:
            images_dir (str): 目录路径，其中包含所有图像。
            dataset_file (str): JSON 文件路径，其中包含图像和描述的映射。
            vit_feature_extractor (callable): ViT 的特征提取器，用于处理图像。
            split (str): 'train', 'val' 或 'test'，表示数据集的划分。
            transform (callable, optional): 额外的图像预处理函数（可选）。
        """
        self.images_dir = images_dir
        self.vit_feature_extractor = vit_feature_extractor
        self.transform = transform

        # 载入 JSON 数据
        with open(dataset_file, 'r') as f:
            self.dataset = json.load(f)

        # 根据 split 标签筛选数据
        self.dataset = [img_data for img_data in self.dataset['images'] if img_data['split'] == split]

        # 初始化 BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('D:\\study\\bert')
        self.vit_model = vitmodel


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # 获取图像数据
        img_data = self.dataset[idx]
        img_filename = img_data['filename']

        # 载入图像
        img_path = os.path.join(self.images_dir, img_filename)
        image = Image.open(img_path).convert('RGB')

        # 图像预处理
        if self.transform:
            image = self.transform(image)

        # 使用 ViT 处理图像
        image = self.vit_feature_extractor(images=image, return_tensors="pt").pixel_values
        with torch.no_grad():  # 不计算梯度
            image = self.vit_model(pixel_values=image).last_hidden_state.squeeze(0)

        # 获取文本描述（多个）
        captions = img_data['sentences']
        captions = [self.tokenizer.encode(caption['raw'], max_length=32, truncation=True, padding='max_length') for
                    caption in captions]

        # 计算每个文本的长度
        caplens = [len(caption) for caption in captions]

        # 转换为 tensor
        captions = torch.tensor(captions, dtype=torch.long)
        caplens = torch.tensor(caplens, dtype=torch.long)

        return image, captions, caplens


def create_dataloaders(images_dir, dataset_file, batch_size, vit_feature_extractor,vitmodel, num_workers=4):
    """
    创建训练集、验证集和测试集的数据加载器
    """
    # 创建训练集、验证集和测试集的实例
    train_dataset = Flickr8KDataset(images_dir=images_dir, dataset_file=dataset_file,
                                    vit_feature_extractor=vit_feature_extractor,vitmodel=vitmodel, split='train')
    val_dataset = Flickr8KDataset(images_dir=images_dir, dataset_file=dataset_file,
                                  vit_feature_extractor=vit_feature_extractor,vitmodel=vitmodel, split='val')
    test_dataset = Flickr8KDataset(images_dir=images_dir, dataset_file=dataset_file,
                                   vit_feature_extractor=vit_feature_extractor,vitmodel=vitmodel, split='test')

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                             pin_memory=True)

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    images_dir = './Images'  # 图像目录
    dataset_file = './dataset_flickr8k.json'  # JSON 文件
    batch_size = 32

    # 初始化 ViT 的特征提取器
    vit_feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

    # 创建数据加载器
    train_loader, val_loader, test_loader = create_dataloaders(images_dir, dataset_file, batch_size,
                                                               vit_feature_extractor)

    # 测试数据加载器
    for images, captions, caplens in train_loader:
        print("训练集:")
        print(images.shape)  # (batch_size, 3, 224, 224)
        print(captions.shape)  # (batch_size, num_captions, max_len)
        print(caplens.shape)  # (batch_size, num_captions)
        break

    for images, captions, caplens in val_loader:
        print("验证集:")
        print(images.shape)  # (batch_size, 3, 224, 224)
        print(captions.shape)  # (batch_size, num_captions, max_len)
        print(caplens.shape)  # (batch_size, num_captions)
        break

    for images, captions, caplens in test_loader:
        print("测试集:")
        print(images.shape)  # (batch_size, 3, 224, 224)
        print(captions.shape)  # (batch_size, num_captions, max_len)
        print(caplens.shape)  # (batch_size, num_captions)
        break
