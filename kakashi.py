import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertModel, ViTFeatureExtractor, ViTModel
import torch
from flickr8k import create_dataloaders
class CrossModalFusionModel(nn.Module):
    def __init__(self, d_model, num_patches, num_tokens, window_size, text_d, image_d):
        super(CrossModalFusionModel, self).__init__()
        self.d_model = d_model
        self.num_patches = num_patches
        self.num_tokens = num_tokens
        self.text_d = text_d
        self.image_d = image_d
        self.text_projection = nn.Linear(text_d, d_model)
        self.image_projection = nn.Linear(image_d, d_model)

        self.image_attention = SlidingWindowAttention(d_model,8, window_size)
        self.text_attention = SlidingWindowAttention(d_model,8, window_size)

    def forward(self, images, capitions,text_input_ids):

        text_features = capitions
        image_features = images
        capitions = capitions.to(self.text_projection.weight.dtype)
        capitions = capitions.to(self.text_projection.weight.device)

        if self.text_projection is not None:
            text_features = self.text_projection(capitions)

        if self.image_projection is not None:
            image_features = self.image_projection(images)

        # 融合特征
        fused_image_features = self.image_attention(image_features, text_features)
        fused_text_features = self.text_attention(text_features, image_features)

        # 返回融合后的图像和文本特征
        return fused_image_features, fused_text_features


class SlidingWindowAttention(nn.Module):
    def __init__(self, d_model,num_heads, window_size):
        super(SlidingWindowAttention, self).__init__()
        self.d_model = d_model
        self.window_size = window_size
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)


    def forward(self, x, context):
        B, N, D = x.shape
        T = context.shape[1]
        output = torch.zeros_like(x)

        for i in range(N):
            left = max(0, i - self.window_size // 2)
            right = min(T, i + self.window_size // 2 + 1)
            context_window = context[:, left:(right+1), :]

            if context_window.shape[1] < self.window_size:
                padding = torch.zeros(B, self.window_size - context_window.shape[1], D).to(context.device)
                context_window = torch.cat([context_window, padding], dim=1)

            Q = x[:, i, :].unsqueeze(1)  # (B, 1, D)
            K = context_window  # (B, window_size, D)
            V = context_window  # (B, window_size, D)

            # 多头注意力计算
            attn_output, _ = self.multihead_attn(Q, K, V)
            output[:, i, :] = attn_output.squeeze(1) + x[:, i, :]

        return output


import random



from torch.nn.functional import pairwise_distance


def generate_triplet_data_from_batch(images, captions, caplens):
    """
    从一个批次中生成三元组数据：锚点图像、正样本图像和负样本图像。

    参数:
    - images: 一个包含批次中所有图像的张量
    - captions: 一个包含批次中所有文本描述的张量
    - caplens: 一个包含批次中每个文本描述长度的张量

    返回:
    - pos_images: 正样本图像张量
    - neg_images: 负样本图像张量
    """
    batch_size = images.size(0)
    pos_images = []
    neg_images = []

    # 遍历批次中的每个样本
    for i in range(batch_size):
        pos_image = get_random_positive_sample(images, captions, i)
        neg_image = get_random_negative_sample(images, captions, i)

        pos_images.append(pos_image)
        neg_images.append(neg_image)

    # 将列表转换为张量
    pos_images = torch.stack(pos_images)
    neg_images = torch.stack(neg_images)

    return pos_images, neg_images


def get_positive_sample(images, captions, index):
    """
    从给定批次中选择一个与指定索引对应的正样本图像。

    参数:
    - images: 所有图像的张量
    - captions: 所有文本描述的张量
    - index: 当前锚点图像的索引

    返回:
    - 一个正样本图像
    """
    # 获取锚点图像的文本描述
    anchor_caption = captions[index]

    # 寻找具有相同文本描述的其他图像
    for i in range(images.size(0)):
        if i != index and torch.equal(captions[i], anchor_caption):
            return images[i]

    # 如果找不到正样本，则随机选择一个与锚点图像不同的图像作为正样本
    other_indices = list(range(images.size(0)))
    other_indices.remove(index)
    pos_index = torch.randint(len(other_indices), (1,)).item()
    return images[other_indices[pos_index]]


def get_negative_sample(images, captions, index):
    """
    从给定批次中选择一个与指定索引对应的负样本图像。

    参数:
    - images: 所有图像的张量
    - captions: 所有文本描述的张量
    - index: 当前锚点图像的索引

    返回:
    - 一个负样本图像
    """
    # 获取锚点图像的文本描述
    anchor_caption = captions[index]

    # 寻找具有不同文本描述的其他图像
    for i in range(images.size(0)):
        if i != index and not torch.equal(captions[i], anchor_caption):
            return images[i]

    # 如果找不到负样本，则随机选择一个与锚点图像不同的图像作为负样本
    other_indices = list(range(images.size(0)))
    other_indices.remove(index)
    neg_index = torch.randint(len(other_indices), (1,)).item()
    return images[other_indices[neg_index]]


def get_hard_negative_sample(images, captions, anchor_output, index):
    """
    获取与锚点图像距离最接近但不是正样本的负样本。

    参数:
    - images: 所有图像的张量
    - captions: 所有文本描述的张量
    - anchor_output: 锚点图像经过模型输出的特征
    - index: 当前锚点图像的索引

    返回:
    - 一个负样本图像
    """
    anchor_feature = anchor_output[index].unsqueeze(0)
    distances = F.pairwise_distance(anchor_feature, anchor_output)
    distances[index] = float('inf')  # 避免选择自身为负样本

    # 按距离从小到大排序
    sorted_indices = torch.argsort(distances)

    # 选择最近但不是正样本的负样本
    for neg_index in sorted_indices:
        if not torch.equal(captions[neg_index], captions[index]):
            return images[neg_index]

    # 如果没有找到合适的负样本，则随机选择一个负样本
    return get_random_negative_sample(images, captions, index)
def get_random_positive_sample(images, captions, index):
    """
    随机选择一个与当前图像对应的正样本。

    参数:
    - images: 所有图像的张量
    - captions: 所有文本描述的张量
    - index: 当前锚点图像的索引

    返回:
    - 一个随机正样本图像
    """
    anchor_caption = captions[index]
    candidates = [i for i in range(len(captions)) if i != index and torch.equal(captions[i], anchor_caption)]

    if candidates:
        pos_index = random.choice(candidates)
        return images[pos_index]
    else:
        # 如果没有找到相同描述的正样本，随机选择不同的图像作为正样本
        other_indices = list(range(images.size(0)))
        other_indices.remove(index)
        pos_index = random.choice(other_indices)
        return images[pos_index]
def get_random_negative_sample(images, captions, index):
    """
    随机选择一个与当前图像不同的负样本。

    参数:
    - images: 所有图像的张量
    - captions: 所有文本描述的张量
    - index: 当前锚点图像的索引

    返回:
    - 一个随机负样本图像
    """
    negative_indices = []

    for i in range(images.size(0)):
        if i != index and not torch.equal(captions[i], captions[index]):
            negative_indices.append(i)

    if negative_indices:
        neg_index = torch.randint(len(negative_indices), (1,)).item()
        return images[negative_indices[neg_index]]
    else:
        return images[(index + 1) % images.size(0)]  # 如果没有找到负样本，则返回下一个图像


class TripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # 计算锚点和正样本之间的距离
        distance_positive = F.pairwise_distance(anchor.view(anchor.size(0), -1), positive.view(positive.size(0), -1))
        # 计算锚点和负样本之间的距离
        distance_negative = F.pairwise_distance(anchor.view(anchor.size(0), -1), negative.view(negative.size(0), -1))

        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()

if __name__ == '__main__':
    images_dir = './Images'
    dataset_file = './dataset_flickr8k.json'
    batch_size = 32
    num_patches = 196
    num_tokens = 128

    vit_feature_extractor = ViTFeatureExtractor.from_pretrained('D:\\study\\vit')
    vit_model = ViTModel.from_pretrained('D:\\study\\vit')

    model = CrossModalFusionModel(d_model=768, num_patches=num_patches, num_tokens=num_tokens, window_size=30, text_d=32, image_d=768)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)


    train_loader, val_loader, test_loader = create_dataloaders(images_dir, dataset_file, batch_size, vit_feature_extractor,vit_model)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    triplet_loss_fn = TripletLoss(margin=0.2)

    num_epochs = 1
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0

        for images, captions, caplens in train_loader:
            images = images.to(device)
            captions = captions.to(device)
            caplens = caplens.to(device)

            pos_images, neg_images = generate_triplet_data_from_batch(images, captions, caplens)

            anchor_output, _ = model(images, captions, caplens)
            positive_output, _ = model(pos_images, captions, caplens)
            negative_output, _ = model(neg_images, captions, caplens)

            loss = triplet_loss_fn(anchor_output, positive_output, negative_output)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            print(f'one step is over :{running_loss}')


        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')

    torch.save(model.state_dict(), 'model.pth')
