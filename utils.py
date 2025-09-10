import os
import json
from torch.utils.data import Dataset, DataLoader
import torchvision.io as io, transforms
import numpy as np

class JsonlVideoDataset(Dataset):
    def __init__(self, jsonl_path, video_dir, 
                 transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))):
        """
        Args:
            jsonl_path (str): 存放视频信息的 jsonl 文件
            video_dir (str): 存放视频 mp4 文件的目录
            transform (callable, optional): 可选的视频预处理函数，比如 resize, normalize
        """
        self.video_dir = video_dir
        self.transform = transform

        # 读取 jsonl 文件并保存文件名列表
        self.video_files = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                oss_url = data.get("oss_url")
                if oss_url:
                    filename = oss_url.split("/")[-1]  # 提取最后一段作为文件名
                    file_path = os.path.join(video_dir, filename)
                    if os.path.exists(file_path):
                        self.video_files.append(file_path)
                    else:
                        print(f"Warning: file not found {file_path}")

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = self.video_files[idx]

        # torchvision.io.read_video 返回 (video, audio, info)
        # video: Tensor [T, H, W, C], dtype=torch.uint8
        video, _, _ = io.read_video(video_path, pts_unit="sec")

        # 转成 float，并换维度 -> [T, C, H, W]
        video = video.permute(0, 3, 1, 2).float() / 255.0  

        if self.transform:
            video = self.transform(video)


        return video

def get_video_dataset_var(dataset):
    all_pixels = []
    for video in dataset:
        # video shape: [T, C, H, W]
        all_pixels.append(video.numpy().ravel())
    all_pixels = np.concatenate(all_pixels)
    return np.var(all_pixels)

def get_training_data_and_data_var(jsonl_path, video_dir, batch_size=1, shuffle=True, num_workers=2):
    dataset = JsonlVideoDataset(jsonl_path, video_dir)
    training_data =  DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    x_train_var = get_video_dataset_var(dataloader)
    return training_data, x_train_var

if __name__ == "__main__":
    jsonl_path = "/home/project/video/select/sampled_1000.jsonl"
    video_dir = "/home/project/video/select/sample"


    dataloader, x_train_var = get_training_data_and_data_var(jsonl_path, video_dir, batch_size=1)
    print("Dataset video variance:", x_train_var)
    for batch in dataloader:
        video = batch.squeeze(0)  # [T, C, H, W]
        print("Video shape:", video.shape)
        break
