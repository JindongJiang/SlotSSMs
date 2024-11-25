import os
import glob
import random
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from decord import VideoReader, cpu

class VideoDataset(Dataset):
    def __init__(self, root, seq_len=16, resolution=224, image_norm_mode="imagenet"):
        super().__init__()
        
        # Set up image normalization
        if image_norm_mode == "imagenet":
            norm_mean = [0.485, 0.456, 0.406]
            norm_std = [0.229, 0.224, 0.225]
        elif image_norm_mode == "zero_one":
            norm_mean = [0, 0, 0]
            norm_std = [1, 1, 1]
        else:
            raise ValueError("Invalid image_norm_mode")

        # Find all video files
        self.video_files = glob.glob(os.path.join(root, "**/*.mp4"), recursive=True)
        self.video_files = sorted(self.video_files)
        
        # Set parameters
        self.seq_len = seq_len
        self.resolution = resolution
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize(
                resolution, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=norm_mean,
                std=norm_std
            )
        ])

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        
        # Load video
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        
        # Randomly select starting point for sequence
        if total_frames <= self.seq_len:
            start_idx = 0
        else:
            start_idx = random.randint(0, total_frames - self.seq_len)
        
        # Sample consecutive frames
        frame_indices = list(range(start_idx, start_idx + self.seq_len))
        if len(frame_indices) < self.seq_len:
            # Pad with last frame if video is too short
            frame_indices.extend([frame_indices[-1]] * (self.seq_len - len(frame_indices)))
            
        frames = vr.get_batch(frame_indices).asnumpy()
        
        # Process frames
        processed_frames = []
        for frame in frames:
            image = Image.fromarray(frame)
            processed_frame = self.transform(image)
            processed_frames.append(processed_frame)
            
        # Stack frames into tensor
        video_tensor = torch.stack(processed_frames)
        
        return {"pixel_values": video_tensor}

if __name__ == "__main__":
    dataset = VideoDataset("/data/local/jindong/Datasets/clevrer")
    print(len(dataset))
