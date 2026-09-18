import torch
import clip
from torchvision import transforms


class CLIPScore:
    def __init__(self, device="cpu"):
        self.device = device

        # ✅ CLIP backbone
        self.model, _ = clip.load("ViT-L/14", device=device)
        self.model.eval()

        # ✅ aesthetic head
        self.aesthetic_model = torch.nn.Sequential(
            torch.nn.Linear(768, 1024),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(1024, 128),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 64),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(64, 16),
            torch.nn.Linear(16, 1),
        ).to(device)

        # ⚠️ 改成你的路径
        state_dict = torch.load(
            "ava+logos-l14-linearMSE.pth",
            map_location=device,
            weights_only=False
        )

        state_dict = {k.replace("layers.", ""): v for k, v in state_dict.items()}
        self.aesthetic_model.load_state_dict(state_dict)
        self.aesthetic_model.eval()

        # resize
        self.resize = transforms.Resize((224, 224))

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: (B,3,H,W) in [0,1]
        return: mean aesthetic score
        """
        with torch.no_grad():
            images = self.resize(images)

            feats = self.model.encode_image(images.to(self.device))
            feats = feats / feats.norm(dim=-1, keepdim=True)

            scores = self.aesthetic_model(feats)
            return scores.mean()