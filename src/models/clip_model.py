"""
CLIP + Logistic Regression Classification Pipeline.
Architecture:
Facial Image -> CLIP ViT-B/32 Vision Encoder -> 512-dim Embedding -> Logistic Regression -> 9 Navarasa Classes.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import open_clip
import joblib
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import (
    DEVICE,
    CLIP_MODEL_NAME,
    CLIP_PRETRAINED,
    CLIP_MODEL_PATH,
    CLIP_EMBEDDINGS_CACHE,
    NAVARASA_CLASSES,
    MODELS_DIR,
)
from src.data.preprocessing import get_clip_transforms


class CLIPClassifierPipeline:
    """
    CLIP ViT-B/32 Vision Encoder + Logistic Regression Classifier.
    """

    def __init__(self, device=DEVICE):
        self.device = device
        self.clip_model = None
        self.clip_preprocess = None
        self.classifier = None
        self.embedding_dim = 512
        self.classes = NAVARASA_CLASSES

    def load_clip_encoder(self):
        """Loads and freezes pretrained CLIP ViT-B/32 vision encoder, pruning unused text components."""
        if self.clip_model is None:
            visual_path = MODELS_DIR / "clip_visual_encoder.pt"
            if not visual_path.exists():
                parts = sorted(list(MODELS_DIR.glob("clip_visual_part_*.bin")))
                if parts:
                    import gzip, shutil
                    print(f"Assembling standalone CLIP visual encoder from {len(parts)} binary chunks...")
                    gz_temp = MODELS_DIR / "_temp_clip_visual.gz"
                    with open(gz_temp, "wb") as f_out:
                        for part in parts:
                            with open(part, "rb") as f_in:
                                shutil.copyfileobj(f_in, f_out)
                    with gzip.open(gz_temp, "rb") as f_in:
                        with open(visual_path, "wb") as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    if gz_temp.exists():
                        try:
                            gz_temp.unlink()
                        except Exception:
                            pass

            if visual_path.exists():
                print(f"Loading standalone CLIP visual encoder from {visual_path} on {self.device}...")
                visual = torch.load(visual_path, map_location=self.device, weights_only=False, mmap=True)
                visual.eval()
                for param in visual.parameters():
                    param.requires_grad = False

                from torchvision import transforms
                preprocess = transforms.Compose([
                    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
                    transforms.CenterCrop((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
                ])

                class VisualModelWrapper(torch.nn.Module):
                    def __init__(self, vis):
                        super().__init__()
                        self.visual = vis
                    def encode_image(self, x):
                        return self.visual(x)
                    def forward(self, x):
                        return self.visual(x)

                self.clip_model = VisualModelWrapper(visual)
                self.clip_preprocess = preprocess
                import gc
                gc.collect()
            else:
                print(f"Loading CLIP ({CLIP_MODEL_NAME}, pretrained='{CLIP_PRETRAINED}') on {self.device}...")
                model, _, preprocess = open_clip.create_model_and_transforms(
                    CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED
                )
                model = model.to(self.device)
                model.eval()

                # Memory Optimization: Prune unused text transformer components (~63M params, ~250MB RAM)
                for attr in ["transformer", "token_embedding", "positional_embedding", "text_projection"]:
                    if hasattr(model, attr):
                        try:
                            delattr(model, attr)
                        except Exception:
                            pass
                import gc
                gc.collect()

                # Freeze remaining parameters
                for param in model.parameters():
                    param.requires_grad = False
                self.clip_model = model
                self.clip_preprocess = preprocess
        return self.clip_model, self.clip_preprocess

    def extract_image_embedding(self, image_input):
        """
        Extracts 512-dim normalized embedding for a single PIL image or path.
        """
        self.load_clip_encoder()
        if isinstance(image_input, (str, Path)):
            img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            img = image_input.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image_input)}")

        tensor = self.clip_preprocess(img).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            features = self.clip_model.encode_image(tensor)
            # L2 normalize embedding vector
            features = features / features.norm(dim=-1, keepdim=True)
            emb = features.cpu().numpy().flatten()

        del tensor, features
        return emb

    def extract_batch_embeddings(self, dataloader):
        """
        Extracts embeddings for an entire DataLoader.
        """
        self.load_clip_encoder()
        all_embeddings = []
        all_labels = []

        print(f"Extracting CLIP embeddings for {len(dataloader.dataset)} images...")
        with torch.no_grad():
            for batch_idx, (images, labels, _, _) in enumerate(dataloader):
                images = images.to(self.device)
                features = self.clip_model.encode_image(images)
                features = features / features.norm(dim=-1, keepdim=True)
                all_embeddings.append(features.cpu().numpy())
                all_labels.append(labels.numpy())
                if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(dataloader):
                    print(f"  Processed {min((batch_idx + 1) * dataloader.batch_size, len(dataloader.dataset))}/{len(dataloader.dataset)} images")

        X = np.vstack(all_embeddings)
        y = np.concatenate(all_labels)
        return X, y

    def train_logistic_regression(self, X_train, y_train, X_val, y_val, tune_c=True):
        """
        Trains Logistic Regression on extracted embeddings with hyperparameter tuning on validation split.
        """
        print("\n--- Training Logistic Regression Classifier on CLIP Embeddings ---")
        best_c = 1.0
        best_val_acc = 0.0
        best_model = None

        candidate_c = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0] if tune_c else [1.0]

        for c_val in candidate_c:
            clf = LogisticRegression(
                C=c_val,
                max_iter=1000,
                solver="lbfgs",
                random_state=42
            )
            clf.fit(X_train, y_train)
            val_preds = clf.predict(X_val)
            val_acc = accuracy_score(y_val, val_preds)
            print(f"  Hyperparameter C={c_val:<5} -> Validation Accuracy: {val_acc * 100:.2f}%")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_c = c_val
                best_model = clf

        print(f"\nOptimal Hyperparameter: C={best_c} with Validation Accuracy: {best_val_acc * 100:.2f}%")
        self.classifier = best_model
        return best_model, best_c, best_val_acc

    def save_model(self, save_path=CLIP_MODEL_PATH):
        """Saves trained Logistic Regression model and metadata."""
        if self.classifier is None:
            raise ValueError("No trained classifier to save.")
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "classifier": self.classifier,
            "classes": self.classes,
            "clip_model_name": CLIP_MODEL_NAME,
            "clip_pretrained": CLIP_PRETRAINED,
            "embedding_dim": self.embedding_dim,
        }, save_path)
        print(f"Saved CLIP Logistic Regression model to {save_path}")

    def load_model(self, load_path=CLIP_MODEL_PATH):
        """Loads trained Logistic Regression model."""
        load_path = Path(load_path)
        if not load_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found at {load_path}")
        data = joblib.load(load_path)
        self.classifier = data["classifier"]
        self.classes = data.get("classes", NAVARASA_CLASSES)
        return self.classifier

    def predict(self, image_input):
        """
        Runs single image inference.
        Returns:
            predicted_class: str
            confidence: float (0.0 - 1.0)
            probabilities: dict of {class_name: float}
            embedding: 512-dim np.ndarray
        """
        if self.classifier is None:
            self.load_model()
        embedding = self.extract_image_embedding(image_input)
        emb_2d = embedding.reshape(1, -1)
        probs = self.classifier.predict_proba(emb_2d)[0]
        pred_idx = int(np.argmax(probs))
        pred_class = self.classes[pred_idx]
        confidence = float(probs[pred_idx])
        prob_dict = {cls: float(probs[i]) for i, cls in enumerate(self.classes)}
        return pred_class, confidence, prob_dict, embedding
