import torch
from transformers import AutoTokenizer

from models.bert_classifier import BERTClassifier

class PhishingDetector:
    def __init__(self, model_path: str):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = BERTClassifier()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained("monologg/kobert", trust_remote_code=True)

    def predict(self, text: str) -> float:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs)

            if logits.ndim != 2 or logits.shape[0] == 0 or logits.shape[1] == 0:
                print(f"Invalid logits shape: {logits.shape}")
                return 0.0  # safe fallback

            probs = torch.softmax(logits, dim=-1)

            if probs.shape[-1] == 2:
                score = probs[0][1].item()
            elif probs.shape[-1] == 1:
                score = torch.sigmoid(logits)[0][0].item()
            else:
                print(f"Unexpected logits shape: {logits.shape}")
                return 0.0

        return score
