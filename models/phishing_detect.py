import torch
from transformers import AutoTokenizer

class PhishingDetector:
    def __init__(self, model_path: str):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  # 필요한 tokenizer로 수정

    def predict(self, text: str) -> float:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits
            score = torch.sigmoid(logits).item()
        return score
