import json
import sys
import time
from transformers import DistilBertTokenizerFast
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "../consult_segment/output/bert_model.pth"

id2label = {
    0: "medication-side-effect",
    1: "insurance-related",
    2: "symptom-description",
    3: "pharmaceutical-use",
    4: "vaccine-related"
}


def model_inference(user_input: str, model_path):
    try:
        model = torch.load(model_path)
        model = model.to(device)
        model.eval()

        tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits

        return torch.argmax(logits, dim=-1).item()
    except Exception as e:
        print(f"Precess error: {str(e)}")
        breakpoint()


def check_database():
    pass
    return 1


def main():
    for line in sys.stdin:
        data = json.loads(line.strip())
        user_input = data.get("user_input")
        model_output = model_inference(user_input, model_path)
        question_label = id2label[model_output]
        print(f"We have learned that you need to consult about {question_label} issues.", flush=True)
        print(f"We are checking for you now.", flush=True)
        check_result = check_database()
        print(f"The result is: {check_result}", flush=True)
        break


if __name__ == '__main__':
    main()
