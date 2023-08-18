import torch
from transformers import BertTokenizer, BertModel
import numpy as np

tokenizer, model, quantized_model, device = None, None, None, None


def load_models(lang):
    global tokenizer, model, quantized_model, device

    model_name = "bert-base-en" if lang == "en" else "bert-base-chinese"
    # Load the tokenizer for traditional Chinese
    tokenizer = BertTokenizer.from_pretrained(model_name)
    # Load the model for traditional Chinese
    model = BertModel.from_pretrained(model_name)
    device = torch.device("cpu")
    # Ensure the model is using CPU
    model = model.to(device)
    quantized_model = torch.quantization.quantize_dynamic(
        model,  # the original model
        {torch.nn.Linear},  # a set of layers to dynamically quantize
        dtype=torch.qint8,  # the target dtype for quantized weights
    )
    quantized_model.eval()
    return model, quantized_model, tokenizer


def get_embeddings(text: str, lang: str = "en"):
    global tokenizer, quantized_model

    if not tokenizer:
        normal_model, quantized_model, tokenizer = load_models(lang)

    # Tokenize the text
    inputs = tokenizer(text, return_tensors="pt")

    # Ensure no GPU is being used
    device = torch.device("cpu")
    inputs.to(device)

    # normal model
    outputs = model(**inputs)
    output = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()

    print("normal model output shape: ", output.shape)
    # Assuming that embeddings is your model's embeddings

    min_val = np.min(output)
    max_val = np.max(output)

    # Scale embeddings to 0-1
    embeddings_scaled = (output - min_val) / (max_val - min_val)

    # Quantize the scaled embeddings to the range 0-255
    quantized_outputs = np.round(embeddings_scaled * 255).astype(np.uint8)

    return quantized_outputs, output, np.asarray([min_val, max_val])
