from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Text2TextGenerationPipeline,
)
import torch

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-zh-en")


def translate(text):
    translation = Text2TextGenerationPipeline(model=model, tokenizer=tokenizer)

    if translation:
        return translation(text)[0]["generated_text"]
    else:
        return ""
