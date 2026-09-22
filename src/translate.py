from transformers import MarianMTModel, MarianTokenizer

MODEL_NAME = "Helsinki-NLP/opus-mt-en-de"

tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
model = MarianMTModel.from_pretrained(MODEL_NAME)


def translate_text(text):
    inputs = tokenizer(
        [text],
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    translated = model.generate(**inputs)

    result = tokenizer.decode(
        translated[0],
        skip_special_tokens=True
    )

    return result


if __name__ == "__main__":
    text = "I love machine learning."

    result = translate_text(text)

    print("English:", text)
    print("German:", result)