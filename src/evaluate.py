import pandas as pd
import sacrebleu

from translate import translate_text


DATA_PATH = "data/raw/translation.csv"


def evaluate_model():

    df = pd.read_csv(DATA_PATH)

    predictions = []
    references = []

    for _, row in df.iterrows():

        source = row["source"]
        target = row["target"]

        prediction = translate_text(source)

        predictions.append(prediction)
        references.append(target)

        print("Source:", source)
        print("Expected:", target)
        print("Predicted:", prediction)
        print()

    bleu = sacrebleu.corpus_bleu(
        predictions,
        [references]
    )

    print("--------------------------------")
    print("BLEU Score:", bleu.score)
    print("--------------------------------")

    return bleu.score


if __name__ == "__main__":
    evaluate_model()