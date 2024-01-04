from transformers import BertForSequenceClassification
import torch
from transformers import AutoTokenizer
from config import BASE_MODEL_NAME, DEVICE, MODEL
from utils import clean_sentence, timeit, clean_sentences

# Instantiate the model with the same architecture used during training
loaded_model = BertForSequenceClassification.from_pretrained(BASE_MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
# Load the saved state dictionary
loaded_model.load_state_dict(
    torch.load(
        MODEL,
        map_location=torch.device(DEVICE),
    )
)
# Move the loaded model to the GPU if needed
loaded_model.to(DEVICE)
label_dict = {"Command": 0, "Question": 1}
label_dict_inverse = {v: k for k, v in label_dict.items()}


def predict(model, tokenizer, sentences):
    # Tokenize input sentences
    encoded_data = tokenizer.batch_encode_plus(
        sentences,
        add_special_tokens=True,
        return_attention_mask=True,
        pad_to_max_length=True,
        max_length=256,
        truncation=True,
        return_tensors="pt",
    )

    input_ids = encoded_data["input_ids"]
    attention_masks = encoded_data["attention_mask"]

    # Move tensors to device
    input_ids = input_ids.to(DEVICE)
    attention_masks = attention_masks.to(DEVICE)

    # Make predictions
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_masks)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=1).cpu().numpy()

    return predictions


@timeit
def prediction(sentences_to_predict):
    sentences_to_predict = clean_sentences(sentences_to_predict)
    # Display predictions
    predicted_labels = predict(loaded_model, tokenizer, sentences_to_predict)
    predicted_labels_names = [label_dict_inverse[label] for label in predicted_labels]
    pedicted_sentences = []
    for sentence, label_name in zip(sentences_to_predict, predicted_labels_names):
        pedicted_sentences.append(
            f"Sentence: {sentence} | Predicted Intent: {label_name}"
        )
    return pedicted_sentences


if __name__ == "__main__":
    sentences_to_predict = ["What is the weather like today?", "Tell me a joke."]
    prediction(sentences_to_predict)
