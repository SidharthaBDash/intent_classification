from transformers import BertForSequenceClassification
import torch
import nvidea
from transformers import AutoTokenizer

# Instantiate the model with the same architecture used during training
loaded_model = BertForSequenceClassification.from_pretrained("google/muril-base-cased")

# Load the saved state dictionary
loaded_model.load_state_dict(
    torch.load(
        "/Users/reverie-pc/Desktop/codebase/intent_classification/finetuned_BERT_epoch_5.model"
    )
)
device = "cuda"
# Move the loaded model to the GPU if needed
loaded_model.to(device)

tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")


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
    input_ids = input_ids.to(device)
    attention_masks = attention_masks.to(device)

    # Make predictions
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_masks)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=1).cpu().numpy()

    return predictions


# Assuming you have a label mapping dictionary during training
label_dict = {"Command": 0, "Question": 1}  # Replace with your actual labels

# Create the inverse mapping
label_dict_inverse = {v: k for k, v in label_dict.items()}

import time


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print(
            "\033[1m"
            + "%r >> %2.2f ms" % (method.__name__, (te - ts) * 1000)
            + "\033[0m"
        )
        return result

    return timed


sentences_to_predict = ["What is the weather like today?", "Tell me a joke."]


# Function to clean sentences
def clean_sentence(sentence):
    cleaned_sentence = re.sub(r"[^a-zA-Z0-9\s]", "", sentence)
    cleaned_sentence = re.sub(r"\s+", " ", cleaned_sentence).strip()
    return cleaned_sentence


@timeit
def prediction():
    # Display predictions
    predicted_labels = predict(loaded_model, tokenizer, sentences_to_predict)
    predicted_labels_names = [label_dict_inverse[label] for label in predicted_labels]
    for sentence, label_name in zip(sentences_to_predict, predicted_labels_names):
        print(f"Sentence: {sentence} | Predicted Intent: {label_name}")


prediction()
