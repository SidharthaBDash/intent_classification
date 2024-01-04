import os
import torch

TRAIN_CSV_PATH = os.path.join(os.path.dirname(__file__), "quesion_command.csv")
BASE_MODEL_NAME = "google/muril-base-cased"
MODEL = os.path.join(os.path.dirname(__file__), "finetuned_BERT_epoch_5.model")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
