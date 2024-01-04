import re
import time


# Function to clean sentences
def clean_sentence(sentence):
    cleaned_sentence = re.sub(r"[^a-zA-Z0-9\s]", "", sentence)
    cleaned_sentence = re.sub(r"\s+", " ", cleaned_sentence).strip()
    return cleaned_sentence.lower()


def clean_sentences(sentences):
    cleaned_sentences = []
    for sentence in sentences:
        cleaned_sentences.append(clean_sentence(sentence))
    return cleaned_sentences


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
