from collections import Counter
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Standardize the text format
def clean_text(col):
    return col.fillna("").str.lower().str.replace(r"[^a-z0-9\s]", " ", regex=True).str.split()

# Convert text to numerical
def encode(col, vocab):
    try:
        transformed = []
        for row in clean_text(col):
            ids = [vocab.get(word, 1) for word in row]
            transformed.append(ids)

        return transformed

    except Exception as e:
        print(f"An error occurred.\n{e}")
        raise    

# Pad
def pad_id(id):
    try:
        X_id_pad = pad_sequences(id, maxlen=256, padding='post', truncating='post')
        return X_id_pad
    except Exception as e:
        print(e)
        raise