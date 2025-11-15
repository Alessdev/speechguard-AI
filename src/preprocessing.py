import pandas as pd
import re
import spacy

try:
    nlp = spacy.load("es_core_news_sm")
except OSError:
    raise OSError("Falta el modelo 'es_core_news_sm'. Instala con: python -m spacy download es_core_news_sm")

def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def lemmatize_text(text: str) -> str:
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_punct])

def preprocess_conversations(df: pd.DataFrame):
    df = df.copy()

    df["text_clean"] = df["text"].apply(normalize_text)
    df["text_lemma"] = df["text_clean"].apply(lemmatize_text)

    grouped = (
        df.groupby("conversation_id")
        .agg(
            full_conversation=("text", lambda x: "\n".join(x)),
            asesor_text=("text", lambda x: "\n".join(
                [t for s, t in zip(df.loc[x.index, "speaker"], x) if s == "asesor"]
            )),
            cliente_text=("text", lambda x: "\n".join(
                [t for s, t in zip(df.loc[x.index, "speaker"], x) if s == "cliente"]
            )),
        )
        .reset_index()
    )

    return df, grouped
