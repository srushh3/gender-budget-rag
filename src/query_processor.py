from langdetect import detect
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


MODEL_NAME = "facebook/nllb-200-distilled-600M"

print("Loading translation model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)


def detect_language(query):
    """Detect the language of the query."""
    try:
        return detect(query)
    except Exception:
        return "en"


def translate_to_english(query, language):
    """Translate a non-English query to English using NLLB."""
    try:
        source_language = {
            "hi": "hin_Deva",
        }.get(language)

        if source_language is None:
            return query

        tokenizer.src_lang = source_language

        inputs = tokenizer(
            query,
            return_tensors="pt",
            truncation=True,
            max_length=256
        )

        outputs = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"),
            max_length=128
        )

        translated = tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True
        )[0]

        return translated

    except Exception as e:
        print(f"Translation failed: {e}")
        return query


def process_query(query):
    """
    Detect the query language and create an English version
    when necessary.

    The original query is always preserved.
    """
    language = detect_language(query)

    if language == "en":
        return {
            "original_query": query,
            "translated_query": query,
            "language": language,
            "translated": False,
        }

    translated_query = translate_to_english(query, language)

    return {
        "original_query": query,
        "translated_query": translated_query,
        "language": language,
        "translated": True,
    }