"""Constants for the kept MCE and OOD datasets."""

from pathlib import Path

MCE_TASKS = ["USPTO", "Symptom2Disease", "LawBench"]
MCE_DATA_PATH = str(Path(__file__).resolve().parent)
MCE_DATA_SPLITS = {
    "FiNER": {"train": 200, "val": 100, "test": 100},
    "USPTO": {"train": 50, "val": 30, "test": 100},
    "Symptom2Disease": {"train": 200, "val": 50, "test": 212},
    "LawBench": {"train": 200, "val": 50, "test": 100},
    "AEGIS": {"train": 400, "val": 128, "test": 140},
}

TRANSFER_TASKS = [
    "AGNews",
    "GoEmotions",
    "Banking77",
    "FinancialPhraseBank",
    "SciCite",
    "TweetEval_hate",
    "Amazon5",
    "SciTail",
]

AG_NEWS_LABELS = ["World", "Sports", "Business", "Sci/Tech"]
FINANCIAL_PHRASEBANK_LABELS = ["negative", "neutral", "positive"]
SCICITE_LABELS = ["background", "method", "result"]
TWEETEVAL_HATE_LABELS = ["non_hate", "hate"]
AMAZON5_LABELS = ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"]
SCITAIL_LABELS = ["entailment", "neutral"]
GOEMOTIONS_LABELS = [
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral",
]

SYMPTOM2DISEASE_LABELS = [
    "drug reaction",
    "allergy",
    "chicken pox",
    "diabetes",
    "psoriasis",
    "hypertension",
    "cervical spondylosis",
    "bronchial asthma",
    "varicose veins",
    "malaria",
    "dengue",
    "arthritis",
    "impetigo",
    "fungal infection",
    "common cold",
    "gastroesophageal reflux disease",
    "urinary tract infection",
    "typhoid",
    "pneumonia",
    "peptic ulcer disease",
    "jaundice",
    "migraine",
]



ALL_TASKS = MCE_TASKS + TRANSFER_TASKS
