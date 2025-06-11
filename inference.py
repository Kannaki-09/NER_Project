
from transformers import pipeline

ner_pipeline = pipeline("token-classification", model="./ner-bert-model", tokenizer="./ner-bert-model", aggregation_strategy="simple")

# Try a sentence
test_sentence = [
    "can i get 2 plates of idly with some xtra chutney pls",
    "1 dosa witout onion and lil spicy chutney pls",
    "Need one Pongal no ghee extra sambar ok?",
    "Give me 3 idlis, and don’t forget chutney — lots of it!",
    "plz send dosa 1 qty less oil more crispy",
    "I want a masala dosa NO GHEE and a small vada",
    "extra chutneys and 2 idlies",
    "1plate poori with less oil nd more sabji",
    "can u add xtra chutni to my order of idli x3?",
    "Hey, one pongal (no onion) and idli — total 2 plates"
    "Give me one idli with extra chutney"
]

results = ner_pipeline(test_sentence)

for sentence in test_sentence:
    print(f"\nInput: {sentence}")
    entities = ner_pipeline(sentence)
    for entity in entities:
        print(f"  {entity['word']}: {entity['entity_group']} ({entity['score']:.2f})")

