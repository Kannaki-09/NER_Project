
import numpy as np
from datasets import Dataset
from seqeval.metrics import classification_report
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments, DataCollatorForTokenClassification
from seqeval.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import pipeline


# Step 1: data
data = [
    {"tokens": ["1", "masala", "vada", "with", "spicy"], "labels": ["B-QUANTITY", "B-FOOD", "I-FOOD", "O", "B-CUSTOMIZATION"]},
    {"tokens": ["2", "idlis", "with", "extra", "chutney"], "labels": ["B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["one", "pongal", "less", "oil"], "labels": ["B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["three", "vada", "no", "onion"], "labels": ["B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["plate", "of", "puri", "with", "chutney"], "labels": ["B-QUANTITY", "O", "B-FOOD", "O", "B-CUSTOMIZATION"]},
    {"tokens": ["half", "plate", "idli", "with", "chutney"], "labels": ["B-QUANTITY", "I-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION"]},
    {"tokens": ["two", "pooris", "no", "masala"], "labels": ["B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["3", "vada", "with", "less", "chutney"], "labels": ["B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["just", "coffee"], "labels": ["O", "B-FOOD"]},
    {"tokens": ["one", "filter", "coffee"], "labels": ["B-QUANTITY", "B-FOOD", "I-FOOD"]},
    {"tokens": ["get", "me", "2", "masala", "dosas"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "I-FOOD"]},
    {"tokens": ["i", "want", "hot", "pongal"], "labels": ["O", "O", "B-CUSTOMIZATION", "B-FOOD"]},
    {"tokens": ["4", "idlis", "and", "1", "vada"], "labels": ["B-QUANTITY", "B-FOOD", "O", "B-QUANTITY", "B-FOOD"]},
    {"tokens": ["single", "vada", "with", "no", "chutney"], "labels": ["B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["plate", "of", "samosa", "with", "sauce"], "labels": ["B-QUANTITY", "O", "B-FOOD", "O", "B-CUSTOMIZATION"]},
    {"tokens": ["one", "hot", "idli"], "labels": ["B-QUANTITY", "B-CUSTOMIZATION", "B-FOOD"]},
    {"tokens": ["no", "spicy", "in", "dosa"], "labels": ["B-CUSTOMIZATION", "I-CUSTOMIZATION", "O", "B-FOOD"]},
    {"tokens": ["extra", "sambar", "for", "pongal"], "labels": ["B-CUSTOMIZATION", "I-CUSTOMIZATION", "O", "B-FOOD"]},
    {"tokens": ["one", "pav", "bhaji"], "labels": ["B-QUANTITY", "B-FOOD", "I-FOOD"]},
    {"tokens": ["need", "strong", "tea"], "labels": ["O", "B-CUSTOMIZATION", "B-FOOD"]},
    {"tokens": ["3", "medu", "vada", "with", "sambar"], "labels": ["B-QUANTITY", "B-FOOD", "I-FOOD", "O", "B-CUSTOMIZATION"]},
    {"tokens": ["no", "oil", "in", "pongal"], "labels": ["B-CUSTOMIZATION", "I-CUSTOMIZATION", "O", "B-FOOD"]},
    {"tokens": ["give", "me", "1", "poori", "with", "less", "salt"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["hot", "coffee", "please"], "labels": ["B-CUSTOMIZATION", "B-FOOD", "O"]},
    {"tokens": ["order", "one", "idli", "with", "extra", "chutney"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["i", "want", "2", "hot", "idlis"], "labels": ["O", "O", "B-QUANTITY", "B-CUSTOMIZATION", "B-FOOD"]},
    {"tokens": ["just", "one", "vada", "with", "no", "chutney"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["get", "me", "three", "masala", "dosas", "less", "spicy"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["half", "plate", "pongal", "no", "ghee"], "labels": ["B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["want", "hot", "coffee"], "labels": ["O", "B-CUSTOMIZATION", "B-FOOD"]},
    {"tokens": ["please", "give", "me", "1", "samosa", "with", "extra", "chutney"], "labels": ["O", "O", "O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["two", "pooris", "without", "spice"], "labels": ["B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["bring", "filter", "coffee", "no", "sugar"], "labels": ["O", "B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["need", "one", "vada", "and", "idli"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-FOOD"]},
    {"tokens": ["a", "plate", "of", "pongal", "with", "less", "oil"], "labels": ["O", "B-QUANTITY", "O", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["two", "plates", "of", "idli", "with", "extra", "chutney"], "labels": ["B-QUANTITY", "I-QUANTITY", "O", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["give", "me", "a", "spicy", "masala", "dosa"], "labels": ["O", "O", "O", "B-CUSTOMIZATION", "B-FOOD", "I-FOOD"]},
    {"tokens": ["i", "want", "one", "vada", "without", "onion"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["bring", "half", "plate", "poori", "with", "extra", "sambar"], "labels": ["O", "B-QUANTITY", "I-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["need", "three", "idlis", "with", "no", "chutney"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["just", "a", "cup", "of", "strong", "coffee"], "labels": ["O", "O", "B-QUANTITY", "O", "B-CUSTOMIZATION", "B-FOOD"]},
    {"tokens": ["i", "will", "take", "two", "masala", "dosas"], "labels": ["O", "O", "O", "B-QUANTITY", "B-FOOD", "I-FOOD"]},
    {"tokens": ["get", "me", "a", "hot", "vada"], "labels": ["O", "O", "O", "B-CUSTOMIZATION", "B-FOOD"]},
    {"tokens": ["no", "oil", "in", "my", "pongal"], "labels": ["B-CUSTOMIZATION", "I-CUSTOMIZATION", "O", "O", "B-FOOD"]},
    {"tokens": ["can", "i", "have", "one", "poori", "without", "salt"], "labels": ["O", "O", "O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["two", "plates", "of", "idli", "with", "extra", "sambar"], "labels": ["B-QUANTITY", "I-QUANTITY", "O", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["one", "masala", "dosa", "with", "no", "onion"], "labels": ["B-QUANTITY", "B-FOOD", "I-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["half", "plate", "pongal", "no", "ghee"], "labels": ["B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["three", "vada", "with", "extra", "chutney"], "labels": ["B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["just", "a", "cup", "of", "tea"], "labels": ["O", "O", "B-QUANTITY", "O", "B-FOOD"]},
    {"tokens": ["get", "me", "strong", "filter", "coffee"], "labels": ["O", "O", "B-CUSTOMIZATION", "B-FOOD", "I-FOOD"]},
    {"tokens": ["two", "samosas", "no", "mint", "chutney"], "labels": ["B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["need", "one", "hot", "puri"], "labels": ["O", "B-QUANTITY", "B-CUSTOMIZATION", "B-FOOD"]},
    {"tokens": ["order", "4", "idlis", "with", "less", "spice"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["want", "cold", "coffee"], "labels": ["O", "B-CUSTOMIZATION", "B-FOOD"]},
    {"tokens": ["one", "plate", "medu", "vada", "with", "extra", "sambar"], "labels": ["B-QUANTITY", "I-QUANTITY", "B-FOOD", "I-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["three", "hot", "idlis", "no", "onion"], "labels": ["B-QUANTITY", "B-CUSTOMIZATION", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["half", "cup", "strong", "tea"], "labels": ["B-QUANTITY", "I-QUANTITY", "B-CUSTOMIZATION", "B-FOOD"]},
    {"tokens": ["two", "dosas", "with", "extra", "chutney"], "labels": ["B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["just", "coffee"], "labels": ["O", "B-FOOD"]},
    {"tokens": ["need", "one", "plate", "pongal", "less", "oil"], "labels": ["O", "B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "two", "pooris", "no", "spicy"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["four", "idlis", "with", "extra", "ghee"], "labels": ["B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["one", "hot", "masala", "dosa"], "labels": ["B-QUANTITY", "B-CUSTOMIZATION", "B-FOOD", "I-FOOD"]},
    {"tokens": ["a", "plate", "of", "samosa", "with", "sweet", "chutney"], "labels": ["O", "B-QUANTITY", "O", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["two", "plates", "of", "idli", "with", "extra", "sambar"], "labels": ["B-QUANTITY", "I-QUANTITY", "O", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["get", "me", "three", "dosas", "extra", "crispy"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["i", "want", "a", "bowl", "of", "pongal", "with", "ghee", "on", "top"], "labels": ["O", "O", "O", "B-QUANTITY", "O", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["half", "plate", "poori", "less", "masala"], "labels": ["B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["one", "plain", "dosa", "no", "butter"], "labels": ["B-QUANTITY", "B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["two", "idlis", "and", "a", "vada", "with", "extra", "chutney"], "labels": ["B-QUANTITY", "B-FOOD", "O", "O", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["give", "me", "a", "set", "dosa", "with", "no", "salt"], "labels": ["O", "O", "O", "B-FOOD", "I-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["full", "meals", "with", "extra", "gravy"], "labels": ["B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["small", "portion", "of", "upma", "no", "ginger"], "labels": ["B-QUANTITY", "I-QUANTITY", "O", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["a", "single", "vada", "without", "chutney"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["three", "onion", "dosas", "extra", "spicy"], "labels": ["B-QUANTITY", "B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["half", "plate", "poha", "no", "mustard", "seeds"], "labels": ["B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["a", "cup", "of", "coffee", "with", "less", "sugar"], "labels": ["O", "B-QUANTITY", "O", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["get", "me", "four", "idlis", "and", "no", "chutney"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["medium", "size", "masala", "dosa", "without", "ghee"], "labels": ["B-QUANTITY", "I-QUANTITY", "B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["i", "need", "a", "small", "portion", "of", "kesari"], "labels": ["O", "O", "O", "B-QUANTITY", "I-QUANTITY", "O", "B-FOOD"]},
    {"tokens": ["serve", "two", "vada", "pavs", "less", "spicy"], "labels": ["O", "B-QUANTITY", "B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["please", "get", "me", "three", "chapatis", "no", "oil"], "labels": ["O", "O", "O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["a", "bowl", "of", "curd", "rice", "extra", "curd"], "labels": ["O", "B-QUANTITY", "O", "B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["give", "me", "a", "double", "egg", "dosa", "with", "extra", "onions"], "labels": ["O", "O", "O", "B-QUANTITY", "B-FOOD", "I-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["just", "one", "samosa", "no", "mint", "chutney"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["a", "plate", "of", "lemon", "rice", "without", "nuts"], "labels": ["O", "B-QUANTITY", "O", "B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["two", "plain", "parathas", "with", "butter"], "labels": ["B-QUANTITY", "B-FOOD", "I-FOOD", "O", "B-CUSTOMIZATION"]},
    {"tokens": ["get", "me", "three", "vada", "no", "coconut", "chutney"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["a", "glass", "of", "lassi", "less", "sweet"], "labels": ["O", "B-QUANTITY", "O", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "four", "pani", "puris", "extra", "spicy"], "labels": ["O", "B-QUANTITY", "B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["i", "would", "like", "two", "biryani", "packets", "with", "less", "oil"], "labels": ["O", "O", "O", "B-QUANTITY", "B-FOOD", "I-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["a", "single", "sandwich", "no", "sauce"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["get", "half", "plate", "idiyappam", "extra", "coconut", "milk"], "labels": ["O", "B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["one", "idli", "with", "sambar"], "labels": ["B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION"]},
    {"tokens": ["two", "vada", "no", "chutney"], "labels": ["B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["three", "masala", "dosa", "extra", "spicy"], "labels": ["B-QUANTITY", "B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["a", "plate", "of", "pongal", "without", "cashews"], "labels": ["O", "B-QUANTITY", "O", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["give", "me", "one", "pav", "bhaji", "less", "butter"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["get", "four", "veg", "rolls", "extra", "cheese"], "labels": ["O", "B-QUANTITY", "B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["five", "jalebi", "with", "less", "sugar"], "labels": ["B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["a", "bowl", "of", "curd", "rice", "with", "no", "pickle"], "labels": ["O", "B-QUANTITY", "O", "B-FOOD", "I-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "one", "mango", "lassi", "not", "too", "sweet"], "labels": ["O", "B-QUANTITY", "B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["three", "plain", "roti", "no", "ghee"], "labels": ["B-QUANTITY", "B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["six", "chicken", "momoes", "extra", "dip"], "labels": ["B-QUANTITY", "B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["give", "me", "one", "veg", "thali", "without", "curd"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["get", "a", "half", "plate", "noodles", "spicy"], "labels": ["O", "O", "B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION"]},
    {"tokens": ["just", "one", "coffee", "less", "sugar"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["a", "glass", "of", "buttermilk", "no", "salt"], "labels": ["O", "B-QUANTITY", "O", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["seven", "paneer", "tikka", "extra", "spicy"], "labels": ["B-QUANTITY", "B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "two", "sandwiches", "without", "sauce"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["one", "bottle", "water", "cold"], "labels": ["B-QUANTITY", "B-FOOD", "I-FOOD", "B-CUSTOMIZATION"]},
    {"tokens": ["three", "pani", "puri", "no", "onions"], "labels": ["B-QUANTITY", "B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["a", "cup", "tea", "no", "milk"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["two", "plates", "of", "vada", "with", "extra", "sambar"], "labels": ["B-QUANTITY", "I-QUANTITY", "O", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["one", "idli", "no", "spice", "please"], "labels": ["B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION", "O"]},
    {"tokens": ["half", "dosa", "less", "oil"], "labels": ["B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["give", "me", "biryani", "without", "onions"], "labels": ["O", "O", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["three", "samosas", "extra", "crispy"], "labels": ["B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["one", "cup", "of", "tea", "with", "more", "sugar"], "labels": ["B-QUANTITY", "I-QUANTITY", "O", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "idli", "sambar", "combo"], "labels": ["O", "B-FOOD", "I-FOOD", "I-FOOD"]},
    {"tokens": ["plain", "dosa", "with", "no", "butter"], "labels": ["B-FOOD", "I-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["one", "masala", "dosa", "extra", "spicy"], "labels": ["B-QUANTITY", "B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["single", "vada", "with", "less", "salt"], "labels": ["B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["send", "me", "curd", "rice", "without", "mustard"], "labels": ["O", "O", "B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["give", "two", "veg", "rolls"], "labels": ["O", "B-QUANTITY", "B-FOOD", "I-FOOD"]},
    {"tokens": ["tea", "no", "milk"], "labels": ["B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "half", "plate", "pongal"], "labels": ["O", "B-QUANTITY", "I-QUANTITY", "B-FOOD"]},
    {"tokens": ["three", "vada", "no", "chilli"], "labels": ["B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["plain", "uttapam", "extra", "chutney"], "labels": ["B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["idli", "with", "no", "ghee"], "labels": ["B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["two", "masala", "dosa", "without", "potato"], "labels": ["B-QUANTITY", "B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["get", "me", "vada", "with", "less", "chilli"], "labels": ["O", "O", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["idli", "with", "extra", "coconut", "chutney"], "labels": ["B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["one", "plate", "idli", "with", "extra", "sambar"], "labels": ["B-QUANTITY", "I-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["three", "vada", "no", "onions"], "labels": ["B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["half", "plate", "pongal", "less", "ghee"], "labels": ["B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["masala", "dosa", "without", "spice"], "labels": ["B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["send", "two", "cups", "coffee", "extra", "hot"], "labels": ["O", "B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["get", "me", "plain", "dosa", "with", "no", "salt"], "labels": ["O", "O", "B-FOOD", "I-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["one", "tea", "with", "less", "sugar"], "labels": ["B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["four", "vada", "extra", "crispy"], "labels": ["B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "idli", "chutney", "with", "no", "coriander"], "labels": ["O", "B-FOOD", "I-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["bring", "five", "samosas", "without", "mint", "chutney"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["plain", "dosa", "add", "extra", "chutney"], "labels": ["B-FOOD", "I-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["give", "me", "two", "filter", "coffees"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "I-FOOD"]},
    {"tokens": ["idli", "with", "extra", "gunpowder"], "labels": ["B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["serve", "dosa", "with", "no", "chutney"], "labels": ["O", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["half", "cup", "coffee", "no", "sugar"], "labels": ["B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["get", "three", "idlis", "with", "extra", "sambar"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["tea", "without", "milk"], "labels": ["B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["bring", "me", "pongal", "extra", "pepper"], "labels": ["O", "O", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "vada", "less", "spicy"], "labels": ["O", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["single", "plate", "idli", "without", "chutney"], "labels": ["B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["send", "two", "masala", "dosa", "no", "onions"], "labels": ["O", "B-QUANTITY", "B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["give", "me", "three", "vada", "less", "oil"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["one", "coffee", "no", "milk"], "labels": ["B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["get", "a", "dosa", "extra", "crisp"], "labels": ["O", "O", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["two", "plates", "idli", "extra", "chutney"], "labels": ["B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["filter", "coffee", "with", "more", "sugar"], "labels": ["B-FOOD", "I-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["send", "half", "plate", "pongal"], "labels": ["O", "B-QUANTITY", "I-QUANTITY", "B-FOOD"]},
    {"tokens": ["three", "vada", "without", "spice"], "labels": ["B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "idli", "less", "salt"], "labels": ["O", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["bring", "dosa", "extra", "butter"], "labels": ["O", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["tea", "with", "less", "sugar"], "labels": ["B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["idli", "with", "more", "chutney"], "labels": ["B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["get", "me", "dosa", "without", "ghee"], "labels": ["O", "O", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["serve", "four", "vada", "with", "less", "spice"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["one", "cup", "coffee", "with", "no", "sugar"], "labels": ["B-QUANTITY", "I-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["dosa", "with", "extra", "chutney", "and", "sambar"], "labels": ["B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION", "I-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["give", "me", "plain", "idli"], "labels": ["O", "O", "B-FOOD", "I-FOOD"]},
    {"tokens": ["two", "vada", "without", "chutney"], "labels": ["B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["filter", "coffee", "no", "sugar"], "labels": ["B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "three", "idli", "with", "extra", "chutney"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "three", "idli", "with", "extra", "chutney"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["get", "me", "2", "masala", "dosa", "without", "onion"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["i", "need", "one", "plate", "pongal", "no", "ghee"], "labels": ["O", "O", "B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["send", "five", "vada", "with", "spicy", "chutney"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["give", "me", "idli", "and", "more", "sambar"], "labels": ["O", "O", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["can", "i", "have", "a", "crispy", "dosa"], "labels": ["O", "O", "O", "O", "B-CUSTOMIZATION", "B-FOOD"]},
    {"tokens": ["please", "add", "two", "plates", "of", "poori"], "labels": ["O", "O", "B-QUANTITY", "I-QUANTITY", "O", "B-FOOD"]},
    {"tokens": ["one", "pongal", "with", "no", "onion"], "labels": ["B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["extra", "chutneys", "and", "idli"], "labels": ["B-CUSTOMIZATION", "I-CUSTOMIZATION", "O", "B-FOOD"]},
    {"tokens": ["make", "that", "2", "crispy", "dosa"], "labels": ["O", "O", "B-QUANTITY", "B-CUSTOMIZATION", "B-FOOD"]},
    {"tokens": ["i", "want", "a", "large", "plate", "of", "pongal"], "labels": ["O", "O", "O", "B-QUANTITY", "I-QUANTITY", "O", "B-FOOD"]},
    {"tokens": ["one", "vada", "no", "chilli"], "labels": ["B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["can", "you", "send", "idli", "x", "3", "with", "less", "oil"], "labels": ["O", "O", "O", "B-FOOD", "B-QUANTITY", "I-QUANTITY", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["give", "me", "extra", "sambar", "for", "pongal"], "labels": ["O", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION", "O", "B-FOOD"]},
    {"tokens": ["please", "include", "2", "idlis", "and", "chutney"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "O", "B-FOOD"]},
    {"tokens": ["need", "four", "vada", "no", "ghee", "extra", "crispy"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "1", "masala", "dosa", "with", "extra", "oil"], "labels": ["O", "B-QUANTITY", "B-FOOD", "I-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["send", "me", "some", "hot", "pongal", "and", "idli"], "labels": ["O", "O", "B-QUANTITY", "B-CUSTOMIZATION", "B-FOOD", "O", "B-FOOD"]},
    {"tokens": ["i", "want", "idli", "without", "spicy", "chutney"], "labels": ["O", "O", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["give", "me", "3", "idlis", "and", "2", "vada"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "O", "B-QUANTITY", "B-FOOD"]},
    {"tokens": ["please", "send", "2", "plates", "of", "idli", "with", "extra", "sambar"], "labels": ["O", "O", "B-QUANTITY", "I-QUANTITY", "O", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["I", "want", "one", "masala", "dosa", "no", "onion"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["give", "me", "3", "idlis", "and", "less", "oil"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "one", "vada", "with", "extra", "chutney"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["two", "plates", "of", "pongal", "without", "ghee"], "labels": ["B-QUANTITY", "I-QUANTITY", "O", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["need", "1", "dosa", "with", "spicy", "chutney"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["please", "send", "4", "idlis", "extra", "butter"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["can", "I", "get", "one", "plate", "of", "poori", "with", "less", "oil"], "labels": ["O", "O", "O", "B-QUANTITY", "I-QUANTITY", "O", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["give", "me", "two", "vadais", "extra", "spicy"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "3", "plates", "idli", "with", "extra", "chutney"], "labels": ["O", "B-QUANTITY", "I-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["one", "masala", "dosa", "no", "ghee"], "labels": ["B-QUANTITY", "B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["two", "idlis", "with", "less", "oil"], "labels": ["B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["please", "send", "one", "plate", "pongal", "extra", "chutney"], "labels": ["O", "O", "B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["want", "2", "dosas", "no", "onion"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "one", "vada", "with", "extra", "butter"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["3", "idlis", "less", "oil"], "labels": ["B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["two", "plates", "of", "masala", "dosa"], "labels": ["B-QUANTITY", "I-QUANTITY", "O", "B-FOOD", "I-FOOD"]},
    {"tokens": ["one", "plate", "idli", "extra", "sambar"], "labels": ["B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["give", "me", "4", "dosa", "with", "less", "oil"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["I", "want", "one", "vada", "no", "ghee"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["please", "send", "2", "idlis", "extra", "chutney"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "one", "plate", "poori", "less", "oil"], "labels": ["O", "B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["can", "I", "get", "3", "idlis", "with", "extra", "butter"], "labels": ["O", "O", "O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["give", "me", "one", "plate", "pongal", "no", "onion"], "labels": ["O", "O", "B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["need", "two", "dosas", "extra", "spicy"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "3", "idlis", "with", "extra", "chutney"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["one", "masala", "dosa", "no", "ghee"], "labels": ["B-QUANTITY", "B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["two", "idlis", "with", "less", "oil"], "labels": ["B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["please", "send", "one", "plate", "pongal", "extra", "chutney"], "labels": ["O", "O", "B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["want", "2", "dosas", "no", "onion"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},{"tokens": ["order", "two", "plates", "of", "idli", "with", "extra", "chutney"], "labels": ["O", "B-QUANTITY", "I-QUANTITY", "O", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["please", "get", "one", "masala", "dosa", "no", "ghee"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["I", "want", "three", "idlis", "with", "less", "oil"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["give", "me", "one", "vada", "extra", "spicy"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "four", "plates", "pongal", "without", "onion"], "labels": ["O", "B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["need", "1", "dosa", "with", "extra", "butter"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["send", "two", "idlis", "less", "chutney"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["one", "plate", "poori", "extra", "spicy"], "labels": ["B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["give", "me", "three", "vadais", "with", "no", "ghee"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "1", "plate", "masala", "dosa", "without", "onion"], "labels": ["O", "B-QUANTITY", "I-QUANTITY", "B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["two", "idlis", "extra", "chutney"], "labels": ["B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["please", "send", "one", "plate", "pongal", "less", "oil"], "labels": ["O", "O", "B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["want", "two", "dosas", "extra", "butter"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "3", "idlis", "with", "extra", "sambar"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["one", "masala", "dosa", "no", "ghee"], "labels": ["B-QUANTITY", "B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["two", "idlis", "with", "less", "oil"], "labels": ["B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["please", "send", "one", "plate", "pongal", "extra", "chutney"], "labels": ["O", "O", "B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["need", "two", "vada", "no", "onion"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "one", "plate", "poori", "with", "extra", "butter"], "labels": ["O", "B-QUANTITY", "I-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["can", "I", "get", "three", "idlis", "with", "less", "oil"], "labels": ["O", "O", "O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["give", "me", "one", "plate", "pongal", "no", "onion"], "labels": ["O", "O", "B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["need", "two", "dosa", "extra", "spicy"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "3", "idlis", "with", "extra", "chutney"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["one", "masala", "dosa", "no", "ghee"], "labels": ["B-QUANTITY", "B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["two", "idlis", "with", "less", "oil"], "labels": ["B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["please", "send", "one", "plate", "pongal", "extra", "chutney"], "labels": ["O", "O", "B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["want", "2", "dosas", "no", "onion"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "one", "plate", "idli", "with", "extra", "sambar"], "labels": ["O", "B-QUANTITY", "I-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["three", "idlis", "with", "extra", "chutney"], "labels": ["B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["need", "one", "vada", "with", "no", "ghee"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["give", "me", "two", "plates", "masala", "dosa", "no", "ghee"], "labels": ["O", "O", "B-QUANTITY", "I-QUANTITY", "B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["one", "plate", "pongal", "with", "extra", "chutney"], "labels": ["B-QUANTITY", "I-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "two", "idlis", "less", "oil"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["need", "one", "vada", "no", "onion"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["please", "send", "three", "plates", "poori", "extra", "spicy"], "labels": ["O", "O", "B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["give", "me", "one", "masala", "dosa", "extra", "butter"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "two", "idlis", "with", "extra", "sambar"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["I", "want", "one", "plate", "vada", "less", "oil"], "labels": ["O", "O", "B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["send", "three", "dosas", "no", "ghee"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["need", "one", "plate", "idli", "extra", "chutney"], "labels": ["O", "B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "four", "idlis", "with", "less", "oil"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["please", "send", "one", "vada", "extra", "spicy"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["give", "me", "two", "plates", "poori", "no", "onion"], "labels": ["O", "O", "B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["I", "want", "one", "masala", "dosa", "with", "extra", "butter"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "I-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "two", "idlis", "less", "oil"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["need", "one", "plate", "vada", "no", "ghee"], "labels": ["O", "B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["please", "send", "three", "dosas", "extra", "chutney"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["give", "me", "one", "plate", "idli", "with", "less", "oil"], "labels": ["O", "O", "B-QUANTITY", "I-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "two", "plates", "pongal", "extra", "spicy"], "labels": ["O", "B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["I", "want", "three", "idlis", "no", "ghee"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["can", "I", "have", "two", "masala", "dosas", "with", "extra", "butter"], "labels": ["O", "O", "O", "B-QUANTITY", "B-FOOD", "I-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["please", "send", "one", "plate", "idli", "no", "ghee"], "labels": ["O", "O", "B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "three", "plates", "of", "vada", "with", "less", "oil"], "labels": ["O", "B-QUANTITY", "I-QUANTITY", "O", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["I", "want", "one", "plate", "pongal", "extra", "spicy"], "labels": ["O", "O", "B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["give", "me", "two", "idlis", "with", "extra", "chutney"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["need", "one", "plate", "poori", "less", "oil"], "labels": ["O", "B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "four", "dosas", "with", "extra", "butter"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["one", "plate", "idli", "no", "onion"], "labels": ["B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["send", "three", "vadais", "extra", "spicy"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["give", "me", "two", "plates", "masala", "dosa", "no", "ghee"], "labels": ["O", "O", "B-QUANTITY", "I-QUANTITY", "B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["add", "extra", "sambar", "to", "my", "idli"], "labels": ["O", "B-CUSTOMIZATION", "I-CUSTOMIZATION", "O", "O", "B-FOOD"]},
    {"tokens": ["I", "need", "two", "pooris", "without", "onion"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["give", "three", "vada", "with", "extra", "spicy", "chutney"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["send", "1", "plate", "pongal", "no", "ghee"], "labels": ["O", "B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "four", "idlis", "extra", "chutney"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["please", "add", "less", "oil", "to", "vada"], "labels": ["O", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION", "O", "B-FOOD"]},
    {"tokens": ["I", "want", "one", "masala", "dosa", "with", "no", "onion"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "I-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["give", "me", "extra", "crispy", "poori"], "labels": ["O", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION", "B-FOOD"]},
    {"tokens": ["two", "idli", "no", "salt"], "labels": ["B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["get", "one", "plate", "pongal", "with", "extra", "ghee"], "labels": ["O", "B-QUANTITY", "I-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["bring", "three", "vadais", "with", "no", "spice"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["can", "I", "get", "two", "dosa", "with", "less", "oil"], "labels": ["O", "O", "O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["one", "plate", "pongal", "with", "extra", "chutney"], "labels": ["B-QUANTITY", "I-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["just", "idli", "with", "no", "onion"], "labels": ["O", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["send", "three", "masala", "dosa", "extra", "ghee"], "labels": ["O", "B-QUANTITY", "B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["bring", "two", "idlis", "with", "less", "spice"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["I", "need", "vada", "with", "extra", "sambar"], "labels": ["O", "O", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "three", "plates", "of", "pongal", "no", "oil"], "labels": ["O", "B-QUANTITY", "I-QUANTITY", "O", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["give", "me", "one", "idli", "without", "onion"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["please", "send", "two", "poori", "extra", "crispy"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["need", "one", "idli", "with", "more", "sambar"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["give", "two", "dosa", "without", "ghee"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "three", "vada", "with", "extra", "spice"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["get", "one", "plate", "idli", "no", "onion"], "labels": ["O", "B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["send", "four", "idlis", "extra", "chutney"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["give", "me", "masala", "dosa", "with", "no", "oil"], "labels": ["O", "O", "B-FOOD", "I-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "one", "vada", "extra", "crispy"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["want", "idli", "with", "less", "ghee"], "labels": ["O", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["one", "plate", "poori", "no", "oil"], "labels": ["B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["send", "me", "two", "vada", "with", "extra", "chutney"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["I", "need", "three", "idlis", "no", "onion"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "one", "pongal", "extra", "spice"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["add", "less", "oil", "to", "my", "dosa"], "labels": ["O", "B-CUSTOMIZATION", "I-CUSTOMIZATION", "O", "O", "B-FOOD"]},
    {"tokens": ["give", "four", "idlis", "without", "ghee"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["send", "me", "three", "plates", "of", "vada", "with", "extra", "chutney"], "labels": ["O", "O", "B-QUANTITY", "I-QUANTITY", "O", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["get", "idli", "with", "no", "onion"], "labels": ["O", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["bring", "one", "poori", "extra", "crispy"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["send", "two", "plates", "pongal", "no", "ghee"], "labels": ["O", "B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["add", "extra", "chutney", "to", "vada"], "labels": ["O", "B-CUSTOMIZATION", "I-CUSTOMIZATION", "O", "B-FOOD"]},
    {"tokens": ["I", "want", "masala", "dosa", "with", "extra", "butter"], "labels": ["O", "O", "B-FOOD", "I-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "four", "poori", "without", "spice"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["send", "three", "idlis", "with", "no", "salt"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["one", "plate", "vada", "extra", "chutney"], "labels": ["B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["get", "two", "masala", "dosa", "less", "oil"], "labels": ["O", "B-QUANTITY", "B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["bring", "idli", "no", "onion"], "labels": ["O", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "one", "plate", "pongal", "extra", "ghee"], "labels": ["O", "B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["send", "me", "three", "poori", "extra", "crispy"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["give", "two", "vada", "without", "oil"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "2", "idly", "with", "extra", "chutni"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["plz", "send", "1", "masala", "dosa", "no", "ghee"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["want", "3", "pooriis", "less", "oil", "pls"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION", "O"]},
    {"tokens": ["can", "i", "have", "extra", "chutney", "and", "2", "idlis"], "labels": ["O", "O", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION", "O", "B-QUANTITY", "B-FOOD"]},
    {"tokens": ["send", "me", "one", "plate", "of", "masala", "dosaa"], "labels": ["O", "O", "B-QUANTITY", "I-QUANTITY", "O", "B-FOOD", "I-FOOD"]},
    {"tokens": ["i", "need", "small", "vada", "with", "extra", "spicy", "chutni"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "2", "dossas", "less", "oil"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["please", "send", "one", "idly", "with", "no", "onion"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["give", "me", "3", "plates", "pooris", "with", "extra", "sabji"], "labels": ["O", "O", "B-QUANTITY", "I-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["need", "2", "idly", "with", "more", "chutni"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "one", "dosa", "no", "ghee", "extra", "sambar"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["send", "3", "idlis", "less", "spicy"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["plz", "give", "1", "plate", "pongal", "with", "extra", "chutni"], "labels": ["O", "O", "B-QUANTITY", "I-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["i", "want", "2", "dossas", "no", "onion"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["give", "me", "small", "vada", "with", "extra", "chutney"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "one", "plate", "idly", "and", "extra", "ghee"], "labels": ["O", "B-QUANTITY", "I-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["need", "3", "poori", "with", "less", "oil"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["send", "2", "idlys", "with", "extra", "spicy", "chutni"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["plz", "order", "one", "plate", "masala", "dosa", "no", "ghee"], "labels": ["O", "O", "B-QUANTITY", "I-QUANTITY", "B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["want", "extra", "chutney", "and", "2", "idly"], "labels": ["O", "B-CUSTOMIZATION", "I-CUSTOMIZATION", "O", "B-QUANTITY", "B-FOOD"]},
    {"tokens": ["give", "me", "3", "plates", "dosa", "with", "less", "oil"], "labels": ["O", "O", "B-QUANTITY", "I-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "one", "poori", "extra", "spicy"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["need", "2", "idli", "with", "extra", "sambar"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["send", "me", "one", "plate", "vada", "with", "no", "ghee"], "labels": ["O", "O", "B-QUANTITY", "I-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "3", "idlis", "less", "spicy"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["plz", "give", "one", "plate", "pongal", "extra", "ghee"], "labels": ["O", "O", "B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["i", "want", "2", "dosa", "with", "extra", "chutney"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["give", "me", "small", "vada", "with", "no", "onion"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "one", "idly", "with", "extra", "chutni"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["need", "3", "poori", "less", "oil"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["plz", "send", "2", "masala", "dosas", "wit", "extra", "cheese"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "I-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["can", "i", "have", "1", "idly", "with", "less", "spice"], "labels": ["O", "O", "O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["give", "me", "3", "plate", "poori", "no", "onion"], "labels": ["O", "O", "B-QUANTITY", "O", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "4", "vada", "wth", "extra", "sambar"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["1", "dosa", "no", "ghee", "pls"], "labels": ["B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION", "O"]},
    {"tokens": ["want", "2", "idli", "with", "less", "oil"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["send", "me", "a", "smal", "vada", "no", "chilly"], "labels": ["O", "O", "O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["pls", "add", "3", "masala", "dosa", "extra", "chutni"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["1plate", "idly", "with", "no", "ginger"], "labels": ["B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["i", "need", "4", "dosaa", "less", "salt"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["two", "idlis", "and", "extra", "sambar"], "labels": ["B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["please", "send", "1", "poori", "with", "extra", "oil"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["3", "plates", "vada", "less", "spicy"], "labels": ["B-QUANTITY", "O", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["want", "one", "masaladosa", "with", "extra", "cheese"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["1", "idly", "with", "extra", "ghee"], "labels": ["B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "2", "dossas", "no", "onion"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["3", "idlies", "with", "little", "spicy"], "labels": ["B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["send", "1", "plate", "poory", "extra", "chutni"], "labels": ["O", "B-QUANTITY", "O", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["2", "vadais", "with", "less", "oil"], "labels": ["B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["i", "want", "3", "masala", "dossas", "pls"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "I-FOOD", "O"]},
    {"tokens": ["extra", "chutney", "and", "2", "idli"], "labels": ["B-CUSTOMIZATION", "I-CUSTOMIZATION", "O", "B-QUANTITY", "B-FOOD"]},
    {"tokens": ["1", "plate", "poori", "with", "less", "salt"], "labels": ["B-QUANTITY", "O", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["send", "me", "2", "small", "idlis"], "labels": ["O", "O", "B-QUANTITY", "B-QUANTITY", "B-FOOD"]},
    {"tokens": ["want", "one", "masala", "dos", "with", "extra", "butter"], "labels": ["O", "B-QUANTITY", "B-FOOD", "I-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["pls", "send", "3", "idli", "with", "no", "ghee"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["2", "dosa", "extra", "spicy"], "labels": ["B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["1", "vada", "with", "less", "oil"], "labels": ["B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "3", "plates", "idlis", "with", "extra", "sambar"], "labels": ["O", "B-QUANTITY", "O", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["pls", "send", "1", "masala", "dosa", "without", "onion"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["give", "me", "2", "idli", "extra", "chutney"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["1", "plate", "poori", "less", "oil"], "labels": ["B-QUANTITY", "O", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["want", "3", "dosa", "with", "extra", "ghee"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["2", "idli", "no", "onion"], "labels": ["B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["pls", "add", "1", "vada", "with", "less", "spicy"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["need", "3", "dosa", "extra", "butter"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["send", "2", "idli", "with", "extra", "sambar"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["pls", "send", "one", "poori", "with", "less", "oil"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "1", "plate", "vada", "extra", "spicy"], "labels": ["O", "B-QUANTITY", "O", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["want", "2", "idlies", "no", "ghee"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["give", "me", "3", "masala", "dosa", "extra", "chutney"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["get", "me", "2", "idlis", "no", "onion"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["1", "plate", "dosa", "less", "spicy"], "labels": ["B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["three", "vada", "extra", "chutni"], "labels": ["B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["give", "pongal", "with", "no", "ghee"], "labels": ["O", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["i", "want", "masla", "dosa", "no", "oil"], "labels": ["O", "O", "B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["send", "4", "idlies", "extra", "chutney", "pls"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION", "O"]},
    {"tokens": ["get", "me", "poori", "without", "onion"], "labels": ["O", "O", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["just", "one", "plate", "pongal"], "labels": ["O", "B-QUANTITY", "I-QUANTITY", "B-FOOD"]},
    {"tokens": ["2", "dosas", "less", "oil", "more", "crispy"], "labels": ["B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["pls", "1", "idli", "no", "ghee"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["one", "vada", "extra", "sambar"], "labels": ["B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["bring", "3", "idly", "wit", "no", "oil"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["want", "pongal", "with", "ghee"], "labels": ["O", "B-FOOD", "O", "B-CUSTOMIZATION"]},
    {"tokens": ["order", "4", "idlis", "wit", "spicy", "chutney"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["i", "need", "two", "plates", "of", "idli"], "labels": ["O", "O", "B-QUANTITY", "I-QUANTITY", "O", "B-FOOD"]},
    {"tokens": ["get", "me", "dosa", "with", "extra", "sambhar"], "labels": ["O", "O", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["please", "send", "3", "masala", "dosa", "no", "onions"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["4", "idlis", "no", "ghee"], "labels": ["B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["i", "want", "2", "pooris", "less", "oil"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["dosa", "without", "spice"], "labels": ["B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["1", "poori", "add", "extra", "chutney"], "labels": ["B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["get", "me", "small", "idli"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD"]},
    {"tokens": ["add", "chutni", "for", "my", "dosa"], "labels": ["O", "B-CUSTOMIZATION", "O", "O", "B-FOOD"]},
    {"tokens": ["need", "5", "idlis", "extra", "onion"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["serve", "masala", "dosa", "no", "spicy"], "labels": ["O", "B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["bring", "2", "idlis", "wit", "sambhar"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION"]},
    {"tokens": ["get", "me", "idli", "less", "salt"], "labels": ["O", "O", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["two", "vada", "no", "spice"], "labels": ["B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["1", "masala", "dosaa", "xtra", "chutney"], "labels": ["B-QUANTITY", "B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["need", "idli", "with", "onion"], "labels": ["O", "B-FOOD", "O", "B-CUSTOMIZATION"]},
    {"tokens": ["i", "want", "5", "poori", "no", "oil", "please"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION", "O"]},
    {"tokens": ["send", "me", "3", "idlis", "extra", "sambar"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "1", "plate", "pongal"], "labels": ["O", "B-QUANTITY", "I-QUANTITY", "B-FOOD"]},
    {"tokens": ["give", "one", "dosa", "wit", "chutney"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION"]},
    {"tokens": ["make", "dosa", "extra", "crisp"], "labels": ["O", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["add", "sambar", "on", "top"], "labels": ["B-CUSTOMIZATION", "I-CUSTOMIZATION", "O", "O"]},
    {"tokens": ["send", "me", "big", "idli"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD"]},
    {"tokens": ["one", "idli", "with", "xtra", "chutni"], "labels": ["B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["3", "poori", "extra", "ghee"], "labels": ["B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["need", "dosaa", "no", "salt"], "labels": ["O", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["i", "want", "four", "idlis"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD"]},
    {"tokens": ["order", "2", "vada", "with", "extra", "onion"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["give", "pongal", "with", "chutney"], "labels": ["O", "B-FOOD", "O", "B-CUSTOMIZATION"]},
    {"tokens": ["please", "add", "chutney", "extra"], "labels": ["O", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["want", "idli", "extra", "crisp"], "labels": ["O", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["2", "plate", "idlis", "no", "spicy"], "labels": ["B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["order", "pongal", "witout", "onion"], "labels": ["O", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["give", "1", "dosa", "less", "ghee"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["send", "one", "idli", "no", "spicy"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["1", "idly", "no", "onion", "pls"], "labels": ["B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION", "O"]},
    {"tokens": ["pls", "giv", "me", "3", "idlies", "and", "xtra", "chuttney"], "labels": ["O", "O", "O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["i", "want", "1", "dosa", "widout", "oil"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["2", "plate", "poori", "no", "ghee", "pls"], "labels": ["B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION", "O"]},
    {"tokens": ["send", "one", "pongal", "no", "onion"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["xtra", "crispy", "dosa", "plz"], "labels": ["B-CUSTOMIZATION", "I-CUSTOMIZATION", "B-FOOD", "O"]},
    {"tokens": ["i", "need", "4", "idli", "with", "less", "oil"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["masal", "dos", "1", "plate", "no", "ghee"], "labels": ["B-FOOD", "I-FOOD", "B-QUANTITY", "I-QUANTITY", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["1", "plate", "vada", "no", "coconut", "chutney"], "labels": ["B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["need", "3x", "idlis", "wid", "spicy", "chutny"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["give", "1", "pongal", "n", "2", "vada"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-QUANTITY", "B-FOOD"]},
    {"tokens": ["order", "1", "idli", "wit", "xtra", "sambhar"], "labels": ["O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["pls", "add", "2", "idlis", "with", "no", "ghee"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["i", "take", "3", "idly", "more", "crispy"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["hey", "1", "masala", "dosa", "no", "onion"], "labels": ["O", "B-QUANTITY", "B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["one", "plate", "pongal", "witout", "ghee"], "labels": ["B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["add", "extra", "chutny", "to", "my", "1", "idli"], "labels": ["O", "B-CUSTOMIZATION", "I-CUSTOMIZATION", "O", "O", "B-QUANTITY", "B-FOOD"]},
    {"tokens": ["2", "pooris", "less", "oil"], "labels": ["B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["need", "dosa", "more", "crispy", "n", "spicy"], "labels": ["O", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION", "O", "I-CUSTOMIZATION"]},
    {"tokens": ["xtra", "sambhar", "and", "no", "onion"], "labels": ["B-CUSTOMIZATION", "I-CUSTOMIZATION", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["one", "vada", "plz"], "labels": ["B-QUANTITY", "B-FOOD", "O"]},
    {"tokens": ["get", "me", "2", "idlis", "with", "no", "ghee"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["chutney", "extra", "pls", "and", "3", "idlis"], "labels": ["B-CUSTOMIZATION", "I-CUSTOMIZATION", "O", "O", "B-QUANTITY", "B-FOOD"]},
    {"tokens": ["order", "dosa", "x1", "widout", "spicy", "chutney"], "labels": ["O", "B-FOOD", "B-QUANTITY", "B-CUSTOMIZATION", "I-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["give", "2x", "vada", "more", "crisp"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["send", "4", "idlis", "no", "chutney"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["i", "want", "2", "poori", "with", "xtra", "sambhar"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["3", "pongal", "no", "onion", "n", "ghee"], "labels": ["B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION", "O", "I-CUSTOMIZATION"]},
    {"tokens": ["want", "masala", "dosa", "no", "butter"], "labels": ["O", "B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["add", "idli", "x3", "extra", "chutney"], "labels": ["O", "B-FOOD", "B-QUANTITY", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["need", "1", "plain", "dosa", "no", "oil"], "labels": ["O", "B-QUANTITY", "B-FOOD", "I-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["just", "1", "idly", "xtra", "chuttni", "pls"], "labels": ["O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION", "O"]},
    {"tokens": ["give", "me", "dosa", "less", "ghee"], "labels": ["O", "O", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["2", "plate", "pongal", "no", "onion"], "labels": ["B-QUANTITY", "I-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["1", "poori", "less", "oil", "pls"], "labels": ["B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION", "O"]},
    {"tokens": ["xtra", "hot", "chutny", "with", "my", "idli"], "labels": ["B-CUSTOMIZATION", "I-CUSTOMIZATION", "I-CUSTOMIZATION", "O", "O", "B-FOOD"]},
    {"tokens": ["pls", "send", "4", "idlies", "n", "vada"], "labels": ["O", "O", "B-QUANTITY", "B-FOOD", "O", "B-FOOD"]},
    {"tokens": ["idli", "x2", "wit", "no", "coconut"], "labels": ["B-FOOD", "B-QUANTITY", "O", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["one", "idli", "witout", "oil"], "labels": ["B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    {"tokens": ["give", "extra", "sambhar", "and", "1", "dosa"], "labels": ["O", "B-CUSTOMIZATION", "I-CUSTOMIZATION", "O", "B-QUANTITY", "B-FOOD"]},
    {"tokens": ["can", "i", "get", "3x", "idlis", "xtra", "crispy"], "labels": ["O", "O", "O", "B-QUANTITY", "B-FOOD", "B-CUSTOMIZATION", "I-CUSTOMIZATION"]},
    
    
]

#Dataset Format
dataset = Dataset.from_list(data)
print("✅ Dataset created.")

# Step 2: Define label maps
label_list = ['O', 'B-FOOD', 'I-FOOD', 'B-QUANTITY', 'I-QUANTITY', 'B-CUSTOMIZATION', 'I-CUSTOMIZATION']
label_to_id = {label: idx for idx, label in enumerate(label_list)}
id_to_label = {idx: label for label, idx in label_to_id.items()}

# Step 3: Convert labels to IDs
def convert_labels_to_ids(example):
    # Only convert if label is string; else leave as is
    new_labels = []
    for label in example["labels"]:
        if isinstance(label, str):
            new_labels.append(label_to_id[label])
        else:
            # Already converted label, just keep
            new_labels.append(label)
    example["labels"] = new_labels
    return example


dataset = dataset.map(convert_labels_to_ids)
print("✅ Labels converted to IDs.")

# Step 4: Load tokenizer and model
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=len(label_list),
    id2label=id_to_label,
    label2id=label_to_id
)
print("✅ Tokenizer and model loaded.")

# Step 5: Tokenize and align labels
def tokenize_and_align_labels(example):
    tokenized = tokenizer(
        example["tokens"],
        is_split_into_words=True,
        truncation=True,
        padding="max_length"
    )
    word_ids = tokenized.word_ids()

    new_labels = []
    previous_word_idx = None
    for word_idx in word_ids:
        if word_idx is None:
            new_labels.append(-100)  # Special tokens get -100
        elif word_idx != previous_word_idx:
            # Label for the first token of a word
            new_labels.append(example["labels"][word_idx])
        else:
            # Label for subsequent tokens of the same word: ig no explain  andred in loss
            new_labels.append(-100)
        previous_word_idx = word_idx

    tokenized["labels"] = new_labels
    return tokenized

tokenized_dataset = dataset.map(tokenize_and_align_labels, remove_columns=["tokens", "labels"])

print("✅ Tokenization and label alignment completed.\n")
print(tokenized_dataset.column_names)

# Step 6: View first tokenized sample
print("Example:")
print(tokenized_dataset[0]) 

#step 7: Dataset Splitting
train_testvalid = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = train_testvalid["train"]
valid_dataset = train_testvalid["test"]

# Step 7: Data collator
data_collator = DataCollatorForTokenClassification(tokenizer)

#Custom tokens
custom_tokens = ["idlies", "idly", "idlis", "dossas","chutney","chutni" ,"dosa", "pongal", "sambar","xtra" , "poori", "vada", "1qty","chutneys","pls"]
tokenizer.add_tokens(custom_tokens)
model.resize_token_embeddings(len(tokenizer))
for token in custom_tokens:
    print(f"{token}: {tokenizer.tokenize(token)}")

#Classification Report 
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=2)
    labels = p.label_ids

    true_labels = [
        [id_to_label[label] for label in label_seq if label != -100]
        for label_seq in labels
    ]
    true_preds = [
        [id_to_label[pred] for pred, label in zip(pred_seq, label_seq) if label != -100]
        for pred_seq, label_seq in zip(preds, labels)
    ]

    all_preds = [label for seq in true_preds for label in seq]
    all_labels = [label for seq in true_labels for label in seq]

    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro")

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


#step 8: Trainer and TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=3,
    per_device_eval_batch_size=3,
    num_train_epochs=10,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    report_to="none",  # disable WandB if not using
    load_best_model_at_end=True,
    fp16=True,  # enable mixed precision training if GPU supports
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

## Step 9: Train
trainer.train()

#Save the Model
trainer.save_model("./ner-bert-model")

#Run Evaluation with the Trainer
metrics = trainer.evaluate()
print(metrics)
