1. Introduction

This project builds a Named Entity Recognition (NER) model tailored for food ordering conversations. It extracts key entities like:

FOOD (e.g., dosa, idli, pongal)

QUANTITY (e.g., 1 plate, two)

CUSTOMIZATION (e.g., extra chutney, no ghee, spicy)

The model is fine-tuned on a BERT-based architecture and aims to improve natural language understanding in food ordering systems like chatbots or voice assistants.

2. Follow these steps to get your environment ready and train the model.

   2.1 INSTALLATION
        pip install -r requirements.txt

   2.2 DEPENDENCIES
        pip install transformers datasets seqeval numpy torch

   2.3 RUN TRAINING
        python train_ner.py

    This will:

        Load and preprocess the data

        Train a bert-base-uncased model

        Save logs to logs/

        Save checkpoints to results/

        Save final model to ner-bert-model/

   2.4  INFERENCE
        To test the model on new sentences:

        python inference.py

        The script will print entity predictions for sample inputs.

PROJECT STRUCTURE

NER_FoodOrdering/
├── data/                 # Raw and processed data
├── logs/                 # Training logs (auto-generated)
├── results/              # Model checkpoints and outputs (auto-generated)
├── ner-bert-model/       # Final saved model after training
├── train_ner.py          # Training pipeline script
├── inference.py          # Inference script for predictions
└── README.md             # Project documentation
