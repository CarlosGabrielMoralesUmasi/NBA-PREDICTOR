# NBA Game Winner Prediction Model

This project leverages a fine-tuned BERT model to predict the winner of NBA games based on the teams playing. The model is trained on historical NBA game data and predicts whether the home team will win or lose a given match.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training the Model](#training-the-model)
- [Model Performance](#model-performance)
- [Making Predictions](#making-predictions)
- [Usage](#usage)
- [Requirements](#requirements)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Introduction

The aim of this project is to create a predictive model that can forecast the outcome of NBA games based on the teams involved. We utilize the BERT (Bidirectional Encoder Representations from Transformers) model, originally designed for natural language processing tasks, and adapt it to the task of NBA game prediction by treating team matchups as textual sequences.

## Dataset

The dataset used for training the model was sourced from [Kaggle](https://www.kaggle.com/datasets/wyattowalsh/basketball?select=csv). It contains comprehensive historical data on NBA games, including information on teams, scores, game outcomes, and more.

Files used:
- **game.csv:** Contains the details of individual games, including team names, scores, and outcomes.
- **game_info.csv:** Provides additional metadata about the games.

## Model Architecture

The model is based on the `BertForSequenceClassification` architecture from Hugging Face's Transformers library. This model is originally designed for binary text classification tasks, but it has been adapted here to predict the outcome of NBA games.

### Components:
- **BERT Encoder:** The model uses BERT as its backbone to encode the matchup between two teams into a rich, contextualized representation.
- **Classification Layer:** A fully connected layer that predicts the probability of the home team winning the game.

**Number of Parameters:** 109,483,778

**Trainable Parameters:** 109,483,778

The BERT model is pre-trained on vast amounts of text data, and we fine-tune it on the specific task of game prediction.

## Training the Model

The training process involves the following steps:
1. **Data Preprocessing:** The team names from the dataset are formatted as textual sequences ("Team A vs Team B") and tokenized using the BERT tokenizer.
2. **Model Training:** The model is trained using the AdamW optimizer with a learning rate of 2e-5 for 3 epochs. The training is performed on a GPU to accelerate the process.
3. **Evaluation:** After training, the model is evaluated on a test set to assess its performance.

The target variable is binary, where `1` represents a home team win and `0` represents a loss.

### Code for Training
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import torch
from torch.utils.data import DataLoader, Dataset

# Load and process the dataset
games = pd.read_csv('C:\\Users\\carlo\\Documents\\Proyectos\\NBA_predict\\csv\\game.csv')
game_info = pd.read_csv('C:\\Users\\carlo\\Documents\\Proyectos\\NBA_predict\\csv\\game_info.csv')
games_full = games.merge(game_info, on='game_id', how='left', suffixes=('_x', '_y'))
games_full['game_date'] = pd.to_datetime(games_full['game_date_x'])
games_full.sort_values(by='game_date', inplace=True)
features = games_full[['team_abbreviation_home', 'team_abbreviation_away']]
target = (games_full['wl_home'] == 'W').astype(int)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Tokenize the input features
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
def encode_features(df):
    input_texts = df['team_abbreviation_home'] + " vs " + df['team_abbreviation_away']
    inputs = tokenizer(input_texts.tolist(), padding='max_length', max_length=20, truncation=True, return_tensors="pt")
    return inputs

# Create a custom dataset
class GameDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.features.items()}
        item['labels'] = torch.tensor(self.labels.iloc[idx], dtype=torch.long)
        return item

# Prepare data loaders
train_features = encode_features(X_train)
test_features = encode_features(X_test)
train_dataset = GameDataset(train_features, y_train)
test_dataset = GameDataset(test_features, y_test)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize the model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

# Train the model
model.train()
for epoch in range(3):
    for i, batch in enumerate(train_loader, 0):
        inputs = {'input_ids': batch['input_ids'].to(device), 'attention_mask': batch['attention_mask'].to(device), 'labels': batch['labels'].to(device)}
        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print(f'[{epoch + 1}, {i + 1}] loss: {loss.item()}')
print('Finished Training')

# Save the model
model_save_path = 'nba_bert_model.pth'
torch.save(model.state_dict(), model_save_path)
print(f'Model saved to {model_save_path}')

## Model Performance

After training, the model achieved the following results on the test set:

```
Accuracy: 0.62
Macro Avg:
  Precision: 0.31
  Recall: 0.50
  F1-score: 0.38
Weighted Avg:
  Precision: 0.38
  Recall: 0.62
  F1-score: 0.47
```

These results suggest that while the model has some predictive power, there is room for improvement, especially in terms of precision and F1-score.

## Making Predictions

After training the model, you can use it to predict the outcome of future NBA games. The model accepts the names of the two teams playing and predicts whether the home team will win.

### Example Code for Prediction
```python
import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification

# Load the trained model
model_save_path = 'nba_bert_model.pth'
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model.load_state_dict(torch.load(model_save_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
model.to('cuda' if torch.cuda.is_available() else 'cpu')
model.eval()

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def predecir_ganador(equipo_local, equipo_visitante):
    input_text = f"{equipo_local} vs {equipo_visitante}"
    encoded_features = tokenizer(input_text, padding='max_length', max_length=20, truncation=True, return_tensors="pt").to('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        outputs = model(**encoded_features)
        prediction = torch.argmax(outputs.logits, dim=-1).item()
    if prediction == 1:
        return f"{equipo_local} GANARÍA A {equipo_visitante}"
    else:
        return f"{equipo_visitante} GANARÍA A {equipo_local}"

# Example of usage
if __name__ == "__main__":
    equipo_local = input("Nombre del equipo local: ").strip()
    equipo_visitante = input("Nombre del equipo visitante: ").strip()
    resultado = predecir_ganador(equipo_local, equipo_visitante)
    print(resultado)
```

## Usage

To use this project, follow these steps:

1. **Clone the repository.**
2. **Install the dependencies listed in `requirements.txt`.**
3. **Run the training script to train the model on the NBA dataset.**
4. **Use the prediction script to make predictions on NBA games.**

## Requirements

This project requires the following Python packages:

- `torch`
- `transformers`
- `sklearn`
- `pandas`

Install them using pip:

```bash
pip install torch transformers sklearn pandas
```

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the Boost Software License. See the `LICENSE` file for more details.

## Acknowledgements

- Dataset source: [Kaggle - Basketball Dataset](https://www.kaggle.com/datasets/wyattowalsh/basketball?select=csv)
- BERT model and tokenizer: [Hugging Face Transformers](https://github.com/huggingface/transformers)
