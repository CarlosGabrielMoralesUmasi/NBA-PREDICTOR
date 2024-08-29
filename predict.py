import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification

# Ruta al archivo guardado
model_save_path = 'nba_bert_model.pth'

# Cargar el modelo entrenado
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model.load_state_dict(torch.load(model_save_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
model.to('cuda' if torch.cuda.is_available() else 'cpu')
model.eval()

# Cargar el tokenizador
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def predecir_ganador(equipo_local, equipo_visitante):
    # Crear la entrada de texto para el modelo
    input_text = f"{equipo_local} vs {equipo_visitante}"
    
    # Tokenizar la entrada
    encoded_features = tokenizer(
        input_text,
        padding='max_length', max_length=20, truncation=True, return_tensors="pt"
    ).to('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Realizar la predicción
    with torch.no_grad():
        outputs = model(**encoded_features)
        prediction = torch.argmax(outputs.logits, dim=-1).item()
    
    # Interpretar la predicción
    if prediction == 1:
        return f"{equipo_local} GANARÍA A {equipo_visitante}"
    else:
        return f"{equipo_visitante} GANARÍA A {equipo_local}"

# Interfaz para ingresar equipos en la consola
def main():
    print("Ingrese los nombres de los equipos para predecir el ganador:")
    equipo_local = input("Nombre del equipo local: ").strip()
    equipo_visitante = input("Nombre del equipo visitante: ").strip()

    resultado = predecir_ganador(equipo_local, equipo_visitante)
    print(resultado)

if __name__ == "__main__":
    main()
