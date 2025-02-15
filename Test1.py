import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential, load_model

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print("GPU detectada! TensorFlow rodando na GPU.")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print("Nenhuma GPU detectada. TensorFlow rodando na CPU.")

directory = "/home/ian/ml/dados/dados/"
file_path_save = "/home/ian/ml/ias/modelo_treinado_t4.h5"

if os.path.exists(file_path_save):
    print("Carregando modelo existente...")
    modelo = load_model(file_path_save)
else:
    print("Criando novo modelo...")
    modelo = Sequential([
        Dense(1, activation='relu', input_shape=(10,)),   
        Dense(1, activation='relu'),  
        Dense(5, activation='linear')
    ])
    modelo.compile(optimizer='adam', loss='mse', metrics=['mae'])
for i in range(32):
    file_path = os.path.join(directory, f"dados_pacote_{i}.npz")
    if os.path.exists(file_path):
        print(f"Carregando arquivo {file_path} ({i+1}/32)...")
        data = np.load(file_path)
        sinais = data["sinais"].reshape(1000, 6500, 10)
        labels = data["labels"]
        
        for j in range(1000):
            x_batch = sinais[j]
            y_batch = labels[j]
            modelo.fit(x_batch, y_batch, epochs=1, batch_size=32, verbose=1)
            print(f"Treinamento com lote {j+1}/1000 do arquivo {i+1}/32 concluído.")
        
        modelo.save(file_path_save)
        print(f"Modelo salvo após treinamento com o arquivo {i+1}/32.")
    else:
        print(f"Arquivo {file_path} não encontrado.")

print("Treinamento concluído e modelo salvo!")
