# 匯入所需的庫
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import json
from tqdm import tqdm
import time
import jieba
import gensim
import re

# 檢查並設置訓練設備（GPU 或 CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 印出環境檢測結果：PyTorch版本和訓練設備
print("<環境檢測>")
print(f"PyTorch版本 : {torch.__version__}")
print(f"訓練設備 : {device}")

# 準備建立對話資料集
current_directory = os.path.dirname(os.path.abspath(__file__))
json_model_path = os.path.join(current_directory, 'NagatoSakura_data.json')
with open(json_model_path, 'r', encoding='utf-8') as json_file:
    model_data = json.load(json_file)
    
# 定義用於處理對話資料的 Dataset 類別
class NagatoSakuraDataset(Dataset):
    def __init__(self, NagatoSakuras):
        self.NagatoSakuras = NagatoSakuras

    def __len__(self):
        return len(self.NagatoSakuras)

    def __getitem__(self, idx):
        input_text, output_text = self.NagatoSakuras[idx]['input'], self.NagatoSakuras[idx]['output']
        input_text = str(input_text)
        output_text = str(output_text)
        input_vector = text_to_vector(input_text)
        output_vector = text_to_vector(output_text)
        return input_vector, output_vector

# 提取對話資料中的指令和輸出，建立詞彙表，並將其儲存為 JSON 格式的文件
instructions = [item["instruction"] for item in model_data]
outputs = [item["output"] for item in model_data]
vocab = list(set(''.join([text for text in instructions])))
tokenizer_path = os.path.join(current_directory, 'tokenizer.json')
with open(tokenizer_path, 'w') as json_file:
    json.dump(vocab, json_file, indent=2)
print(f"詞彙表生成完成 Tokenizer 儲存為 .json 於 {tokenizer_path}")

# 載入預先訓練的中文 Word2Vec 模型
chinese_word_vectors_path = os.path.join(current_directory, 'NagatoSakura_vectors.bin')
chinese_model = gensim.models.KeyedVectors.load_word2vec_format(chinese_word_vectors_path, binary=True)

# 定義將文字轉換為向量表示的方法 text_to_vector
# 這個方法使用了 Jieba 進行分詞，並將文字轉換為對應的向量表示
def text_to_vector(text):
    if isinstance(text, torch.Tensor):
        text = text.detach().cpu().numpy()
    text_str = str(text)
    
    # 清洗文本
    text_str = re.sub(r'[^\w\s]', '', text_str)
    
    vectors = []
    max_seq_length = 128
    words = jieba.lcut(text_str)
    
    indices = []
    for word in words:
        if word in vocab:
            indices.append(vocab.index(word))
        else:
            indices.append(0)
    
    if len(indices) < max_seq_length:
        padding = [0] * (max_seq_length - len(indices))
        indices.extend(padding)
    elif len(indices) > max_seq_length:
        indices = indices[:max_seq_length]

    return torch.tensor(indices, dtype=torch.long).to(device)

# 建立 Tokenizer 配置文件，並儲存為 JSON 格式的文件
tokenizer_config = {
    "special_tokens_map": {
        "<unk>": 0,
        "<s>": 1,
        "</s>": 2,
        "<pad>": 3
    },
    "tokenizer_type": "BPE",
}
tokenizer_config_path = os.path.join(current_directory, 'tokenizer_config.json')
with open(tokenizer_config_path, 'w') as json_file:
    json.dump(tokenizer_config, json_file, indent=2)
print(f"Tokenizer 配置文件生成完成於 {tokenizer_config_path}")

# 建立特殊 tokens 映射文件，並儲存為 JSON 格式的文件
special_tokens_map = {
    0: "<unk>",
    1: "<s>",
    2: "</s>",
    3: "<pad>"
}
special_tokens_map_path = os.path.join(current_directory, 'special_tokens_map.json')
with open(special_tokens_map_path, 'w') as json_file:
    json.dump(special_tokens_map, json_file, indent=2)
print(f"特殊 tokens 映射文件生成完成於 {special_tokens_map_path}")

# 設定模型相關的參數
vocab_size = len(vocab)
d_model = 1024   # 隱藏層維度 
num_layers = 7  # 模型層數
num_heads = 8   # 注意力頭數
dropout = 0.1    
max_seq_length = 128

# 定義對話 Transformer 模型
class NagatoSakuraTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dropout=dropout, batch_first=True),
            num_layers=num_layers
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=num_heads, dropout=dropout, batch_first=True),
            num_layers=num_layers
        )
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        memory = self.encoder(src)
        output = self.decoder(tgt, memory)
        output = self.fc(output)
        return output


# 開始訓練模型
print("<訓練開始>")
NagatoSakura_model = NagatoSakuraTransformer(vocab_size, d_model, num_layers, num_heads, dropout).to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(NagatoSakura_model.parameters(), lr=1e-5)

# 初始化訓練參數
training_start_time = time.time()
batch_size = 32
NagatoSakuras = [
    {"input": item["instruction"], "output": item["output"]} for item in model_data
]
NagatoSakura_dataset = NagatoSakuraDataset(NagatoSakuras)
batch_size = 32
NagatoSakura_loader = DataLoader(NagatoSakura_dataset, batch_size=batch_size, shuffle=True)

epochs = 25
for epoch in range(epochs):
    total_loss = 0
    start_time = time.time()

    # 迭代進行訓練
    for batch_x, batch_y in NagatoSakura_loader:
        optimizer.zero_grad()
        batch_x_processed = []
        batch_y_processed = []
        for text_x, text_y in zip(batch_x, batch_y):
            processed_x = text_to_vector(text_x)
            processed_y = text_to_vector(text_y)
            batch_x_processed.append(processed_x)
            batch_y_processed.append(processed_y)

        batch_x_processed = torch.stack(batch_x_processed).to(device)
        batch_y_processed = torch.stack(batch_y_processed).to(device)

        output = NagatoSakura_model(batch_x_processed, batch_y_processed)
        loss = criterion(output.view(-1, vocab_size), batch_y_processed.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    # 計算平均損失和訓練進度
    average_loss = total_loss / len(NagatoSakura_loader)
    elapsed_time = time.time() - start_time
    eta = (epochs - epoch - 1) * elapsed_time
    total_training_time = time.time() - training_start_time
    
    progress = epoch + 1
    percentage = progress / epochs * 100
    fill_length = int(50 * progress / epochs)
    space_length = 50 - fill_length
    print(f"Processing: {percentage:3.0f}%|{'█' * fill_length}{' ' * space_length}| {progress}/{epochs} [{total_training_time:.2f}<{eta:.2f}, {1 / elapsed_time:.2f}it/s, Loss: {average_loss:.4f}] ")
       
print("訓練完成")

# 生成模型配置文件並儲存為 JSON 格式的文件
model_config = {
    "_name_or_path": "NagatoSakura",
    "model_type": "NagatoSakuraTransformer",
    "architectures": ["Nagato Sakura Model"],
    "vocab_size": vocab_size,
    "d_model": d_model,
    "num_layers": num_layers,
    "num_heads": num_heads,
    "dropout": dropout,
    "learning_rate": optimizer.param_groups[0]['lr'],
    "max_seq_length": max_seq_length
}

config_path = os.path.join(current_directory, 'config.json')
with open(config_path, 'w') as json_file:
    json.dump(model_config, json_file, indent=4)
print(f"模型配置文件生成完成 模型配置文件儲存於 {config_path}")

# 儲存訓練完成的模型權重
print("<保存模型>")
model_path = os.path.join(current_directory, 'NagatoSakura_model.bin')
torch.save(NagatoSakura_model.state_dict(), model_path)
print(f"保存模型完成，模型已保存於 {model_path}")