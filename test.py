import torch
import json
import os
import jieba
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def load_model_config():
    with open('config.json', 'r') as config_file:
        model_config = json.load(config_file)
    return model_config

def create_model(model_config):
    vocab_size = model_config['vocab_size']
    d_model = model_config['d_model']
    num_layers = model_config['num_layers']
    num_heads = model_config['num_heads']
    dropout = model_config['dropout']
    
    model = NagatoSakuraTransformer(vocab_size, d_model, num_layers, num_heads, dropout)
    model.to(device) 
    
    model.eval()
    
    weights_path = 'NagatoSakura_model.bin'  
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path))

    return model

def text_to_vector(text, vocab, max_seq_length):
    if isinstance(text, torch.Tensor):
        text = text.detach().cpu().numpy()
    text_str = str(text)
    vectors = []
    
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

    return torch.tensor(indices, dtype=torch.long, device=device)

def generate_response(input_text, vocab, model, max_seq_length):
    input_vector = text_to_vector(input_text, vocab, max_seq_length)
    output = model(input_vector.unsqueeze(0), input_vector.unsqueeze(0))
    output = output.argmax(dim=2).squeeze(0)
    print(output)
    decoded_response = ''.join([vocab[int(idx)] for idx in output])
    return decoded_response

def main():
    model_config = load_model_config()
    
    vocab = None
    with open('tokenizer.json', 'r') as json_file:
        vocab = json.load(json_file)

    NagatoSakura_model = create_model(model_config)

    max_seq_length = model_config['max_seq_length']

    while True:
        user_input = input("你：")
        if user_input.lower() == 'exit':
            print("對話結束。")
            break
        response = generate_response(user_input, vocab, NagatoSakura_model, max_seq_length) 
        print("模型：", response)

if __name__ == "__main__":
    main()