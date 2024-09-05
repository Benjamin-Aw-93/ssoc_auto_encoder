import os
import sys
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Move the model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    return model, tokenizer, device

def generate_embeddings(data, tokenizer, model, device, batch_size=8):
    embeddings = []

    for index in range(0, len(data), batch_size):
        batch = data.iloc[index:index+batch_size]
        
        job_title_tokens = tokenizer(batch['job_title'].tolist(), return_tensors="pt", padding=True, truncation=True)
        job_description_tokens = tokenizer(batch['job_description'].tolist(), return_tensors="pt", padding=True, truncation=True)

        # Move tokens to the device
        job_title_tokens = {k: v.to(device) for k, v in job_title_tokens.items()}
        job_description_tokens = {k: v.to(device) for k, v in job_description_tokens.items()}

        job_title_embedding = model(**job_title_tokens).last_hidden_state.mean(dim=1).squeeze().tolist()
        job_description_embedding = model(**job_description_tokens).last_hidden_state.mean(dim=1).squeeze().tolist()

        for i, row in batch.iterrows():
            embeddings.append((row['SSOC'], job_title_embedding[i], job_description_embedding[i]))

    return embeddings

def main():
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    model_output_path = sys.argv[3]

    model_name = "./model"
    model, tokenizer, device = load_model_and_tokenizer(model_name)

    data = pd.read_csv(input_path)

    embeddings = generate_embeddings(data, tokenizer, model, device)
    df_embeddings = pd.DataFrame(embeddings, columns=["SSOC"] + [f"e{i}" for i in range(2*len(embeddings[0])-1)])
    df_embeddings.to_csv(output_path, index=False)

    # Save the model and tokenizer to S3
    tokenizer.save_pretrained(model_output_path)
    model.save_pretrained(model_output_path)

if __name__ == "__main__":
    main()