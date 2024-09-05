from transformers import AutoTokenizer, AutoModel

model_name = 'BAAI/bge-base-en'
AutoTokenizer.from_pretrained(model_name).save_pretrained('./model')
AutoModel.from_pretrained(model_name).save_pretrained('./model')