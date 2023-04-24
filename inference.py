import re
import string
import torch
import argparse
import pandas as pd

from models.bert_cnn import BERT_CNN
from utils.tree_helper import Tree_Helper
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from transformers import BertTokenizer

def Inference(text, bert_model, dropout_prob, checkpoint, max_length, num_classes):
    stop_words = StopWordRemoverFactory().get_stop_words()
    stemmer = StemmerFactory().create_stemmer()
    tokenizer = BertTokenizer.from_pretrained(bert_model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    text = text.lower()
    text = re.sub(r"[^A-Za-z0-9(),!?\'\-`]", " ", text)
    text = re.sub('\n', ' ', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub("'", '', text)
    text = re.sub(r'\d+', '', text)
    text = ' '.join([word for word in text.split() if word not in stop_words and len(word) > 1])
    text = stemmer.stem(text.strip())

    token = tokenizer.encode_plus(
        text=text,
        add_special_tokens=True,
        max_length=max_length,
        return_tensors='pt',
        padding="max_length", 
        truncation=True)
    
    input_ids = token['input_ids'].to(device)
    
    model = BERT_CNN(num_classes=num_classes, bert_model=bert_model, dropout=dropout_prob)
    model.load_state_dict(checkpoint['model_state'])
    model.to(device)
    model.zero_grad()

    model.eval()

    with torch.no_grad():
        logits = model(input_ids)        
        print('Logits:', logits)
        preds = torch.argmax(logits, dim=1)

    return preds

def parsing_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='small')
    parser.add_argument("--bert_model", type=str, default='indolem/indobert-base-uncased')
    parser.add_argument("--dropout", type=float, default=0.1)
    config = vars(parser.parse_args())

    return config['dataset'], config['bert_model'], config['dropout']

if __name__ == '__main__':
    type_set, bert_model, dropout_prob = parsing_argument()
    text = input('Insert text to predict: ')
        
    dataset = pd.read_csv(f'datasets/{type_set}_product_tokopedia.csv')
    sentences_token = []
    for row in dataset.values.tolist():
        row = str(row[0]).split()
        sentences_token.append(row)
    token_length = [len(token) for token in sentences_token]
    max_length = max(token_length) + 5

    tree = Tree_Helper(tree_file=f'datasets/{type_set}_hierarchy.tree')
    tree.generate_hierarchy()
    level_on_nodes_indexed, idx_on_section, section_on_idx, section_parent_child = tree.get_hierarchy()
    
    print('Index for each section:', idx_on_section)
    print('Grouped child with parent:', section_parent_child)
    
    pivot = list(section_parent_child['root'])[0]
    section = section_on_idx[pivot]
    print('Inferencing categories...')

    num_level = len(level_on_nodes_indexed)
    for level in range(num_level):
        preds = Inference(text=text, bert_model=bert_model, dropout_prob=dropout_prob, checkpoint=torch.load(f"checkpoints/section_result/section_{section}_temp.pt"), max_length=max_length, num_classes=len(idx_on_section[section]))
        
        if level < (num_level - 1):
            category = idx_on_section[section][preds]
            print('Current predicted:', category)
            pivot = list(section_parent_child[category])[0]
            section = section_on_idx[pivot]
            print('Next section:', section)

        else:
            category = idx_on_section[section][preds]
            print('Final predicted', category)
