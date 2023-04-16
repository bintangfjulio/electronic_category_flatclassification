import re
import string
import torch
import pandas as pd

from models.bert_cnn import BERT_CNN
from utils.tree_helper import Tree_Helper
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from transformers import BertTokenizer

def Inference(checkpoint, max_length, num_classes):
    stop_words = StopWordRemoverFactory().get_stop_words()
    stemmer = StemmerFactory().create_stemmer()
    tokenizer = BertTokenizer.from_pretrained('indolem/indobert-base-uncased')

    text = input('Insert text to predict:')
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
    
    input_ids = token['input_ids'].to('cuda')
    
    model = BERT_CNN(num_classes=num_classes, bert_model='indolem/indobert-base-uncased', dropout=0.1)
    model.load_state_dict(checkpoint['model_state'])
    model.to('cuda')
    model.zero_grad()
    
    model.eval()

    with torch.no_grad():
        logits = model(input_ids)        
        print('Logits:', logits)

    return torch.argmax(logits, dim=1)

if __name__ == '__main__':
    # get max length
    dataset = pd.read_csv(f'datasets/small_product_tokopedia.csv')
    sentences_token = []
    for row in dataset.values.tolist():
        row = str(row[0]).split()
        sentences_token.append(row)
    token_length = [len(token) for token in sentences_token]
    max_length = max(token_length) + 5

    # get hierarchy
    tree = Tree_Helper(tree_file=f'datasets/small_hierarchy.tree')
    level_on_nodes_indexed, idx_on_section, section_on_idx, section_parent_child = tree.get_hierarchy()
    print('Idx for Each Section:', idx_on_section)
    print('Grouped Child with Parent:', section_parent_child)
    
    # setup root section
    pivot = list(section_parent_child['root'])[0]
    section = section_on_idx[pivot]
    print('Section', section, 'Inferencing...')

    # inference hierarchically
    num_level = len(level_on_nodes_indexed)
    for level in range(num_level):
        preds = Inference(checkpoint=torch.load(f"checkpoints/section_result/section_{section}_temp.pt"), max_length=max_length, num_classes=len(idx_on_section[section]))
        
        if level < (num_level - 1):
            category = idx_on_section[section][preds]
            print('Current predicted:', category)
            pivot = list(section_parent_child[category])[0]
            section = section_on_idx[pivot]
            print('Next section:', section)

        else:
            category = idx_on_section[section][preds]
            print('Final predicted', category)
