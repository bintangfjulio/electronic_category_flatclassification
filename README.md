# Still progress for research 
## Dataset
Repository for all scraping results and also the scraping utils for web scraping on https://www.tokopedia.com/ are:<br />https://huggingface.co/datasets/bintangfjulio/hierarchylabels_tokopediaproduct/tree/main (for now still private)
## Code cell must be run to use it on Google Collab (GPU must on!)
1. !git clone https://github.com/bintangfjulio/productcategories_hierarchicalclass_classification.git
2. %cd productcategories_hierarchicalclass_classification
3. !pip install -r requirements.txt
4. !python main.py --model=option choosed --method=option choosed
</br>There are chooses option for run main.py:</br>1. model: bert, bert-cnn, bert-bilstm, bert-lstm</br>2. method: flat, hierarchy
