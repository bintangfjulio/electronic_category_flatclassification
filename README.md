## Dataset
Repository for all scraping results and also the scraping utils for web scraping on https://www.tokopedia.com/ are:<br />https://www.kaggle.com/datasets/bintangfajarjulio/product-with-categories-from-tokopedia (For now still private)
## Code cell must be run to use it on Notebook Python (GPU must on!)
1. !git clone https://github.com/bintangfjulio/productcategories_hierarchicalclass_classification.git
2. %cd productcategories_hierarchicalclass_classification
3. !pip install -r requirements.txt
4. !python main.py --model=option choosed --method=option choosed
</br>There are option for run main.py:</br>1. model: bert, bert-cnn, bert-bilstm, bert-lstm</br>2. method: flat, hierarchy
