# Still progress for research 
## Dataset
Repository for all scraping results and also the scraping utils for web scraping on https://www.tokopedia.com/ are:<br />https://huggingface.co/datasets/bintangfjulio/hierarchylabels_tokopediaproduct/tree/main (for now still private)
## Code cell must be run to use it on Google Collab (GPU must on!)
1. !git clone https://github.com/bintangfjulio/productcategories_hierarchicalclass_classification.git
2. %cd productcategories_hierarchicalclass_classification
3. !git lfs pull
4. !pip install -r requirements.txt
5. !python main.py --model={model} --flat={flat} --hierarchy={hierarchy}</br>
Note:</br>there is chooses option for run main.py are:</br>model = 'bert', 'bert-cnn', 'bert-bilstm', 'bert-lstm'</br>flat = True, False</br>hierarchy = True, False
