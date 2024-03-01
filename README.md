## Dataset
The repository for all scraping utils and dataset results from web scraping on Tokopedia is: https://www.kaggle.com/datasets/bintangfajarjulio/product-with-categories-from-tokopedia <br><br>
You also can import it from this repository dataset releases

## Run It
```
python main.py --method= ...
```
There are three method options for run main.py: flat, level, section <br> 

Flat: Training model with only lowest level class as multiclass classification <br>
Level: Training model each level and transfer the parameters to lower level <br>
Section: Training model with segmentation based on upper-level category to lower-level category as child class
