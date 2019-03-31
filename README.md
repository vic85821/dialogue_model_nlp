# Dialogue Model using NLP 

This project is about the dialogue QA. There would be some utterances and what we need to do is select the best answer from the 100 candidates. There are three models implemented, which are RNN w/o attention, RNN w/ attention, and the model with the best performance. In the report, I list some comparison between differenct RNN models (e.g. LSTM, GRU), and the implementation details.

## Requirements

```
    conda env create -f nlp.yml
```

## Training

### Prepare data
* put the train.json, valid.json, test.json in the `data` folder
* download the english word vectors `crawl-300d-2M.vec` from FastText (https://fasttext.cc/docs/en/english-vectors.html) and also put it into `data` folder

So there are these files in the data folder as follow:
```
    ./data/config.json # config setting
    ./data/train.json # training data
    ./data/valid.json # validation data
    ./data/test.json # testing data
    ./data/crawl-300d-2M.vec # english word vectors
```

### Train the model
* prepare the `models` folder 
* create experiment folder, e.g. `lstm`
* add the `config.json` which contains the experiment settings into the experiment folder
```
    ./models/lstm/config.json
```

run the training process
```
    cd src
    bash preprocess.sh # preprocess the json to pickle 
    bash train.sh model_path cuda_device
```

### Pre-trained model

Use `gdrive` package (https://github.com/gdrive-org/gdrive) to download the pre-trained model
```
    bash download.sh
```

## Testing

`bash rnn.sh/attention.sh/best.sh ${1} ${2}`
* `${1}` path_to_the_test_json 
* `${2}` path_to_the_predictions


## Attention Score Plot
* there should be a `best` folder in the `models`
* need to preprocess the data to the `pkl` format
* need to prepare `embedding.pkl` which contains the englist word embedding info

```
    cd src
    python visual.py data_path, embed_path
    
    # example
    python visual.py ../data/valid.pkl ./embedding.pkl
```