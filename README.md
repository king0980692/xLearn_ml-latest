# xLearn ml-latest

Experienmet apply xLearn FM model on movielens dataset.

---

## Environment set up

### Clone this project

```bash
git clone --recurse-submodules git@github.com:king0980692/xLearn_ml-latest.git
```

### Prerequisites
using python version: 3.8.1
```bash
pyenv local 3.8.1
```

tell poetry to using python3.8 and install the dependency package
```bash
poetry env use python3.8
poetry install
```

enter the virtual enviornment for runing script more simply .
```bash
poetry shell
```


## Experiment step

below using the movielens-100k dataset to describe the detail step for this experiment.

### Prepare data
```bash
mkidr ./data
wget https://files.grouplens.org/datasets/movielens/ml-100k.zip -P ./data
unzip ./data/ml-100k.zip -d ./data
```

### Using encoderder to generate the sparse format data

we will create the libsvm format training data, and the all pairs of user and item test data to predict the probability.


```bash
python3 ./encoderder/encoderder.py -c ./100k.json
```

#### 100k.json
this is a json file for encoderder to generate the sparse format of datast, it will look like :
```json=
"train": {
    "input": "./data/ml-100k/ua.base",
    "output": "./exp/ml.train",
    "cached": true,
    "seperator": "\t",
    "header": false,
    "sparse": false,
    "target_columns": [
        {
            "index": 0,
            "type": "cat"
        },
        {
            "index": 1,
            "type": "cat"
        },
        {
            "index": 2,
            "type": "truth"
        }
    ]
}

```
there some important points need to illustrate : 
* input : the input file to generate the sparse format
* output : the generated file
* target columns : select your interested column you want to encode, and specify its column type: 
    * cat : categorical type data
    * num: numerical type data
    * truth : the labeled data

* others config : you can see the more infomation in encoderder repository



### Training and Testing

```bash

python3 train_predict.py --train ./exp/ml.train --test ./exp/ml.test.all_pair --output ./result/output.txt
```

### Generate the user pred pickle

```bash
python3 gen_user_pred.py --score_file ./result/output.txt --truth_file ./exp/ml.test.all_pair
```


### Evaluation

```bash
python3 eval.py --predict ./result/user_pred_dict.pkl --truth ./exp/ml.test
```
## Performance


|         | MAP@10    |
| ------- | --------- |
| ml-100k | 0.0036496 |
| ml-10m  | 0.0023612 |
| ml-latest | 0.001496 |


