import xlearn as xl
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--train", help="training file")
parser.add_argument("--test", help="testing file")
parser.add_argument("--output", help="output file")
args = parser.parse_args()

# Training task
fm_model = xl.create_fm()  # Use factorization machine
fm_model.setTrain(args.train)    # Training data
fm_model.setOnDisk()   #  Set xlearn to use on-disk training.
fm_model.setTXTModel('./exp/model.txt')


# param:
#  0. regression task
#  1. learning rate: 0.2
#  2. regular lambda: 0.002
#  3. evaluation metric: mae
param = {'task':'reg', 'lr':0.25, 'k':40, 'epoch':50,
        'lambda':0.002, 'metric':'rmse','opt':'adagrad'}

# Start to train
# The trained model will be stored in model.out
fm_model.fit(param, './exp/model.out')

# Prediction task
#fm_model.setTest(args.test)  # Test data

# Start to predict
# The output result will be stored in output.txt
#fm_model.predict("./exp/model.out",args.output)
