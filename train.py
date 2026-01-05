import argparse
import os
import pandas as pd
import xgboost as xgb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, default='/opt/ml/input/data/train')
    parser.add_argument('--model_dir', type=str, default='/opt/ml/model')
    args = parser.parse_args()

    # Load training data (CSV without header)
    train_path = os.path.join(args.train, 'xgb_train.csv')
    df = pd.read_csv(train_path, header=None)

    # Split features and label (label must be last column)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    dtrain = xgb.DMatrix(X, label=y)

    params = {
        'objective': 'binary:logistic',
        'max_depth': 5,
        'eta': 0.2,
        'eval_metric': 'logloss'
    }

    model = xgb.train(params, dtrain, num_boost_round=200)

    # Save model where SageMaker expects it
    model_path = os.path.join(args.model_dir, 'xgboost-model')
    model.save_model(model_path)

    print('Model saved to', model_path)

if __name__ == '__main__':
    main()
