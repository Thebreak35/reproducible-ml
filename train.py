import copy as cp
import operator
import logging
import os
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import pycrfsuite

from sefr_cut.preprocessing import Preprocessor
import sefr_cut.extract_features as extract_features


logger = logging.getLogger(__name__)


def create_dir():
    Path('./model').mkdir(parents=True, exist_ok=True)


@hydra.main(config_path="configs", config_name="default")
def train(cfg: DictConfig):
    wandb.login(key="")
    wandb.init(**cfg.wandb.init, config=cfg.trainer)
    logger.info("Training model")

    PATH = Path(hydra.utils.get_original_cwd())
    PATH_CORPUS = [PATH / cfg.path.path_corpus]

    logger.info("Preprocess")
    
    prepro = Preprocessor()
    # Create x, y
    x, y_true = prepro.preprocess_x_y(PATH_CORPUS)

    # 2D to 1D
    y_true = [j for sub in y_true for j in sub]
    x = [j for sub in x for j in sub]

    y_pred,y_entropy,y_prob = prepro.predict_(x) # DeepCut Baseline/BEST+WS/WS

    X_data = []
    for idx,item in enumerate(x):
        X_data.append(extract_features.extract_features_crf(x[idx],y_entropy[idx],y_prob[idx]))
    y_data = [list(map(str, l)) for l in y_true]

    #2D to 1D
    X_data_1d = [j for sub in X_data for j in sub] 
    y_data_1d = [j for sub in y_data for j in sub]
    
    logger.info("Create train, test, split")

    X_train, X_test, y_train, y_test = train_test_split(X_data_1d, y_data_1d, **cfg.train_test_split)

    CRF_MODEL_NAME = cfg.model.name

    trainer = pycrfsuite.Trainer(verbose=True)

    for xseq, yseq in zip(X_train, y_train):
        trainer.append(xseq, yseq)

    logger.debug("Set trainer parameters")
    logger.info("\n" + OmegaConf.to_yaml(cfg.trainer))

    trainer.set_params(cfg.trainer)

    MODEL_PATH = PATH / "model" / CRF_MODEL_NAME
    logger.debug("Set params completed")

    trainer.train(str(MODEL_PATH))

    logger.info("Train model completed")

    # Load model
    logger.info("Load model for evaluation")

    tagger = pycrfsuite.Tagger()
    tagger.open(str(MODEL_PATH))
    y_pred = [tagger.tag(xseq) for xseq in X_test]

    logger.info("Loaded model")

    # Evaluate
    logger.info("Evaluating model")

    labels = {'1': 1, "0": 0} # classification_report() needs values in 0s and 1s
    predictions = np.array([labels[tag] for row in y_pred for tag in row])
    truths = np.array([labels[tag] for row in y_test for tag in row])

    logger.info("\n"
    +classification_report(
        truths, 
        predictions, 
        target_names=["B", "I"]))
    wandb.log(classification_report(
        truths, 
        predictions, 
        target_names=["B", "I"],
        output_dict=True))
    wandb.finish()


if __name__ == '__main__':
    create_dir()
    train()
