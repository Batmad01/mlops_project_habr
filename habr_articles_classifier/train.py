import hydra
import pandas as pd
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

import habr_articles_classifier.utils as u
from habr_articles_classifier.model import RuBertpl
from habr_articles_classifier.module import ruBertDataModule, ruBertDataset


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def train(cfg: DictConfig):
    sample = True
    if sample:
        # Обучение на сэмпле данных для ускорения процесса
        df = pd.read_parquet(cfg.train_config.sample_path)
    else:
        # Обучение на всём датасете для максимального качества
        df = pd.read_parquet(cfg.train_config.data_path)

    # Данные
    X, y = u.df_preprocess(df)
    X_train, X_val, y_train, y_val = train_test_split(
        X.tolist(), y, test_size=0.25, random_state=42
    )
    num_labels = len(y_train[0])

    # Модель
    model_name = cfg.train_config.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = RuBertpl(
        model_name=model_name,
        num_labels=num_labels,
        lr=cfg.train_config.learning_rate,
        weight_decay=cfg.train_config.weight_decay,
    )

    # Датасеты
    train_dataset = ruBertDataset(
        X_train, y_train, tokenizer, max_length=cfg.train_config.max_length
    )
    val_dataset = ruBertDataset(X_val, y_val, tokenizer, max_length=cfg.train_config.max_length)
    dm = ruBertDataModule(train_dataset, val_dataset, batch_size=cfg.train_config.batch_size)

    # Колбэки
    callbacks = ModelCheckpoint(
        dirpath=cfg.train_config.dirpath,
        filename=cfg.train_config.filename,
        save_top_k=cfg.train_config.save_top_k,
        monitor=cfg.train_config.metric,
        mode=cfg.train_config.mode,
        save_weights_only=True,
    )

    # Логинг
    logger = TensorBoardLogger("tb_logs", name=cfg.expname)

    # Трейнер
    trainer = pl.Trainer(
        max_epochs=cfg.train_config.max_epochs,
        accelerator="gpu",
        logger=logger,
        devices=1,
        precision=cfg.train_config.precision,
        callbacks=callbacks,
    )
    trainer.fit(model, dm)


# Запуск
if __name__ == "__main__":
    train()
