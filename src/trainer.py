from sklearn.metrics import classification_report
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.optim import AdamW, Adam
from yaml.loader import SafeLoader
from sklearn.model_selection import train_test_split
from datetime import datetime
from tqdm import tqdm
from time import time
from src import utils
import numpy as np
from torch import nn
import logging
import random
import json
import torch
import yaml
import os

from src.models.tim_net import TimNet
from src.dataset import SER_Dataset
from src.models.conformer import CNN_Conformer

current_time = datetime.now()
current_time = current_time.strftime("%d-%m-%Y_%H:%M:%S")

if not os.path.exists("logs"):
    os.mkdir("logs")
    
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"logs/log_{current_time}.log"),
        logging.StreamHandler()
    ],
    force = True
)

class Trainer():
    def __init__(self, config) -> None:
        self.config = config
        self.device = "cpu" if not torch.cuda.is_available() else config["device"]
        self.set_random_state(int(config["seed"]))
        
        with open(self.config["data_config"]) as f:
            self.data_config = yaml.load(f, Loader=SafeLoader)
            
        json_obj = json.dumps(self.data_config, indent=4, ensure_ascii=False)
        logging.info("data config: ")
        logging.info(json_obj)
        
        json_obj = json.dumps(self.config, indent=4, ensure_ascii=False)
        logging.info("general config: ")
        logging.info(json_obj)

        self.prepare_diretories_and_logger()
        self.cre_loss = torch.nn.CrossEntropyLoss()
        
        if config["mode"] == "train":
            train_features, train_masks, train_labels = utils.load_data(
                path=self.data_config["train_path"]
            )
            if config["data_config"].endswith("ravdess.yaml"):
                train_features, valid_features, train_masks, valid_masks, train_labels, valid_labels = train_test_split(
                    train_features, train_masks, train_labels, test_size=self.data_config["valid_size"], random_state=self.config["random_seed"]
                )
                train_features, test_features, train_masks, test_masks, train_labels, test_labels = train_test_split(
                    train_features, train_masks, train_labels, test_size=self.data_config["test_size"], random_state=self.config["random_seed"]
                )
                
            elif config["data_config"].endswith("tth_vlsp.yaml"):
                test_features, test_masks, test_labels = utils.load_data(
                    path=self.data_config["test_path"]
                )
                
                test_features, valid_features, test_masks, valid_masks, test_labels, valid_labels = train_test_split(
                    test_features, test_masks, test_labels, test_size=self.data_config["valid_size"], random_state=self.config["random_seed"]
                )
                
            logging.info(f"train size: {train_features.shape}")
            logging.info(f"val size: {valid_features.shape} ")
            logging.info(f"test size: {test_features.shape}")
            
            self.train_dl = self.prepare_dataloader(train_features, train_labels, train_masks, mode="train")
            self.valid_dl = self.prepare_dataloader(valid_features, valid_labels, valid_masks, mode="test")
            self.test_dl = self.prepare_dataloader(test_features, test_labels, test_masks, mode="test")
                    
        elif config["mode"] == "infer":
            pass
        
        model = self.init_model()
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        logging.info(f"num params: {params}")
        
    def init_weight(self, layer):
        if isinstance(layer, nn.Linear):
            torch.nn.init.xavier_uniform_(layer.weight)

    def set_random_state(self, seed):
        logging.info(f'set random_seed = {seed}')
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)
        
    def prepare_diretories_and_logger(self):
        current_time = datetime.now()
        current_time = current_time.strftime("%d-%m-%Y_%H:%M:%S")

        log_dir = f"{self.config['log_dir']}/{current_time}"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            logging.info(f"logging into {log_dir}")
            
        checkpoint_dir = self.config["checkpoint_dir"]
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
            logging.info(f'mkdir {checkpoint_dir}')
        
        self.writer = SummaryWriter(
            log_dir=log_dir
        )
        
    def init_model(self):
        with open(self.config["model_config"]) as f:
            model_config = yaml.load(f, Loader=SafeLoader)
        self.model_config = model_config
        
        if "light_ser_cnn" in self.config["model_config"]:
            pass
            # model = Light_SER(self.model_config).to(self.device)
        elif "tim_net" in self.config["model_config"]:
            model = TimNet(
                n_filters=self.data_config["hidden_dim"],
                n_label=len(self.data_config["label"].keys())).to(self.device)
            print(self.data_config["label"].keys())
        elif "cnn_transformer" in self.config["model_config"]:
            pass
            # model = CNN_Transformer().to(self.device)
            
        elif "conformer" in self.config["model_config"]:
            model = CNN_Conformer(
                self.model_config, 
                n_label=len(self.data_config["label"].keys())).to(self.device)
        
        return model
            
    def init_optimizer(self, model):
        optimizer = Adam(
            params=model.parameters(),
            betas=(self.config["beta1"], self.config["beta2"]),
            lr=float(self.config["learning_rate"]),
            weight_decay=float(self.config["weight_decay"]))
        
        return optimizer
    
    def prepare_dataloader(self, features, labels, masks, mode="train"):
        dataset = SER_Dataset(
            features, labels, masks, mode=mode)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=int(self.config["batch_size"]),
            num_workers=int(self.config["num_worker"]),
            pin_memory=True,
            drop_last=False,
            shuffle=True)
        
        return dataloader
    def load_checkpoint(self, path):
        state_dict = torch.load(path, map_location="cpu")
        
        model_state_dict = state_dict["model_state_dict"]
        optim_state_dict = state_dict["optim_state_dict"]

        logging.info(f'load checkpoint from {path}')        
        
        return {
            "model_state_dict":model_state_dict,
            "optim_state_dict":optim_state_dict
        }
    def train(self):
        logging.info("########## start training #########")
        logging.info("################# init model ##################")
        model = self.init_model()
        logging.info("############### init optimizer #################")
        optimizer = self.init_optimizer(model)
        
        model.train()
        best_acc, best_wa, best_uwa = -1, -1, -1
        for epoch in range(int(self.config["n_epoch"])):
            train_losses, valid_losses = [], []
            _train_tqdm = tqdm(self.train_dl, desc=f"Epoch={epoch}")
            for i, batch in enumerate(_train_tqdm):
                optimizer.zero_grad()
                
                features = batch["feature"].float().to(self.device)
                labels = batch["label"].float().to(self.device)
                masks = batch["mask"].bool().to(self.device)
                
                _, preds = model(inputs=features, masks=masks)
                
                loss = self.cre_loss(preds, labels)
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                loss.backward()
                
                train_losses.append(loss.item())
                optimizer.step()
                
                _train_tqdm.set_postfix(
                    {"loss":loss.item()}
                )
            if (epoch+1) % int(self.config["evaluate_per_epoch"])==0:
                train_loss = np.mean(train_losses)
                target_names = list(self.data_config["label"].keys())
                model.eval()
                logging.info(f"start validation (epoch={epoch}): ")
                valid_results = self.evaluate(valid_dl=self.valid_dl, model=model)
                valid_cls_result = classification_report(
                    y_pred=valid_results["predicts"], 
                    y_true=valid_results["labels"],
                    output_dict=False, zero_division=0,
                    target_names=target_names)
                
                logging.info(f"validation result (epoch={epoch}): \n {valid_cls_result}")
                
                logging.info(f"start testing (epoch={epoch}): ")
                test_results = self.evaluate(valid_dl=self.test_dl, model=model)
                test_results = classification_report(
                    y_pred=test_results["predicts"], 
                    y_true=test_results["labels"],
                    output_dict=False, zero_division=0,
                    target_names=target_names)
                
                model.train()
                
                logging.info(f"test result (epoch={epoch}): \n{test_results}")
                   
                valid_cls_result = classification_report(
                    y_pred=valid_results["predicts"], 
                    y_true=valid_results["labels"],
                    output_dict=True, zero_division=0,
                    target_names=target_names)  
                
                # if best_acc < valid_cls_result["accuracy"]:
                #     best_acc = valid_cls_result["accuracy"]
                #     path = f'{self.config["checkpoint_dir"]}/best_acc_checkpoint.pt'
                #     self.save_checkpoint(path, model=model, optimizer=optimizer, epoch=epoch, loss=train_loss)
                    
                if best_wa < valid_cls_result["weighted avg"]["recall"]:
                    best_wa = valid_cls_result["weighted avg"]["f1-score"]
                    path = f'{self.config["checkpoint_dir"]}/best_war_checkpoint.pt'
                    self.save_checkpoint(path, model=model, optimizer=optimizer, epoch=epoch, loss=train_loss)
                    logging.info(f"test with current best checkpoint (epoch={epoch}): ")
                    self.test(checkpoint=path,test_dl=self.test_dl)                      
                # if best_uwa < valid_cls_result["macro avg"]["f1-score"]:
                #     best_uwa = valid_cls_result["macro avg"]["f1-score"]
                #     path = f'{self.config["checkpoint_dir"]}/best_uwar_checkpoint.pt'
                #     self.save_checkpoint(path, model=model, optimizer=optimizer, epoch=epoch, loss=train_loss)
                
                logging.info("############################################")
                
                json_obj = json.dumps({
                    "weighted_avg":best_wa, 
                    "epoch":epoch, 
                    "valid_loss":valid_results["loss"].tolist(),
                    "train_loss":train_loss
                    }, indent=4, ensure_ascii=False)
                
                message = "validation result: \n" + json_obj
                logging.info(message)
                logging.info("############################################")
                # path = f'{self.config["checkpoint_dir"]}/best_acc_checkpoint.pt'
                # self.test(checkpoint=path,test_dl=self.test_dl)
                
                # path = f'{self.config["checkpoint_dir"]}/best_war_checkpoint.pt'
                # self.test(checkpoint=path,test_dl=self.test_dl)
                
                # path = f'{self.config["checkpoint_dir"]}/best_uwar_checkpoint.pt'
                # self.test(checkpoint=path,test_dl=self.test_dl)
                                 
    def save_checkpoint(self, path, model, optimizer, epoch, loss):
        state_dict = {
            "model_state_dict":model.state_dict(),
            "optim_state_dict":optimizer.state_dict(),
            "loss":loss,
            "epoch":epoch,
        }
        torch.save(state_dict, path)

    def evaluate(self, model, valid_dl, mode="test"):
        predicts, labels = [], []
        
        with torch.no_grad():
            losses = []
            for i, batch in enumerate(valid_dl):
                features = batch["feature"].float().to(self.device)
                _labels = batch["label"].float().to(self.device)
                masks = batch["mask"].bool().to(self.device)
                
                _, preds = model(inputs=features, masks=masks)
                
                loss = self.cre_loss(preds, _labels)
                
                preds = torch.nn.functional.softmax(preds, dim=-1)
                preds = torch.argmax(preds, dim=-1)
                _labels = _labels.argmax(dim=-1)
                
                labels += _labels.cpu().tolist()
                predicts += preds.cpu().tolist()
                
                losses.append(loss.item())        
        return {
            "loss":torch.tensor(losses).mean(),
            "predicts":np.array(predicts),
            "labels":np.array(labels),
        }
    def test(self, checkpoint, test_dl):        
        model = self.init_model()            
        state_dict = self.load_checkpoint(checkpoint)
        model.load_state_dict(state_dict["model_state_dict"])
        model.eval()
        target_names = list(self.data_config["label"].keys())
        
        with torch.no_grad():
            test_results = self.evaluate(valid_dl=test_dl, model=model)
        
        test_cls_result = classification_report(
            y_pred=test_results["predicts"], 
            y_true=test_results["labels"],
            output_dict=False, zero_division=0,
            target_names=target_names)
        
        logging.info(test_cls_result)
        
        test_cls_result = classification_report(
            y_pred=test_results["predicts"], 
            y_true=test_results["labels"],
            output_dict=True, zero_division=0,
            target_names=target_names)
                
        return {
            "acc":test_cls_result["accuracy"],
            "war":test_cls_result["weighted avg"]["recall"],
            "uwar":test_cls_result["macro avg"]["recall"],
        }