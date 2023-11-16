import argparse
import datetime
import os
import random
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import Adam

from torch.utils.tensorboard import SummaryWriter

import logging
from generate_data import Stock
import models
logging.basicConfig(level=logging.DEBUG)
# Disable all messages from matplotlib
logging.getLogger('matplotlib').setLevel(logging.CRITICAL)

log = logging.getLogger(__name__)


seed = 46
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# to get the same results, but makes training times longer
"""if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False"""
    
METRICS_VALUE_NDX=0
METRICS_PRED_NDX=1
METRICS_LOSS_NDX=2
METRICS_SIZE = 3

class TrainingApp:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]
            
        parser = argparse.ArgumentParser()
        parser.add_argument('--batch-size',
                            help='Batch size for training',
                            default=128,
                            type=int)
        
        parser.add_argument('--sequence-length',
                            help='Lenght of values for the input',
                            default=30,
                            type=int)
        
        parser.add_argument('--future-time-steps',
                            help='Length of predicted values',
                            default=14,
                            type=int)
        
        parser.add_argument('--include-input',
                            help='Include the input values in the target tensor',
                            default='store_false')
        
        parser.add_argument('--epochs',
                            help='Number of epochs to train',
                            default=100,
                            type=int)
        
        parser.add_argument('--tb-prefix',
                            default='LSTM',
                            help="Data prefix to use for Tensorboard run.",
                            )

        parser.add_argument('comment',
                            help="Comment suffix for Tensorboard run.",
                            nargs='?',
                            default='StockComposedCNN',
                            )
        
        self.args = parser.parse_args(sys_argv)
        self.includeInput = self.args.include_input
        self.path = 'E:Data/AMZN_10.csv'
        self.totalTrainingSamples_count = 0
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        
        self.dataset = self.initDataset()
        self.split_index = int(len(self.dataset)* 0.85)
        
        self.model = self.initModel()
        self.optimizer = self.initOptimizer()
        
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
        self.writer_val = self.initTensorboardWriters("val")
        self.writer_trn = self.initTensorboardWriters("train")
        
    
    def initDataset(self, idx=None, isVal=False):
        sequence_length = self.args.sequence_length
        future_time_steps = self.args.future_time_steps
        self.includeInput = self.args.include_input
        
        
        dataset = Stock(path=self.path, sequence_length=sequence_length, future_time_steps=future_time_steps, idx=idx, isVal=isVal, includeInput=self.includeInput)
        
        return dataset
    
    
    def initLoaders(self):
        batch_size = self.args.batch_size
        train_dataset = self.initDataset(self.split_index, False)
        val_dataset = self.initDataset(self.split_index, True)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        

        return train_loader, val_loader
    
    def initModel(self):
        #model = models.Sequence()# it overfits a lot
        # model = models.Sequence2()
        #model = models.LSTMPredictor()
        #model = models.LSTMPredictor(input_size=1, hidden_size= 64, num_layers=4, dropout_prob=0.40)
        #model = models.BidirectionalLSTM(input_size=1, hidden_size= 32, num_layers=2, dropout_prob=0.20)
        #model = models.BidirectionalLSTMComplex(input_size=1, hidden_size=32, num_layers=3, dropout_prob=0.35)
        #model = models.StockPriceCNN(input_length=self.args.sequence_length, input_channels=1, output_length=self.args.future_time_steps)
        model = models.StockComposedCNN(input_length=self.args.sequence_length, input_channels=16, output_length=self.args.future_time_steps)
        
        param_num = [p.numel() for p in model.parameters()]
        if self.use_cuda:
            log.info("Time {}, Using CUDA; {} devices.".format(datetime.datetime.now(), torch.cuda.device_count()))
            log.info("Total paramteters {}; List of parameters {}".format(sum(param_num), param_num))
            model = model.to(self.device)
            model.double()
        
        return model
            
    def initOptimizer(self):
        return Adam(self.model.parameters(), lr=3e-4, weight_decay=0.001)
    
    def initTensorboardWriters(self, phase):
        log_dir = os.path.join('runs', self.args.tb_prefix, self.time_str)
        if phase == "train":
            writer = SummaryWriter(log_dir=log_dir + '-trn-' + self.args.comment)
        elif phase == "val":
            writer = SummaryWriter(log_dir=log_dir + '-val-' + self.args.comment)
        return writer

    
    


    def main(self):
        train_loader, val_loader = self.initLoaders()
        log.info("Train size {}; Val size {}".format(len(train_loader.dataset), len(val_loader.dataset)))
        
        self.validation_cadence = 2
        self.plot_cadence = self.validation_cadence * 2
        for epoch in range(1, self.args.epochs +1):
            log.info(f"Time {datetime.datetime.now()}, Epoch {epoch}")
            
            train_metrics = self.TrainingEpoch(epoch, train_loader)
            self.logMetrics(epoch, "trn", train_metrics)
            
            if epoch == 1 or epoch % self.validation_cadence == 0:
                val_metrics = self.TrainingEpoch(epoch, val_loader)
                self.logMetrics(epoch, "val", val_metrics)
                
                if epoch == 1 or epoch % self.plot_cadence == 0:
                    self.plot_stock(val_loader, num_graphs=3, epoch=epoch)
            
        # close the tensorboard writers
        self.writer_val.close()
        self.writer_trn.close()
            
          
          
    def TrainingEpoch(self, epoch, train_loader):
        if self.includeInput == True:
            batch_metrics_train = torch.zeros(METRICS_SIZE, len(train_loader.dataset), self.args.future_time_steps + self.args.sequence_length, device=self.device)
        else:
            batch_metrics_train = torch.zeros(METRICS_SIZE, len(train_loader.dataset), self.args.future_time_steps, device=self.device)
            
        self.model.train()
        # log.info(f"Time {datetime.datetime.now()}, Epoch {epoch}")
        
        for batch_idx, batch_tuple in enumerate(train_loader):
            self.optimizer.zero_grad()
            
            loss_var = self.computeBatchLoss(
                batch_idx,
                batch_tuple,
                train_loader.batch_size,
                batch_metrics_train,
            )
            loss_var.backward()
            self.optimizer.step()
            
        self.totalTrainingSamples_count += len(train_loader.dataset)
        
        return batch_metrics_train.to('cpu')
    
    def ValidationEpoch(self, epoch, val_loader):
        with torch.no_grad():
            self.model.eval()
            
            
            if self.includeInput == True:
                batch_metrics_val = torch.zeros(METRICS_SIZE, len(val_loader.dataset), self.args.future_time_steps + self.args.sequence_length ,  device=self.device)
            else:
                batch_metrics_val = torch.zeros(METRICS_SIZE, len(val_loader.dataset), self.args.future_time_steps,  device=self.device)

        
            # log.info(f"Time {datetime.datetime.now()}, Epoch {epoch}")
        
            for batch_idx, batch_tuple in enumerate(val_loader):
                self.optimizer.zero_grad()
                
                self.computeBatchLoss(
                    batch_idx,
                    batch_tuple,
                    val_loader.batch_size,
                    batch_metrics_val,
                )
                        
        return batch_metrics_val.to('cpu')
    
    
    def computeBatchLoss(self, batch_idx, batch_tuple, batch_size, batch_metrics):
        
        x_batch, y_batch = batch_tuple
        x_batch = x_batch.permute(2, 0, 1)
        
        x_batch_gpu = x_batch.to(self.device)
        y_batch_gpu = y_batch.to(self.device)
                
        outputs = self.model(x_batch_gpu, self.args.future_time_steps)

        
        loss_func = nn.MSELoss(reduction='none')
        
        loss_gpu = loss_func(
            outputs,
            y_batch_gpu,
        )
        
        start_idx = batch_idx * batch_size
        end_idx = start_idx + y_batch.size(0)

        batch_metrics[METRICS_VALUE_NDX, start_idx:end_idx, :] = y_batch_gpu
        
        if self.includeInput:
            batch_metrics[METRICS_PRED_NDX, start_idx:end_idx, :] = outputs[:,:y_batch.shape[1]]
        else:
            batch_metrics[METRICS_PRED_NDX, start_idx:end_idx, :] = outputs
        
        batch_metrics[METRICS_LOSS_NDX, start_idx:end_idx, :] = loss_gpu
        return loss_gpu.mean()
      
    def logMetrics(
        self,
        epoch,
        phase,
        metrics,
    ):
        """
        Root Mean Squared Error (RMSE):
            rmse = torch.sqrt(loss)
        
        Mean Absolute Error (MAE):
            mae = torch.mean(torch.abs(predictions - targets))
        
        Percentage Explained
            variance_targets = torch.var(targets)
            percentage_explained = 1 - loss / variance_targets
        
        Prediction Direction Accuracy (PDA):
            correct_direction = ((predictions[1:] - predictions[:-1]) * (targets[1:] - targets[:-1])) > 0
            pda = torch.mean(correct_direction.float())
        
        Coefficient of Determination (R²)
            variance_targets = torch.var(targets)
            r_squared = 1 - (loss / variance_targets)
        
        Explained Variance Score
            explained_variance = 1 - (loss / variance_targets)
        
        Normalized RMSE (NRMSE):
            range_targets = torch.max(targets) - torch.min(targets)
            nrmse = rmse / range_targets

        """
        loss = metrics[METRICS_LOSS_NDX].mean()
        rmse = torch.sqrt(loss)
        mae = torch.mean(torch.abs(metrics[METRICS_PRED_NDX] - metrics[METRICS_VALUE_NDX]))
        variance_targets = torch.var(metrics[METRICS_VALUE_NDX])
        # percentage_explained = 1 - loss / variance_targets
        correct_direction = ((metrics[METRICS_PRED_NDX, 1:] - metrics[METRICS_PRED_NDX, :-1]) * (metrics[METRICS_VALUE_NDX, 1:] - metrics[METRICS_VALUE_NDX, :-1])) > 0
        pda = torch.mean(correct_direction.float())
        r_squared = 1 - (loss / variance_targets)
        # explained_variance = 1 - (loss / variance_targets)
        range_targets = torch.max(metrics[METRICS_VALUE_NDX]) - torch.min(metrics[METRICS_VALUE_NDX])
        nrmse = rmse / range_targets
        
        metrics_dict = {}
        metrics_dict['loss'] = metrics[METRICS_LOSS_NDX].mean()
        metrics_dict['RMSE'] = rmse
        metrics_dict['MAE'] = mae
        # metrics_dict['Percentage_Explained'] = percentage_explained
        metrics_dict['PDA'] = pda 
        metrics_dict['r_squared'] = r_squared
        # metrics_dict['explained_variance '] = explained_variance 
        metrics_dict['NRMSE'] = nrmse
        
        time = datetime.datetime.now()

        log.info(
            ("Time {} | Epoch {} | Phase {} | Loss: {:.4f} | RMSE: {:.4f} | MAE: {:.4f} |"
            + " PDA: {:.4f} | r_squared: {:.4f} |"
            + " NRMSE: {:.4f}").format(
            time, epoch, phase,
            metrics_dict['loss'].item(), metrics_dict['RMSE'].item(),
            metrics_dict['MAE'].item(),
            metrics_dict['PDA'].item(), metrics_dict['r_squared'].item(),
            metrics_dict['NRMSE'].item()
            )
        )
        
        writer = getattr(self, 'writer_' + phase)
        
        for key, value in metrics_dict.items():
            writer.add_scalar(key, value, self.totalTrainingSamples_count)

    def plot_stock(self, dataloader, num_graphs=3, epoch=0):
        for batch_idx, batch_tuple in enumerate(dataloader):
            x_batch, y_batch = batch_tuple
            x_batch = x_batch.permute(2, 0, 1)
            
            x_batch_gpu = x_batch.to(self.device)
            y_batch_gpu = y_batch.to(self.device)
            
            with torch.no_grad():
                outputs = self.model(x_batch_gpu, self.args.future_time_steps)

        
                loss_func = nn.MSELoss(reduction='none')
                
                loss_gpu = loss_func(
                    outputs,
                    y_batch_gpu,
                )
                log.info("Mean Loss of the batch: {}".format(loss_gpu.mean()))
            

            colors = [['r', 'g'], ['b', 'y'], ['k', 'm']]
            for i in range(num_graphs):
                plt.plot(np.arange(outputs[0].shape[0]), outputs[10 * i].cpu().numpy(), colors[i][0], linewidth = 2.0, label='Predicted Close{}'.format(10*i))
                plt.plot(np.arange(outputs[0].shape[0]), y_batch[10 * i].cpu().numpy(), colors[i][1] + ':', 
                linewidth = 2.0, label='Actual Close{}'.format(10*i))
            
            plt.xlabel('Day')
            plt.ylabel('Close')
            plt.legend()
            self.save_plot(epoch)
            plt.close()
            break

    def save_plot(self, epoch):
        # Define the main folder where plots will be saved
        main_folder = "graphs"

        

        # Create a subfolder based on the current time
        subfolder = os.path.join(main_folder, self.time_str)

        # Check if the subfolder exists; if not, create it
        if not os.path.exists(subfolder):
            os.makedirs(subfolder)

        # Save the plot in the subfolder
        plot_filename = os.path.join(subfolder, '{}{}.pdf'.format(self.args.comment, epoch))
        plt.savefig(plot_filename)
        print("Saved plot at: {}".format(plot_filename))

    
    
    
if __name__ == '__main__':
    app = TrainingApp()  # クラスのインスタンスを作成
    app.main()  # インスタンスからメソッドを呼び出す