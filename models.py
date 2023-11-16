import torch.nn as nn
import torch
import math

class Sequence(nn.Module):
    def __init__(self, n_hidden=51):
        super(Sequence, self).__init__()
        self.n_hidden = n_hidden
        
        
        self.lstm1 = nn.LSTMCell(1, self.n_hidden)
        self.lstm2 = nn.LSTMCell(self.n_hidden, self.n_hidden)
        self.linear = nn.Linear(self.n_hidden, 1)
        

    def forward(self, x, future=0):
        device = next(self.parameters()).device
        outputs = []
        x.double()
        h_t = torch.zeros(x.size(1), self.n_hidden, dtype=torch.double).to(device)
        c_t = torch.zeros(x.size(1), self.n_hidden, dtype=torch.double).to(device)
        h_t2 = torch.zeros(x.size(1), self.n_hidden, dtype=torch.double).to(device)
        c_t2 = torch.zeros(x.size(1), self.n_hidden, dtype=torch.double).to(device)
        

        for time_step in range(x.size(0)):
            h_t, c_t = self.lstm1(x[time_step], (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]

        for i in range(future):  # if we should predict the future
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        
        outputs = torch.cat(outputs, dim=1)

        return outputs
    
    
class Sequence2(nn.Module):
    def __init__(self, n_hidden=2, dropout_prob=0.2):
        super(Sequence2, self).__init__()
        self.n_hidden = n_hidden
        
        self.lstm1 = nn.LSTMCell(1, self.n_hidden)
        self.lstm2 = nn.LSTMCell(self.n_hidden, self.n_hidden)
        self.linear = nn.Linear(self.n_hidden, 1)
        self.dropout = nn.Dropout(dropout_prob)
        

    def forward(self, x, future=0):
        device = next(self.parameters()).device
        outputs = []
        x.double()
        
        h_t = torch.zeros(x.size(1), self.n_hidden, dtype=torch.double).to(device)
        c_t = torch.zeros(x.size(1), self.n_hidden, dtype=torch.double).to(device)
        h_t2 = torch.zeros(x.size(1), self.n_hidden, dtype=torch.double).to(device)
        c_t2 = torch.zeros(x.size(1), self.n_hidden, dtype=torch.double).to(device)
        

        for time_step in range(x.size(0)):
            h_t, c_t = self.lstm1(x[time_step], (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            
            output = self.dropout(h_t2)
            output = self.linear(output)
            outputs += [output]

        for i in range(future):  # if we should predict the future
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            
            output = self.dropout(h_t2)
            output = self.linear(output)
            outputs += [output]
        
        outputs = torch.cat(outputs, dim=1)

        return outputs
    
    
class LSTMPredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=8, num_layers=2, output_size=1, dropout_prob=0.2):
        super(LSTMPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout_prob)
        
        # create lstm cells list
        self.lstm_cells = nn.ModuleList([nn.LSTMCell(input_size, self.hidden_size)])
        
        # add the lstm cells
        for _ in range(num_layers - 1):
            self.lstm_cells.append(nn.LSTMCell(hidden_size, hidden_size))
            
        # linear layer for output
        self.linear = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, future=0):
        h_t = [torch.zeros(x.size(1), self.hidden_size, dtype=x.dtype, device=x.device)]
        c_t = [torch.zeros(x.size(1), self.hidden_size, dtype=x.dtype, device=x.device)]
        outputs = []    
        
        # loop through the LSTM cells
        for i in range(self.num_layers-1):
            h_t.append(torch.zeros(x.size(1), self.hidden_size, dtype=x.dtype, device=x.device))
            c_t.append(torch.zeros(x.size(1), self.hidden_size, dtype=x.dtype, device=x.device))
         
        for time_step in range(x.size(0)):
            h_t[0], c_t[0] = self.lstm_cells[0](x[time_step], (h_t[0], c_t[0]))
            for i in range(self.num_layers-1):
                h_t[i+1], c_t[i+1] = self.lstm_cells[i+1](h_t[i], (h_t[i+1], c_t[i+1]))
            
            output = self.dropout(h_t[-1])
            output = self.linear(output)
            outputs += [output]
            
        for i in range(future):
            h_t[0], c_t[0] = self.lstm_cells[0](output, (h_t[0], c_t[0]))
            for i in range(self.num_layers-1):
                h_t[i+1], c_t[i+1] = self.lstm_cells[i+1](h_t[i], (h_t[i+1], c_t[i+1]))
            
            output = self.dropout(h_t[-1])
            output = self.linear(output)
            outputs += [output]
        
        outputs = torch.cat(outputs, dim=1)
        return outputs
    
class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=8, num_layers=2, output_size=1, dropout_prob=0.2):
        super(BidirectionalLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout_prob)
        
        # create lstm cells list
        self.forward_lstm = nn.LSTMCell(input_size, self.hidden_size)
        self.backward_lstm = nn.LSTMCell(input_size, self.hidden_size)
        
            
        # linear layer for output
        self.linear = nn.Linear(2 * hidden_size, output_size)
        
    def forward(self, x, future=0):
        sequence_length, batch_size, input_size = x.size()

        device = next(self.parameters()).device
        forward_outputs = []
        backward_outputs = []
        outputs = []
        x.double()
        
        h_t_forward = torch.zeros(batch_size, self.hidden_size, dtype=torch.double).to(device)
        c_t_forward = torch.zeros(batch_size, self.hidden_size, dtype=torch.double).to(device)
        h_t_backward = torch.zeros(batch_size, self.hidden_size, dtype=torch.double).to(device)
        c_t_backward = torch.zeros(batch_size, self.hidden_size, dtype=torch.double).to(device)
        
        
        for i in range(sequence_length):  # if we should predict the future
            forward_variable = i
            backward_variable = future - i
            
            h_t_forward, c_t_forward = self.forward_lstm(x[forward_variable], (h_t_forward, c_t_forward))
            forward_outputs.append(h_t_forward)
            
            h_t_backward, c_t_backward = self.backward_lstm(x[backward_variable], (h_t_backward, c_t_backward))
            backward_outputs.insert(0, h_t_backward)
            
            # Concatenate forward and backward outputs
            h_t_combined = torch.cat((h_t_forward, h_t_backward), dim=1)
        
            # Apply fully connected layer to get the final output
            output = self.linear(h_t_combined)
            outputs += [output]
            
        for i in range(future):  # if we should predict the future
            forward_variable = i
            backward_variable = future - i
            
            h_t_forward, c_t_forward = self.forward_lstm(outputs[-1], (h_t_forward, c_t_forward))
            forward_outputs.append(h_t_forward)
            
            h_t_backward, c_t_backward = self.backward_lstm(outputs[-1], (h_t_backward, c_t_backward))
            backward_outputs.insert(0, h_t_backward)
            
            
            h_t_combined = torch.cat((h_t_forward, h_t_backward), dim=1)
        
            # Apply fully connected layer to get the final output
            output = self.linear(h_t_combined)
            outputs += [output]

        outputs = torch.cat(outputs, dim=1)
        return outputs
    
class BidirectionalLSTMComplex(nn.Module):
    def __init__(self, input_size=1, hidden_size=8, num_layers=2, output_size=1, dropout_prob=0.2):
        super(BidirectionalLSTMComplex, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout_prob)
        
        # create lstm cells list
        self.forward_lstm = nn.ModuleList([nn.LSTMCell(input_size, self.hidden_size)])
        self.backward_lstm = nn.ModuleList([nn.LSTMCell(input_size, self.hidden_size)])
        
        # add the lstm cells
        for _ in range(num_layers - 1):
            self.forward_lstm.append(nn.LSTMCell(hidden_size, hidden_size))
            self.backward_lstm.append(nn.LSTMCell(hidden_size, hidden_size))
            
        # linear layer for output
        self.linear = nn.Linear(2 * hidden_size, output_size)
        
    def forward(self, x, future=0):
        sequence_length, batch_size, input_size = x.size()

        outputs = []
        x.double()
        
        h_t_forward = [torch.zeros(batch_size, self.hidden_size, dtype=x.dtype, device=x.device)]
        c_t_forward = [torch.zeros(batch_size, self.hidden_size, dtype=x.dtype, device=x.device)]
        h_t_backward = [torch.zeros(batch_size, self.hidden_size, dtype=x.dtype, device=x.device)]
        c_t_backward = [torch.zeros(batch_size, self.hidden_size, dtype=x.dtype, device=x.device)]
        
        # loop through the LSTM cells
        for i in range(self.num_layers-1):
            h_t_forward.append(torch.zeros(batch_size, self.hidden_size, dtype=x.dtype, device=x.device))
            c_t_forward.append(torch.zeros(batch_size, self.hidden_size, dtype=x.dtype, device=x.device))
            h_t_backward.append(torch.zeros(batch_size, self.hidden_size, dtype=x.dtype, device=x.device))
            c_t_backward.append(torch.zeros(batch_size, self.hidden_size, dtype=x.dtype, device=x.device))
        
        
        for i in range(sequence_length):  # if we should predict the future
            forward_variable = i
            backward_variable = future - i
            
            h_t_forward[0], c_t_forward[0] = self.forward_lstm[0](x[forward_variable], (h_t_forward[0], c_t_forward[0]))
            for i in range(self.num_layers-1):
                h_t_forward[i+1], c_t_forward[i+1] = self.forward_lstm[i+1](h_t_forward[i], (h_t_forward[i+1], c_t_forward[i+1]))
            
            
            h_t_backward[0], c_t_backward[0] = self.backward_lstm[0](x[backward_variable], (h_t_backward[0], c_t_backward[0]))
            for i in range(self.num_layers-1):
                h_t_backward[i+1], c_t_backward[i+1] = self.backward_lstm[i+1](h_t_backward[i], (h_t_forward[i+1], c_t_backward[i+1]))
                
            h_t_combined = torch.cat((h_t_forward[-1], h_t_backward[-1]), dim=1)
            output = self.dropout(h_t_combined)
            # Apply fully connected layer to get the final output
            output = self.linear(output)
            outputs += [output]
            
        for i in range(future):  # if we should predict the future
            forward_variable = i
            backward_variable = future - i
            
            
            h_t_forward[0], c_t_forward[0] = self.forward_lstm[0](outputs[-1], (h_t_forward[0], c_t_forward[0]))
            for i in range(self.num_layers-1):
                h_t_forward[i+1], c_t_forward[i+1] = self.forward_lstm[i+1](h_t_forward[i], (h_t_forward[i+1], c_t_forward[i+1]))
            
            
            h_t_backward[0], c_t_backward[0] = self.backward_lstm[0](outputs[-1], (h_t_backward[0], c_t_backward[0]))
            for i in range(self.num_layers-1):
                h_t_backward[i+1], c_t_backward[i+1] = self.backward_lstm[i+1](h_t_backward[i], (h_t_forward[i+1], c_t_backward[i+1]))
                
            h_t_combined = torch.cat((h_t_forward[-1], h_t_backward[-1]), dim=1)
            output = self.dropout(h_t_combined)
            # Apply fully connected layer to get the final output
            output = self.linear(output)
            outputs += [output]

        outputs = torch.cat(outputs, dim=1)
        return outputs
    

class StockPriceCNN(nn.Module):
    def __init__(self, input_length=30, input_channels=1, output_length=14):
        super(StockPriceCNN, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=16, kernel_size=3)
        self.conv1_batchnorm = nn.BatchNorm1d(num_features=16)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=64, kernel_size=3)
        self.conv2_batchnorm = nn.BatchNorm1d(num_features=64)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3)
        self.conv3_batchnorm = nn.BatchNorm1d(num_features=128)
        self.activation = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.linear = nn.Linear(128 * 2, output_length)
        
    def forward(self, x, future=0):
        x = x.permute(1, 2, 0)
        x = self.conv1_batchnorm(self.conv1(x))
        x = self.pool(self.activation(x))
        x = self.conv2_batchnorm(self.conv2(x))
        x = self.pool(self.activation(x))
        x = self.conv3_batchnorm(self.conv3(x))
        x = self.pool(self.activation(x))
        
        x = x.view(-1, 128 * 2)
        
        x = self.linear(x)
        return x
    
    
class StockBlock(nn.Module):
    def __init__(self, input_channels=1, conv_channels=16):
        super(StockBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=conv_channels, kernel_size=3, padding=1)
        self.conv1_batchnorm = nn.BatchNorm1d(num_features=conv_channels)
        self.conv2 = nn.Conv1d(in_channels=conv_channels, out_channels=conv_channels, kernel_size=3, padding=1)
        self.conv2_batchnorm = nn.BatchNorm1d(num_features=conv_channels)
        self.activation = nn.ReLU()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {
                nn.Linear,
                nn.Conv3d,
                nn.Conv2d,
                nn.ConvTranspose2d,
                nn.ConvTranspose3d,
            }:
                nn.init.kaiming_normal_(
                    m.weight.data, a=0, mode='fan_out', nonlinearity='relu'
                )
                if m.bias is not None:
                    fan_in, fan_out = \
                        nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)
        
    def forward(self, x):
        x = self.conv1_batchnorm(self.conv1(x))
        x = self.activation(x)
        x = self.conv2_batchnorm(self.conv2(x))
        x = self.activation(x)

        return x
    
class StockComposedCNN(nn.Module):
    def __init__(self, input_length=30, input_channels=32, output_length=14):
        super(StockComposedCNN, self).__init__()
        self.input_channels = input_channels
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=input_channels//2, kernel_size=3, stride=2)
        self.conv1_batchnorm = nn.BatchNorm1d(num_features=input_channels//2)
        self.block1 = StockBlock(input_channels//2, input_channels)
        self.block2 = StockBlock(input_channels, input_channels*2)
        self.block3 = StockBlock(input_channels*2, input_channels*4)
        self.block4 = StockBlock(input_channels*4, input_channels*8)
        
        self.activation = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.linear = nn.Linear(input_channels*8 * 3, output_length)
        
        
    def forward(self, x, future=0):
        x = x.permute(1, 2, 0)
        x = self.conv1_batchnorm(self.conv1(x))
        x = self.block1(self.activation(x))
        x = self.block2(x)
        x = self.pool(self.block3(x))
        x = self.pool(self.block4(x))
        

        x = x.view(-1, self.input_channels * 8 * 3)
        
        x = self.linear(x)
        return x