import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn.utils import weight_norm
'''
Model definition:
1.CNNLSTMNet_Fusion: Full model.
2.CNNLSTMNet_Fusion_wo_mass: The mass_fc layer has been removed, and the input dimension of the LSTM has been adjusted to include only the dimension of acoustic features.
3.CNNLSTMNet_Fusion_wo_lstm: The LSTM and attention layers have been removed. After feature fusion, global average pooling is used to aggregate the temporal dimension (seq_len), 
which then directly feeds into the fully connected layer. Note that the input dimension to the fully connected layer now corresponds to the fused dimension.
4.CNNLSTMNet_Fusion_wo_attn: Retain the LSTM but remove the attention layer. Replace the original attention-weighted summation with a simple average summation.(use the final state is okay too)
5.
'''
class CNNLSTMNet_Fusion(nn.Module):
    def __init__(self, ac_size=1500, num_classes=3, input_channels=1):
        super(CNNLSTMNet_Fusion, self).__init__()
        ac_high_dimension_number = 256
        mass_dim_number = 4
        lstm_dim = 256

        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        length = ac_size // 2**3   # 187
        conv_output_size = 128 * length
        self.fc1 = nn.Linear(conv_output_size, ac_high_dimension_number)
        self.mass_fc = nn.Sequential(
            nn.Linear(1, 2*mass_dim_number),
            nn.ReLU(),
            nn.Linear(2*mass_dim_number, mass_dim_number)
        )
        fusion_dim = ac_high_dimension_number + mass_dim_number
        self.lstm = nn.LSTM(fusion_dim, hidden_size=lstm_dim, num_layers=2,
                            batch_first=True, dropout=0.2)

        self.attention = nn.Sequential(
            nn.Linear(lstm_dim, lstm_dim),
            nn.Tanh(),
            nn.Linear(lstm_dim, 1)
        )
        self.fc_out = nn.Linear(lstm_dim, num_classes)

    def forward(self, x_ac, x_mass):
        batch_size, channels, seq_len, feat_size = x_ac.size()
        # (batch, channels, 100, 1500) -> (batch*100, channels, 1500)
        x_ac = x_ac.view(batch_size * seq_len, channels, feat_size)
        x_ac = self.pool(F.relu(self.conv1(x_ac)))
        x_ac = self.pool(F.relu(self.conv2(x_ac)))
        x_ac = self.pool(F.relu(self.conv3(x_ac)))
        # Flatten
        x_ac = x_ac.view(batch_size, seq_len, -1)
        x_ac = F.relu(self.fc1(x_ac))  # (batch, 100, 256)
        # mass mapping
        x_mass = F.relu(self.mass_fc(x_mass))  # (batch, 100, 4)
        # frame-level fusion
        x = torch.cat([x_ac, x_mass], dim=-1)  # (batch, 100, 128+4)
        # LSTM + attention
        lstm_out, _ = self.lstm(x) 
        weights = F.softmax(self.attention(lstm_out), dim=1) 
        feat = torch.sum(lstm_out * weights, dim=1)  # (batch, 128)
        out = F.relu(self.fc_out(feat))
        return out

class CNNLSTMNet_Fusion_wo_mass(nn.Module):
    def __init__(self, ac_size=1500, num_classes=3, input_channels=1):
        super(CNNLSTMNet_Fusion_wo_mass, self).__init__()
        ac_high_dimension_number = 256
        # Remove mass_dim_number
        lstm_dim = 256

        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        length = ac_size // 2**3   # 187
        conv_output_size = 128 * length
        self.fc1 = nn.Linear(conv_output_size, ac_high_dimension_number)
        
        # Remove mass_fc
        
        # fusion_dim now only contains the dimension of ac
        fusion_dim = ac_high_dimension_number 
        self.lstm = nn.LSTM(fusion_dim, hidden_size=lstm_dim, num_layers=2,
                            batch_first=True, dropout=0.2)

        self.attention = nn.Sequential(
            nn.Linear(lstm_dim, lstm_dim),
            nn.Tanh(),
            nn.Linear(lstm_dim, 1)
        )
        self.fc_out = nn.Linear(lstm_dim, num_classes)

    def forward(self, x_ac, x_mass=None):
        # To preserve parameter bits for compatibility, but not to use them.
        batch_size, channels, seq_len, feat_size = x_ac.size()
        x_ac = x_ac.view(batch_size * seq_len, channels, feat_size)
        x_ac = self.pool(F.relu(self.conv1(x_ac)))
        x_ac = self.pool(F.relu(self.conv2(x_ac)))
        x_ac = self.pool(F.relu(self.conv3(x_ac)))
        x_ac = x_ac.view(batch_size, seq_len, -1)
        x_ac = F.relu(self.fc1(x_ac))  # (batch, 100, 256)
        
        # Remove mass mapping and Concatenation
        # x = torch.cat([x_ac, x_mass], dim=-1) 
        x = x_ac # Directly use the AC features as input to the LSTM

        lstm_out, _ = self.lstm(x)
        weights = F.softmax(self.attention(lstm_out), dim=1)
        feat = torch.sum(lstm_out * weights, dim=1)
        out = F.relu(self.fc_out(feat))
        return out

class CNNLSTMNet_Fusion_wo_lstm(nn.Module):
    def __init__(self, ac_size=1500, num_classes=3, input_channels=1):
        super(CNNLSTMNet_Fusion_wo_lstm, self).__init__()
        ac_high_dimension_number = 256
        mass_dim_number = 4
        # lstm_dim is no longer required, but the output layer needs to know the dimensions of the previous layer.
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        length = ac_size // 2**3
        conv_output_size = 128 * length
        self.fc1 = nn.Linear(conv_output_size, ac_high_dimension_number)
        
        self.mass_fc = nn.Sequential(
            nn.Linear(1, 2*mass_dim_number),
            nn.ReLU(),
            nn.Linear(2*mass_dim_number, mass_dim_number)
        )
        
        fusion_dim = ac_high_dimension_number + mass_dim_number
        # Remove LSTM and Attention
        # The input dimension of the final FC layer becomes fusion_dim (since there is no LSTM to alter the dimension).
        self.fc_out = nn.Linear(fusion_dim, num_classes)

    def forward(self, x_ac, x_mass):
        batch_size, channels, seq_len, feat_size = x_ac.size()
        x_ac = x_ac.view(batch_size * seq_len, channels, feat_size)
        x_ac = self.pool(F.relu(self.conv1(x_ac)))
        x_ac = self.pool(F.relu(self.conv2(x_ac)))
        x_ac = self.pool(F.relu(self.conv3(x_ac)))
        x_ac = x_ac.view(batch_size, seq_len, -1)
        x_ac = F.relu(self.fc1(x_ac))
        
        x_mass = F.relu(self.mass_fc(x_mass))
        x = torch.cat([x_ac, x_mass], dim=-1)  # (batch, 100, 260)
        
        # Remove LSTM and Attention
        # Use Global Average Pooling to aggregate across the temporal dimension: (batch, 100, 260) -> (batch, 260)
        feat = torch.mean(x, dim=1)         
        out = F.relu(self.fc_out(feat))
        return out

class CNNLSTMNet_Fusion_wo_attn(nn.Module):
    def __init__(self, ac_size=1500, num_classes=3, input_channels=1):
        super(CNNLSTMNet_Fusion_wo_attn, self).__init__()
        ac_high_dimension_number = 256
        mass_dim_number = 4
        lstm_dim = 256

        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        length = ac_size // 2**3
        conv_output_size = 128 * length
        self.fc1 = nn.Linear(conv_output_size, ac_high_dimension_number)
        self.mass_fc = nn.Sequential(
            nn.Linear(1, 2*mass_dim_number),
            nn.ReLU(),
            nn.Linear(2*mass_dim_number, mass_dim_number)
        )
        fusion_dim = ac_high_dimension_number + mass_dim_number
        self.lstm = nn.LSTM(fusion_dim, hidden_size=lstm_dim, num_layers=2,
                            batch_first=True, dropout=0.2)

        # Remove Attention

        self.fc_out = nn.Linear(lstm_dim, num_classes)

    def forward(self, x_ac, x_mass):
        batch_size, channels, seq_len, feat_size = x_ac.size()
        x_ac = x_ac.view(batch_size * seq_len, channels, feat_size)
        x_ac = self.pool(F.relu(self.conv1(x_ac)))
        x_ac = self.pool(F.relu(self.conv2(x_ac)))
        x_ac = self.pool(F.relu(self.conv3(x_ac)))
        x_ac = x_ac.view(batch_size, seq_len, -1)
        x_ac = F.relu(self.fc1(x_ac))
        
        x_mass = F.relu(self.mass_fc(x_mass))
        x = torch.cat([x_ac, x_mass], dim=-1)
        lstm_out, _ = self.lstm(x) 
        
        # Remove Attention calculate
        # Perform mean pooling directly on the LSTM output as features.
        # may also choose to take the last moment: feat = lstm_out[:, -1, :]
        feat = torch.mean(lstm_out, dim=1) # (batch, 128)      
        out = F.relu(self.fc_out(feat))
        return out
    
class MidFusionCNNLSTMClassifier(nn.Module): 
    def __init__(self, ac_feat_size=1500, mass_feat_size=1, num_classes=3):
        super(MidFusionCNNLSTMClassifier, self).__init__()
        
        ac_high_dimension_number = 256
        mass_high_dimension_number = 8

        # Acoustic Feature Extractor
        self.ac_conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.ac_conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.ac_conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.ac_pool = nn.MaxPool1d(2)
        
        length = ac_feat_size // 2**3
        ac_conv_output_size = 128 * length
        self.ac_fc1 = nn.Linear(ac_conv_output_size, ac_high_dimension_number)

        # Quality Feature Extractor
        self.mass_fc1 = nn.Linear(mass_feat_size, mass_high_dimension_number)

        # Integrated downstream task module
        fused_feature_size = ac_high_dimension_number + mass_high_dimension_number
        self.lstm = nn.LSTM(fused_feature_size, hidden_size=128, num_layers=2, batch_first=True, dropout=0.2)
        self.attention = nn.Linear(128, 1)
        
        self.fc_out = nn.Linear(128, num_classes)

    def forward(self, x_ac, x_mass):
        # x_ac input shape: (batch, 1, 100, 1500) from dataloader
        batch_size, channels, seq_len, feat_size = x_ac.size()
        x_ac = x_ac.view(batch_size * seq_len, channels, feat_size)
        
        x_ac = self.ac_pool(F.relu(self.ac_conv1(x_ac)))
        x_ac = self.ac_pool(F.relu(self.ac_conv2(x_ac)))
        x_ac = self.ac_pool(F.relu(self.ac_conv3(x_ac)))
        x_ac = x_ac.view(batch_size, seq_len, -1)
        features_ac = F.relu(self.ac_fc1(x_ac)) # (batch, 100, 256)

        # mass input shape: (batch, 100, 1)
        features_mass = F.relu(self.mass_fc1(x_mass)) # (batch, 100, 8)
        
        # Mid-level integration
        fused_features = torch.cat((features_ac, features_mass), dim=2)

        # Downstream tasks
        lstm_out, (hn, cn) = self.lstm(fused_features)
        weights = F.softmax(self.attention(lstm_out), dim=1)
        averaged_lstm_out = torch.sum(lstm_out * weights, dim=1)
        
        # Output classification logits
        output = self.fc_out(averaged_lstm_out)
        return output
    
class UNet2D(nn.Module):
    def __init__(self):
        super().__init__()
        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
        self.enc1 = conv_block(1, 16)
        self.pool1 = nn.MaxPool2d(2)       # → (16,50,750)
        self.enc2 = conv_block(16, 32)
        self.pool2 = nn.MaxPool2d(2)       # → (32,25,375)
        self.enc3 = conv_block(32, 64)

        self.up2  = nn.ConvTranspose2d(64, 32, 2, stride=2)  # → (32,50,750)
        self.dec2 = conv_block(32+32, 32)
        self.up1  = nn.ConvTranspose2d(32, 16, 2, stride=2)  # → (16,100,1500)
        self.dec1 = conv_block(16+16, 16)

        self.outc = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        # x: [B,1,100,1500]
        e1 = self.enc1(x)         # (B,16,100,1500)
        p1 = self.pool1(e1)       # (B,16,50,750)
        e2 = self.enc2(p1)        # (B,32,50,750)
        p2 = self.pool2(e2)       # (B,32,25,375)
        e3 = self.enc3(p2)        # (B,64,25,375)

        u2 = self.up2(e3)         # (B,32,50,750)
        u2 = torch.cat([u2, e2], dim=1)  # (B,64,50,750)
        d2 = self.dec2(u2)        # (B,32,50,750)

        u1 = self.up1(d2)         # (B,16,100,1500)
        u1 = torch.cat([u1, e1], dim=1)  # (B,32,100,1500)
        d1 = self.dec1(u1)        # (B,16,100,1500)

        out = self.outc(d1)       # (B,1,100,1500)
        return out
    
class ResNet18(nn.Module):
    def __init__(self, ac_size=1500, num_classes=3, input_channels=1, pretrained=False):
        super(ResNet18, self).__init__()
        
        # 1. Acoustics Branch
        self.resnet = models.resnet18(weights=None)
        if pretrained == True:
            self.resnet.load_state_dict(torch.load("./resnet18-f37072fd.pth"))
        # Modify the first layer to adapt to a single channel
        self.resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Remove original FC
        num_ftrs = self.resnet.fc.in_features  # 512
        self.resnet.fc = nn.Identity()
        
        # 2. Mass Branch。
        mass_seq_len = 100
        mass_input_dim = mass_seq_len * 1
        mass_hidden_dim = 32
        
        self.mass_fc = nn.Sequential(
            nn.Linear(mass_input_dim, mass_hidden_dim), # (Batch, 100) -> (Batch, 32)
            nn.ReLU(),
            nn.Linear(mass_hidden_dim, mass_hidden_dim)
        )
        
        # 3. Integration Layer
        # Fusion: ResNet Features(512) + Mass Features(32)
        fusion_dim = num_ftrs + mass_hidden_dim
        self.fc_out = nn.Linear(fusion_dim, num_classes)

    def forward(self, x_ac, x_mass):
        # x_ac: (batch, 1, 100, 1500)
        batch_size = x_ac.size(0)
        
        # Acoustic Forward
        ac_feat = self.resnet(x_ac)  # (batch, 512)
        
        # Mass Forward
        # x_mass: (batch, 100) or (batch, 100, 1)
        # Flatten it
        x_mass = x_mass.view(batch_size, -1) # (batch, 100)
        
        mass_feat = self.mass_fc(x_mass)     # (batch, 32)
        
        # Fusion
        fusion = torch.cat([ac_feat, mass_feat], dim=1) # (batch, 512 + 32)
        out = self.fc_out(fusion)
        return out

class ResNet18_pt(ResNet18):
    def __init__(self, ac_size=1500, num_classes=3, input_channels=1):
        # extends ResNet18 with pretrained=True
        super(ResNet18_pt, self).__init__(ac_size, num_classes, input_channels, pretrained=True)

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TCN(nn.Module):
    def __init__(self, ac_size=1500, num_classes=3, input_channels=1):
        super(TCN, self).__init__()
        # Reuse the feature extraction component of CNNLSTMNet_Fusion
        self.ac_conv_net = CNNLSTMNet_Fusion_wo_lstm(ac_size, num_classes, input_channels)
        
        # Redefine the FC and Mass sections to maintain consistency.
        ac_high_dimension_number = 256
        mass_dim_number = 4
        
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        length = ac_size // 2**3
        self.fc1 = nn.Linear(128 * length, ac_high_dimension_number)
        
        self.mass_fc = nn.Sequential(
            nn.Linear(1, 2*mass_dim_number),
            nn.ReLU(),
            nn.Linear(2*mass_dim_number, mass_dim_number)
        )
        
        # TCN replaces LSTM
        fusion_dim = ac_high_dimension_number + mass_dim_number
        # TCN Channel Configuration: [256, 256, 256] indicates a 3-layer TCN Block
        self.tcn = TemporalConvNet(num_inputs=fusion_dim, num_channels=[256, 256, 256], kernel_size=3, dropout=0.2)
        
        self.attention = nn.Sequential(
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )
        self.fc_out = nn.Linear(256, num_classes)

    def forward(self, x_ac, x_mass):
        batch_size, channels, seq_len, feat_size = x_ac.size()
        # CNN feature extraction
        x_ac = x_ac.view(batch_size * seq_len, channels, feat_size)
        x_ac = self.pool(F.relu(self.conv1(x_ac)))
        x_ac = self.pool(F.relu(self.conv2(x_ac)))
        x_ac = self.pool(F.relu(self.conv3(x_ac)))
        x_ac = x_ac.view(batch_size, seq_len, -1)
        x_ac = F.relu(self.fc1(x_ac)) # (batch, 100, 256)
        
        # Mass
        x_mass = F.relu(self.mass_fc(x_mass)) # (batch, 100, 4)
        
        # Fusion
        x = torch.cat([x_ac, x_mass], dim=-1) # (batch, 100, 260)
        
        # TCN forward
        # TCN expects input (N, C_in, L), so we place the channels in the middle
        x = x.transpose(1, 2) # (batch, 260, 100)
        tcn_out = self.tcn(x) # (batch, 256, 100)
        tcn_out = tcn_out.transpose(1, 2) # (batch, 100, 256)
        
        # Attention
        weights = F.softmax(self.attention(tcn_out), dim=1)
        feat = torch.sum(tcn_out * weights, dim=1) # (batch, 256)
        
        out = F.relu(self.fc_out(feat))
        return out


# Static Model Based on Channel Attention
class SELayer(nn.Module):
    """ Squeeze-and-Excitation Layer (Channel Attention) """
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (Batch, Channel, Length)
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class static_channel(nn.Module):
    def __init__(self, ac_size=1500, num_classes=3, input_channels=1):
        super(static_channel, self).__init__()
        ac_high_dimension_number = 256
        mass_dim_number = 4
        
        # 1. Basic CNN Backbone
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        
        length = ac_size // 2**3  # 1500 -> 187
        conv_output_size = 128 * length
        self.fc1 = nn.Linear(conv_output_size, ac_high_dimension_number)
        
        # 2. Mass mapping
        self.mass_fc = nn.Sequential(
            nn.Linear(1, 2*mass_dim_number),
            nn.ReLU(),
            nn.Linear(2*mass_dim_number, mass_dim_number)
        )
        
        # 3. Static Processing
        fusion_dim = ac_high_dimension_number + mass_dim_number
        
        self.fc_out = nn.Linear(fusion_dim, num_classes)

    def forward(self, x_ac, x_mass):
        # Input shape: 
        # x_ac: (Batch, 1, 100, 1500)
        # x_mass: (Batch, 100, 1)
        
        batch_size, channels, seq_len, feat_size = x_ac.size()
        
        # Flatten Batch and Time dimensions immediately.
        # Treat every frame as an independent sample.
        # New shape: (Batch * 100, 1, 1500)
        x_ac = x_ac.view(batch_size * seq_len, channels, feat_size)
        
        # Process Mass: (Batch, 100, 1) -> (Batch * 100, 1)
        x_mass = x_mass.view(batch_size * seq_len, -1) 

        # CNN Forward (Same as before, but now runs on B*T samples)
        x_ac = self.pool(F.relu(self.conv1(x_ac)))
        x_ac = self.pool(F.relu(self.conv2(x_ac)))
        x_ac = self.pool(F.relu(self.conv3(x_ac)))
        
        # Flatten features: (B*T, 128, 187) -> (B*T, 128*187)
        x_ac = x_ac.view(x_ac.size(0), -1)
        x_ac = F.relu(self.fc1(x_ac)) # (B*T, 256)
        
        # Mass Forward
        x_mass = F.relu(self.mass_fc(x_mass)) # (B*T, 4)
        
        # Fusion
        x = torch.cat([x_ac, x_mass], dim=-1) # (B*T, 260)
        
        # perform regression/classification on EACH frame.
        out = self.fc_out(x) # (B*T, num_classes) -> (Batch * 100, num_classes)
        
        return out
    
# MVUE (Minimum Variance Unbiased Estimator Net)
class MVUE(nn.Module):
    def __init__(self, ac_size=1500, num_classes=3, input_channels=1):
        """
        MVUE Implementation Logic:
        It is not merely a simple weighting, but rather explicitly learning the ‘variance’ of each feature at every time step.
        According to the MVUE theory, w_i ~ 1/sigma_i^2
        """
        super(MVUE, self).__init__()
        ac_high_dimension_number = 256
        mass_dim_number = 4
        
        # 1. Basic CNN Backbone
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        length = ac_size // 2**3
        self.fc1 = nn.Linear(128 * length, ac_high_dimension_number)
        
        # 2. Mass mapping
        self.mass_fc = nn.Sequential(
            nn.Linear(1, 2*mass_dim_number),
            nn.ReLU(),
            nn.Linear(2*mass_dim_number, mass_dim_number)
        )
        
        # 3. MVUE Core Module
        # Assume that each time step after passing through the CNN constitutes an independent observation.
        # Estimate the observed variance(scalar)
        fusion_dim = ac_high_dimension_number + mass_dim_number
        
        self.variance_estimator = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Softplus() # Ensure variance > 0
        )
        
        self.fc_out = nn.Linear(fusion_dim, num_classes)

    def forward(self, x_ac, x_mass):
        # x_ac: (batch, 1, 100, 1500)
        # x_mass: (batch, 100)
        
        batch_size, channels, seq_len, feat_size = x_ac.size()
        
        # Feature Extraction
        x_ac = x_ac.view(batch_size * seq_len, channels, feat_size)
        x_ac = self.pool(F.relu(self.conv1(x_ac)))
        x_ac = self.pool(F.relu(self.conv2(x_ac)))
        x_ac = self.pool(F.relu(self.conv3(x_ac)))
        x_ac = x_ac.view(batch_size, seq_len, -1)
        x_ac = F.relu(self.fc1(x_ac)) # (B, 100, 256)
        
        # Mass Fusion
        # Ensure x_mass dimension adaptation
        if x_mass.dim() == 1:
            x_mass = x_mass.unsqueeze(-1) # (B, 100) -> (B, 100, 1)
        elif x_mass.dim() == 2 and x_mass.shape[1] == seq_len:
             x_mass = x_mass.unsqueeze(-1) # (B, 100) -> (B, 100, 1)
             
        x_mass = F.relu(self.mass_fc(x_mass)) # (B, 100, 4)
        x = torch.cat([x_ac, x_mass], dim=-1) # (B, 100, 260)
        
        # MVUE Aggregation
        # 1. Estimate the variance at each time step sigma^2
        variances = self.variance_estimator(x) + 1e-6 # (B, 100, 1), Estimate the variance at each time step
        
        # 2. Estimate the variance at each time step
        # w_i = (1 / var_i) / Sum(1 / var_j)
        inverse_variance = 1.0 / variances
        weights = inverse_variance / torch.sum(inverse_variance, dim=1, keepdim=True)
        
        # 3. Weighted Average
        # The “true” feature vector estimated by MVUE
        feat = torch.sum(x * weights, dim=1) # (B, 260)
        
        out = F.relu(self.fc_out(feat))
        return out