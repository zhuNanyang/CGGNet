import torch
import torch.nn as nn
from bbbb8888.example.source_and_load.param import source_load_param
from bbbb8888.nn.modules.lstm import LSTM
from bbbb8888.nn.losses import M_Loss
from bbbb8888.nn.utils import initial_parameter
from bbbb8888.nn.model import Model
from bbbb8888.nn.modules.attn import selfattention
from bbbb8888.example.source_and_load.encoder import TransformerEncoder
from bbbb8888.example.source_and_load.attention import AttentionGrandularity
from bbbb8888.example.source_and_load.embed import DataEmbedding
torch.set_default_tensor_type(torch.DoubleTensor)

class power_model(Model):
    def __init__(
        self,
        x: int = 3,
        feature_size: int = source_load_param.lstm["feature_size"][0],
        initial_method=None,
    ):
        super(power_model, self).__init__()

        self.lstm = nn.GRU(
            input_size=x,
            hidden_size=feature_size,
            num_layers=2,
            dropout=0.1,
        )
        self.lstmf = nn.GRU(
            input_size=x,
            hidden_size=feature_size,
            num_layers=2,
            dropout=0.1,
        )
        self.attn = selfattention(feature_size, num_attention_heads=8)
        self.attnf = selfattention(feature_size, num_attention_heads=8)
        self.granularity_attn = AttentionGrandularity(d_model=feature_size, n_head=4)

        self.decoder = nn.Linear(
           640, source_load_param.pred_len, bias=True
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        initial_parameter(self)

    def _forward(self, **inputs):
        input_x = inputs["input_x"]
        input_fx = inputs["input_fine_x"]
        input_x, _ = self.lstm(input_x.squeeze(-2))
        input_x = self.attn(input_x, input_x, input_x)
        input_f = input_x
        input_bf = torch.zeros(
            input_fx.shape[0],
            input_fx.shape[1],
            input_fx.shape[2],
            source_load_param.lstm["feature_size"][0],
        )

        for t, n in enumerate(range(input_fx.shape[-2])):
            input_fn = input_fx[:, :, n, :]
            input_fn, _ = self.lstm(input_fn)
            input_attn = self.attnf(queries=input_x, keys=input_fn, values=input_fn)

            input_f = torch.cat([input_f, input_attn], dim=-1)
            input_bf[:, :, t, :] = input_attn

        input_bf = input_bf.to(input_x.device)
        input_fx = self.granularity_attn(
            queries=input_bf, keys=input_x.unsqueeze(-2), keys_0=input_x.unsqueeze(-2)
        )
        x = torch.cat([input_fx, input_f], dim=-1)
        x = x[:, -1, :]
        x_output = x
        output = self.decoder(x_output)
        return  output.unsqueeze(-1)

    def forward(self, **inputs):
        output = self._forward(**inputs)
        input_y = inputs["input_y"]
        critern = M_Loss()
        l = critern(output, input_y)
        return {"output": output, "input_y": input_y, "loss": l}

    def predict(self, **inputs):
        output = self._forward(**inputs)
        input_y = inputs["input_y"]
        # input_y = input_y.squeeze(-1)
        return {
            "output": output,
            "input_x": inputs["input_x"],
            "input_y": input_y,
        }
class power_ablation1(Model):
    def __init__(
        self,
        x: int = 3,
        feature_size: int = source_load_param.lstm["feature_size"][0],
        initial_method=None,
    ):
        super(power_ablation1, self).__init__()
        self.embed = DataEmbedding(c_in=x, d_model=feature_size)
        self.transformer = TransformerEncoder(d_model=feature_size)

        self.decoder = nn.Linear(
           feature_size, source_load_param.pred_len, bias=True
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        #initial_parameter(self)

    def _forward(self, **inputs):
        input_x = inputs["input_x"]
        input_fx = inputs["input_fine_x"]
        input_x = self.embed(input_x)
        input_x = self.transformer(input_x)
        input_x = input_x[:, -1, :]
        output = self.decoder(input_x)
        return  output.unsqueeze(-1)

    def forward(self, **inputs):
        output = self._forward(**inputs)
        input_y = inputs["input_y"]
        critern = M_Loss()
        l = critern(output, input_y)
        return {"output": output, "input_y": input_y, "loss": l}

    def predict(self, **inputs):
        output = self._forward(**inputs)
        input_y = inputs["input_y"]
        # input_y = input_y.squeeze(-1)
        return {
            "output": output,
            "input_x": inputs["input_x"],
            "input_y": input_y,
        }

class SNet_power_model(Model):
    def __init__(
        self,
        x: int = 3,
        feature_size: int = source_load_param.lstm["feature_size"][0],

        initial_method=None,
    ):

        super(SNet_power_model, self).__init__()

        self.lstm = LSTM(
            input_size=x,
            hidden_size=feature_size,
            num_layers=2,
            dropout=0.1,
        )
        self.lstmf = LSTM(
            input_size=x,
            hidden_size=feature_size,
            num_layers=2,
            dropout=0.1,
        )
        self.attn = selfattention(feature_size, num_attention_heads=8)
        self.attnf = selfattention(feature_size, num_attention_heads=8)
        self.granularity_attn = AttentionGrandularity(d_model=feature_size, n_head=4)

        self.lstm_0 = nn.LSTM(256, 512, num_layers=2, batch_first=True) # (640, 768)
        self.linear = nn.ModuleList(
            [
                nn.Linear(512, 128), # (768, 128)
            ]
        )
        self.decoder = nn.Linear(
           128, source_load_param.pred_len, bias=True
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        initial_parameter(self)

    def _forward(self, **inputs):
        input_x = inputs["input_x"]
        input_fx = inputs["input_fine_x"]
        input_x, _ = self.lstm(input_x.squeeze(-2))
        input_x = self.attn(input_x, input_x, input_x)
        input_f = input_x
        input_bf = torch.zeros(
            input_fx.shape[0],
            input_fx.shape[1],
            input_fx.shape[2],
            source_load_param.lstm["feature_size"][0],
        )

        for t, n in enumerate(range(input_fx.shape[-2])):
            input_fn = input_fx[:, :, n, :]
            input_fn, _ = self.lstm(input_fn)
            input_bf[:, :, t, :] = input_fn

        input_bf = input_bf.to(input_x.device)
        input_fx = self.granularity_attn(
            queries=input_bf, keys=input_x.unsqueeze(-2), keys_0=input_x.unsqueeze(-2)
        )
        x = torch.cat([input_fx, input_f], dim=-1)
        x, _ = self.lstm_0(x)
        x = x[:, -1, :]
        x_output = x
        for layer in self.linear:
            x_output = self.dropout(self.relu(layer(x_output)))
        output = self.decoder(x_output)
        return  output.unsqueeze(-1)

    def forward(self, **inputs):
        output = self._forward(**inputs)
        input_y = inputs["input_y"]
        critern = M_Loss()
        l = critern(output, input_y)
        return {"output": output, "input_y": input_y, "loss": l}

    def predict(self, **inputs):
        output = self._forward(**inputs)
        input_y = inputs["input_y"]
        # input_y = input_y.squeeze(-1)
        return {
            "output": output,
            "input_x": inputs["input_x"],
            "input_y": input_y,
        }
class PNet_power_model(Model):
    def __init__(
        self,
        x: int = 3,
        feature_size: int = source_load_param.lstm["feature_size"][0],

        initial_method=None,
    ):

        super(PNet_power_model, self).__init__()

        self.lstm = LSTM(
            input_size=x,
            hidden_size=feature_size,
            num_layers=2,
            dropout=0.1,
        )
        self.lstmf = LSTM(
            input_size=x,
            hidden_size=feature_size,
            num_layers=2,
            dropout=0.1,
        )
        self.attn = selfattention(feature_size, num_attention_heads=8)
        self.attnf = selfattention(feature_size, num_attention_heads=8)

        self.lstm_0 = nn.LSTM(640, 768, num_layers=2, batch_first=True) # (640, 768)
        self.linear = nn.ModuleList(
            [
                nn.Linear(768, 128), # (768, 128)
               # nn.Linear(384, 128)
            ]
        )
        self.decoder = nn.Linear(
           128, source_load_param.pred_len, bias=True
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        initial_parameter(self)

    def _forward(self, **inputs):
        input_x = inputs["input_x"]
        input_fx = inputs["input_fine_x"]
        input_x, _ = self.lstm(input_x.squeeze(-2))
        input_x = self.attn(input_x, input_x, input_x)
        input_f = input_x

        for t, n in enumerate(range(input_fx.shape[-2])):
            input_fn = input_fx[:, :, n, :]
            input_fn, _ = self.lstm(input_fn)
            input_attn = self.attnf(queries=input_x, keys=input_fn, values=input_fn)

            # plt.imsave(f"attn_{index}.png", input_attn[0, :, :].cpu().detach().numpy())
            input_f = torch.cat([input_f, input_attn], dim=-1)
        x = torch.cat([input_x, input_f], dim=-1)
        x, _ = self.lstm_0(x)
        x = x[:, -1, :]
        x_output = x
        for layer in self.linear:
            x_output = self.dropout(self.relu(layer(x_output)))
        output = self.decoder(x_output)
        return  output.unsqueeze(-1)

    def forward(self, **inputs):
        output = self._forward(**inputs)
        input_y = inputs["input_y"]
        critern = M_Loss()
        l = critern(output, input_y)
        return {"output": output, "input_y": input_y, "loss": l}

    def predict(self, **inputs):
        output = self._forward(**inputs)
        input_y = inputs["input_y"]
        # input_y = input_y.squeeze(-1)
        return {
            "output": output,
            "input_x": inputs["input_x"],
            "input_y": input_y,
        }
class power_lstm_attn(Model):
    # lstm_attention_6_20128
    def __init__(
        self,
        feature_size: int = source_load_param.lstm["feature_size"][0],
        num_layers: int = 3,
        x: int = 3,
        initial_method=None,
    ):

        super(power_lstm_attn, self).__init__()
        # self.fc = nn.Linear(x, feature_size)
        self.lstm = LSTM(
            input_size=x,
            hidden_size=feature_size,
            num_layers=num_layers,
            dropout=0.1,
        )
        self.attn = selfattention(feature_size, num_attention_heads=8)
        self.granularity_attn = AttentionGrandularity(d_model=feature_size, n_head=8)
        self.linear = nn.ModuleList(
            [
                nn.Linear(feature_size, feature_size * 2),
                nn.Linear(feature_size * 2, feature_size * 2),
            ]
        )
        self.decoder = nn.Linear(
            feature_size * 2, len(source_load_param.target), bias=True
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        initial_parameter(self)

    def _forward(self, **inputs):
        input_x = inputs["input_x"]
        input_fx = inputs["input_fine_x"]

        input_x, _ = self.lstm(input_x.squeeze(-2))
        input_x = self.attn(input_x, input_x, input_x)
        x_output = input_x
        for layer in self.linear:
            x_output = self.dropout(self.relu(layer(x_output)))
        output = self.decoder(x_output)
        output = output[:, -source_load_param.pred_len:, :]
        return output

    def forward(self, **inputs):
        output = self._forward(**inputs)
        # print("output:{}".format(output))
        input_y = inputs["input_y"]
        input_y = input_y.squeeze()
        critern = M_Loss()
        l = critern(output, input_y)
        return {"output": output, "input_y": input_y, "loss": l}

    def predict(self, **inputs):
        output = self._forward(**inputs)
        input_y = inputs["input_y"]
        input_y = input_y.squeeze()
        # input_y = input_y.squeeze(-1)
        return {
            "output": output,
            "input_x": inputs["input_x"],
            "input_y": input_y,
        }
class power_model0(Model): # abondon granularityatten
    def __init__(
        self,
        feature_size: int = source_load_param.lstm["feature_size"][0],
        num_layers: int = 3,
        x: int = 3,
        initial_method=None,
    ):

        super(power_model0, self).__init__()

        self.lstm = LSTM(
            input_size=x,
            hidden_size=feature_size,
            num_layers=num_layers,
            dropout=0.1,
        )
        self.lstmf = LSTM(
            input_size=x,
            hidden_size=feature_size,
            num_layers=num_layers,
            dropout=0.1,
        )
        self.attn = selfattention(feature_size, num_attention_heads=8)
        self.attnf = selfattention(feature_size, num_attention_heads=8)
        self.linear = nn.ModuleList(
            [
                nn.Linear(feature_size * 6, feature_size * 6),
                nn.Linear(feature_size * 6, feature_size * 2),
            ]
        )
        self.decoder = nn.Linear(
            feature_size * 2, len(source_load_param.target), bias=True
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        initial_parameter(self)

    def _forward(self, **inputs):
        x = inputs["input_x"]
        fx = inputs["input_fine_x"]

        x, _ = self.lstm(x.squeeze(-2))
        x = self.attn(x, x, x)
        fo = x
        for t, n in enumerate(range(fx.shape[-2])):
            fn = fx[:, :, n, :]
            fn, _ = self.lstm(fn)
            #input_attn = self.attnf(queries=input_x, keys=input_fn, values=input_fn)

            # plt.imsave(f"attn_{index}.png", input_attn[0, :, :].cpu().detach().numpy())
            fo = torch.cat([fo, fn], dim=-1)
            #input_bf[:, :, t, :] = input_fn
        x = torch.cat([x, fo], dim=-1)
        # #plt.imsave(f"attn.png", x[0, :, :].cpu().detach().numpy())
        # x = input_x
        x_output = x
        for layer in self.linear:
            x_output = self.dropout(self.relu(layer(x_output)))
        output = self.decoder(x_output)
        output = output[:, -source_load_param.pred_len :, :]
        return x, output

    def forward(self, **inputs):
        x, output = self._forward(**inputs)
        # print("output:{}".format(output))
        input_y = inputs["input_y"]
        input_y = input_y.squeeze()
        critern = M_Loss()
        l = critern(output, input_y)
        return {"output": output, "input_y": input_y, "loss": l}

    def predict(self, **inputs):
        x, output = self._forward(**inputs)
        input_y = inputs["input_y"]
        input_y = input_y.squeeze()
        # input_y = input_y.squeeze(-1)
        return {
            "output": output,
            "input_x": inputs["input_x"],
            "input_y": input_y,
        }
class power_lstm(Model):
    def __init__(
        self,
        feature_size: int = source_load_param.lstm["feature_size"][0],
        num_layers: int = 3,
        x: int = 3,
        initial_method=None,
    ):

        super(power_lstm, self).__init__()

        self.lstm = LSTM(
            input_size=x,
            hidden_size=feature_size,
            num_layers=num_layers,
            dropout=0.1,
        )
        self.linear = nn.ModuleList(
            [
                nn.Linear(feature_size, feature_size * 2),
                nn.Linear(feature_size * 2, feature_size * 2),
            ]
        )
        self.decoder = nn.Linear(
            feature_size * 2, len(source_load_param.target), bias=True
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        initial_parameter(self)

    def _forward(self, **inputs):
        x = inputs["input_x"]

        x, _ = self.lstm(x.squeeze(-2))
        x_output = x
        for layer in self.linear:
            x_output = self.dropout(self.relu(layer(x_output)))
        output = self.decoder(x_output)
        output = output[:, -source_load_param.pred_len :, :]
        return x, output

    def forward(self, **inputs):
        x, output = self._forward(**inputs)
        # print("output:{}".format(output))
        input_y = inputs["input_y"]
        input_y = input_y.squeeze()
        critern = M_Loss()
        l = critern(output, input_y)
        return {"output": output, "input_y": input_y, "loss": l}

    def predict(self, **inputs):
        x, output = self._forward(**inputs)
        input_y = inputs["input_y"]
        input_y = input_y.squeeze()
        # input_y = input_y.squeeze(-1)
        return {
            "output": output,
            "input_x": inputs["input_x"],
            "input_y": input_y,
        }