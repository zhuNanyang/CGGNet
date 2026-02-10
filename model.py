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
