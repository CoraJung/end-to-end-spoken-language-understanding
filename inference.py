import data
import models
import soundfile as sf
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config = data.read_config("experiments/no_unfreezing.cfg"); _,_,_=data.get_SLU_datasets(config)
model = models.Model(config).eval()
model.load_state_dict(torch.load("experiments/no_unfreezing/training/model_state.pth", map_location=device)) # load trained model

signal, _ = sf.read("test.wav")
signal = torch.tensor(signal, device=device).float().unsqueeze(0)

model.decode_intents(signal)