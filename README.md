# End-to-End (E2E) Spoken-Language-Understanding(SLU) in PyTorch
CDS 2nd year - capstone project - buiild a E2E speech to intent (S2I) model with the application of transfer learning

This repo contains modified Pytorch code adopted from Loren Lugosch. For more information, please refer to his papers "[Speech Model Pre-training for End-to-End Spoken Language Understanding](https://arxiv.org/abs/1904.03670)" and "[Using Speech Synthesis to Train End-to-End Spoken Language Understanding Models](https://arxiv.org/abs/1910.09463)".
Codes are modified for our own purposes. Please refer to the final report in this repo for more information on modifications.

If you have any questions about this code or have problems getting it to work, please send me an email at ```<cora.jung@nyu.edu>```.

## Dependencies
PyTorch, torchaudio, numpy, soundfile, pandas, tqdm, textgrid.py

## Pre-Training
First, change the ```asr_path``` and/or ```slu_path``` in the config file (like ```experiments/unfreeze_word_pretrain.cfg```, or whichever experiment you want to run) to point to where the LibriSpeech data and/or Fluent Speech Commands data are stored on your computer.

_SLU training:_ To train the model on an SLU dataset, run the following command:
```
python main_pretrain.py --train --config_path=<config path>
```
Now the best model_state.pth should be saved in ```experiments/unfreeze_word_pretrain/training/model_state.pth```

## Fine-Tuning
We are using the pre-trained model_state.pth to fine-tune. Run the following command after changing the ```asr_path``` and/or ```slu_path``` in the config file (like ```experiments/unfreeze_word_finetune.cfg```, or whichever experiment you want to run) to point to where speech data for fine-tuning are stored in your computer.

```
python main_finetune.py --train --restart --config_path=<path to .cfg> --model_path=<path to .pth>
```
**model path** should point to the saved model_state.pth. We need to give path upto ``experiments/unfreeze_word_pretrain/training/```
**config path** should point to the config file that contains information about the fine-tuning model (in this example, ```experiments/unfreeze_word_finetune.cfg``` would serve the job. Don't forget to change the ```asr_path``` and/or ```slu_path```)

_ASR pre-training:_ **Note:** the experiment folders in this repo already have a pre-trained LibriSpeech model that you can use. LibriSpeech is pretty big (>100 GB uncompressed), so don't do this part unless you want to re-run the pre-training part with different hyperparameters. If you want to do this, you will first need to download our LibriSpeech alignments [here](https://zenodo.org/record/2619474#.XKDP2VNKg1g), put them in a folder called "text", and put the LibriSpeech audio in a folder called "audio". To pre-train the model on LibriSpeech, run the following command:
```
python main.py --pretrain --config_path=<path to .cfg>
```

## Inference
You can perform inference with a trained SLU model as follows (thanks, Nathan Folkman!):
```python
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
```
The ```test.wav``` file included with this repo has a recording of Loren saying "Hey computer, could you turn the lights on in the kitchen please?", and so the inferred intent should be ```{"activate", "lights", "kitchen"}```.

## Citation
- Loren Lugosch, Mirco Ravanelli, Patrick Ignoto, Vikrant Singh Tomar, and Yoshua Bengio, "Speech Model Pre-training for End-to-End Spoken Language Understanding", Interspeech 2019.
- Loren Lugosch, Brett Meyer, Derek Nowrouzezahrai, and Mirco Ravanelli, "Using Speech Synthesis to Train End-to-End Spoken Language Understanding Models", ICASSP 2020.
