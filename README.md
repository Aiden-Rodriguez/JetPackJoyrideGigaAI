### Python version 3.11.14

### Install  requirements
pip install -r requirements.txt

### Install torch (for nvidia GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

### Make sure venv is being used
cntrl+shift+p --> Python: Select Interpreter --> python from venv


# General Workflow: Vision

## Train Vision
- Run Python jetpack.py, and in another terminal run python collect_data.py. Play the game until collect_data finishes.
-> This looks at the gameplay, creates training data and labels. Saves to /dataset

- Train data. Python train_yolo.py
-> Downlaods pretrained YOLO model, fine tunes it based on inputs.

## Run Vision
- Run Python jetpack.py, and also "python vision.py -yolo" in another terminal
-> This will visualize the game based upon trained data.
-> Can also run python vision.py -color, but that doesn't use training data, just bases it off the colors. 

- Best.pt is put in root if you wanna skip training.

# General Workflow: Proximal Policy Optimization (PPO) model

## Train PPO
- If you want to continue training, the same model, look in train_ppo.py and uncomment the lines for loading the model, and comment out the lines specifying the new model.
- If you want to train a new model, nothing needs to be changed
- Run python train_ppo.py

## Run PPO
- Run python watch_model.py

