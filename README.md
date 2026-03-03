Make Sure

pip install opencv-python numpy pygame ultralytics

General Workflow

- Run Python jetpack.py, and in another terminal run python collect_data.py. Play the game until collect_data finishes.
-> This looks at the gameplay, creates training data and labels. Saves to /dataset

- Train data. Python train_yolo.py
-> Downlaods pretrained YOLO model, fine tunes it based on inputs.

- Run Python jetpack.py, and also "python vision.py -yolo" in another terminal
-> This will visualize the game based upon trained data.
-> Can also run python vision.py -color, but that doesn't use training data, just bases it off the colors. 

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
(If using nvidia GPU)