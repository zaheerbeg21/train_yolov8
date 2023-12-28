CUDA_LAUNCH_BLOCKING=1

from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.yaml')  # build a new model from YAML
# model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
# model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

conf_path = '/home/train_yolov8/local_env/config.yaml'
number_of_epochs = 100

# Load a model
# model = YOLO('path/to/last.pt')  # load a partially trained model



# Use the model
results = model.train(data=conf_path, epochs=number_of_epochs)  # train the model

# Resume training
# results = model.train(resume=True)

# Train the model with 2 GPUs
# results = model.train(data=conf_path, epochs=number_of_epochs, device=[0,1])



# -----------with checkpoints-------------
# from ultralytics import YOLO
# from torch.utils.checkpoint import Checkpoint

# # Load a model
# model = YOLO('yolov8n.yaml')  # build a new model from YAML
# model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
# model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

# conf_path = 'config.yaml'
# number_of_epochs = 100


# optimizer = torch.optim.Adam(model.parameters())

# checkpoint = Checkpoint(model, optimizer)
# for epoch in range(100):
#     # Train your model...
#     if epoch % 10 == 0:
#         checkpoint.save(f"checkpoint_{epoch}.pt")