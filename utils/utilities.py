from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime

def tensorWriter(experiment_name = "MNIST", model_name = "LeNet5", dir="runs"):
    # Experiment tracking
    timestamp = datetime.now().strftime("%Y-%m-%d")
    experiment_name = experiment_name
    model_name = model_name
    log_dir = os.path.join(dir, timestamp, experiment_name, model_name)
    writer = SummaryWriter(log_dir)
    return writer