from src.gui.utils import ModelConfig
from src.gui.app import App

ssdd_prefix = "./models/pt/ssdd/"
models = [
    "detect/v8n",
    "detect/v8n-dc",
    "detect/v8n-sh",
    "detect/v8n-sh-dc",
    "detect/v8s",
    "obb/v8n",
    "obb/v8n-dc",
    "obb/v8n-sh",
    "obb/v8n-sh-dc",
    "obb/v8s",
]
weights = "/weights/best.pt"
structure_img = "/results.png"
train_img = "/results.png"
eval_img = "/PR_curve.png"


model_configs = [
    ModelConfig(
        name=model,
        path=ssdd_prefix + model + weights,
        structure_img_path=ssdd_prefix + model + structure_img,
        train_img_path=ssdd_prefix + model + train_img,
        eval_img_path=ssdd_prefix + model + eval_img,
    )
    for model in models
]

if __name__ == "__main__":
    app = App(model_configs)
    app.run()
