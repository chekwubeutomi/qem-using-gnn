import json
import matplotlib.pyplot as plt

with open("best_model.history.json") as f:
    history = json.load(f)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(history["train_mse"], label="Train")
plt.plot(history["val_mse"],   label="Val")
plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.legend(); plt.title("Loss")

plt.subplot(1, 3, 2)
plt.plot(history["val_r2"])
plt.xlabel("Epoch"); plt.ylabel("R²"); plt.title("R² Score")

plt.subplot(1, 3, 3)
plt.plot([v * 100 for v in history["val_mitigation_ratio"]])
plt.xlabel("Epoch"); plt.ylabel("Mitigation %"); plt.title("Mitigation Ratio")

plt.tight_layout()
plt.savefig("training_history.png", dpi=150)