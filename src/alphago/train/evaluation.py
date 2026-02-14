import numpy as np
from tqdm import tqdm


import torch
import torch.nn as nn


class GoEvaluator:
    def __init__(self, model, test_loader, device="cuda"):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device

    def evaluate(self, checkpoint_path=None):
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            state_dict = checkpoint["model_state_dict"]
            if hasattr(self.model, "module"):
                self.model.module.load_state_dict(state_dict)
            else:
                self.model.load_state_dict(state_dict)
            print(f"Loaded checkpoint from: {checkpoint_path}")

        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        top5_correct = 0
        predictions = []
        targets = []

        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc="Evaluating")

            for data, target in pbar:
                data, target = data.to(self.device), target.to(self.device)

                if target.dim() > 1 and target.size(1) > 1:
                    target = torch.argmax(target, dim=1)

                output = self.model(data)
                loss = criterion(output, target)

                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

                _, top5_pred = torch.topk(output.data, 5, dim=1)
                top5_correct += sum(
                    [target[i] in top5_pred[i] for i in range(target.size(0))]
                )

                predictions.extend(predicted.cpu().numpy())
                targets.extend(target.cpu().numpy())

                pbar.set_postfix(
                    {
                        "Loss": f"{loss.item():.4f}",
                        "Acc": f"{100.0 * correct / total:.2f}%",
                    }
                )

        avg_loss = total_loss / len(self.test_loader)
        accuracy = 100.0 * correct / total
        top5_accuracy = 100.0 * top5_correct / total

        print("Test Results:")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Top-1 Accuracy: {accuracy:.2f}%")
        print(f"Top-5 Accuracy: {top5_accuracy:.2f}%")
        print(f"Total samples: {total}")

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "top5_accuracy": top5_accuracy,
            # "predictions": predictions,
            # "targets": targets,
        }

    def predict(self, data, return_probabilities=False):
        self.model.eval()

        with torch.no_grad():
            if isinstance(data, np.ndarray):
                data = torch.from_numpy(data).float()

            data = data.to(self.device)

            if data.dim() == 3:
                data = data.unsqueeze(0)

            output = self.model(data)

            if return_probabilities:
                probabilities = torch.softmax(output, dim=1)
                return probabilities.cpu().numpy()
            else:
                _, predicted = torch.max(output, 1)
                return predicted.cpu().numpy()
