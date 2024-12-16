import numpy as np

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class LossCategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)
            
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
    
    def backward(self, output, y_true):
        samples = len(output)
        labels = len(output[0])
        
        output_clipped = np.clip(output, 1e-7, 1 - 1e-7)
        
        self.dinputs = -y_true / output_clipped
        self.dinputs = self.dinputs / samples
        

class LossBinaryCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        if y_true.shape[1] > 1:
            y_true = np.argmax(y_true, axis=1)

        if y_pred.shape[1] > 1:
            y_pred = y_pred[:, 1]
        
        # Clip predictions to prevent log(0)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        sample_losses = -(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))

        return np.mean(sample_losses)
    
    def backward(self, output, y_true):
        if y_true.shape[1] > 1:
            y_true = np.argmax(y_true, axis=1)

        if output.shape[1] > 1:
            output = output[:, 1]
        
        output_clipped = np.clip(output, 1e-7, 1 - 1e-7)

        self.dinputs = -(y_true / output_clipped - (1 - y_true) / (1 - output_clipped))
        self.dinputs = self.dinputs / len(output)