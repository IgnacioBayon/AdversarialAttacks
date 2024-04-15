import torch
import torch.nn as nn
import torch.optim as optim

def gbda_example(model, input_sequence, true_label, epsilon=0.1, iterations=10, perturbation_limit=0.05):
    input_sequence_var = torch.tensor(input_sequence, requires_grad=True, dtype=torch.float)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD([input_sequence_var], lr=epsilon)  
    
    for i in range(iterations):
        optimizer.zero_grad()
        output = model(input_sequence_var.unsqueeze(0))  
        loss = criterion(output, torch.tensor([true_label]))
        loss.backward()
        
        # Perturbación restringida
        perturbation = epsilon * input_sequence_var.grad.sign()
        perturbation = torch.clamp(perturbation, -perturbation_limit, perturbation_limit)
        
        input_sequence_var.data = input_sequence_var.data + perturbation  
        input_sequence_var.data = torch.clamp(input_sequence_var.data, 0, 1)
    
    return input_sequence_var.detach().numpy()
