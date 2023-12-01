ee616
=====
It is repository for the EE616 project "RealAugFL: Augmentation for Real-World Federated Learning"


### Create conda env 
    conda env create -f environment.yml --name ee616

### How to run the code
    python main.py

### Args for main.py
    --seed: Seed for reproducibility (default: 0, type: int).
    --dataset: Dataset to use for training (default: "cifar10", type: str).
    --model: Model structure to use (default: "resnet18", type: str).
    --gpu: Specific GPU to train on, choices are [0, 1, 2, 3, 4, 5, 6, 7] (default: 7, type: int).
    --method: Model structure (default: "base", type: str).
    For FL data distribution:
    
    --data_distribution: Choose data distribution: "iid", "non_iid_class", "non_iid_domain", "non_iid" (default: "iid", type: str).
    --num_user: Number of artificial clients in FL training (default: 100, type: int).
    --ratio_samples_per_client: Ratio of num samples per client (default: 1, type: float).
    --num_classes_per_client: Number of classes each client will have in non-iid setting (default: 2, type: int).
    --num_domains_per_client: Number of domains each client will have in non-iid setting (default: 2, type: int).
    For FL training:
    
    --federated_round: Number of global rounds in FL training (default: 100, type: int).
    --local_epoch: Number of local epochs in FL training (default: 5, type: int).
    --percentage_selected_client: Percentage of clients to be selected every round (default: 0.1, type: float).
    Other training parameters:
    
    --lr: Learning rate (default: 0.1, type: float).
    --bs: Batch size (default: 64, type: int).
