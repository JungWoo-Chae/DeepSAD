from .mnist import load_mnist

def load_dataset(cfg):
    dataset_name = cfg['dataset_name']
    
    implemented_datasets = ('mnist', 'fmnist', 'cifar10',
                            'arrhythmia', 'cardio', 'satellite', 'satimage-2', 'shuttle', 'thyroid')
    
    assert dataset_name in implemented_datasets
    
    dataset = None
    
    if dataset_name == 'mnist':
        dataset = load_mnist(cfg)
        
    return dataset