from ray import tune


def parameters_selection(zone):
    model = LgbModel(zone=zone, log_name='lgb_model')
    

if __name__ == '__main__':
    
    search_space = {
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([32, 64, 128]),
        # Add other hyperparameters here
    }