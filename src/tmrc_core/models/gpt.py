def validate_config(config):
    """Some basic sanity checks for the model config."""    
    assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"