class Platform:
    """A basic platform abstraction to manage distributed training.
    Or, if we are not running on multiple nodes, identifies the single
     GPU device. """
    
    def get_device_str(self):
        pass