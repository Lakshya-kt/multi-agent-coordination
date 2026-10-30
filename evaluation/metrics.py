def calculate_success_rate(predictions, ground_truth):
    """
    Calculate the success rate of predictions against ground truth.

    Args:
        predictions (list): A list of predicted values.
        ground_truth (list): A list of actual values.

    Returns:
        float: The success rate as a percentage.
    """
    raise NotImplementedError("This function is not yet implemented.")

def calculate_mean_completion_time(completion_times):
    """
    Calculate the mean completion time from a list of completion times.

    Args:
        completion_times (list): A list of completion times in seconds.

    Returns:
        float: The mean completion time.
    """
    raise NotImplementedError("This function is not yet implemented.")

def calculate_mean_return(episode_returns):
    """
    Calculate the mean return from a list of episode returns.

    Args:
        episode_returns (list): A list of returns for each episode.
        
    Returns:
        float: The mean return.
    """
    raise NotImplementedError("This function is not yet implemented.")

def calculate_redundant_visits(agent_visits):
    """
    TODO:
    Measure repeated visits to alrady convered target.
    """
    raise NotImplementedError
    