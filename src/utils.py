def move_to(obj, device):
    """
    move the object to the desired device (cpu, cuda)
    :param obj: object to move
    :param device: destination to transport to
    :return:
    """
    if hasattr(obj, "to"):
        return obj.to(device)
    elif isinstance(obj, list):
        return [move_to(x, device) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(move_to(list(obj), device))
    elif isinstance(obj, set):
        return set(move_to(list(obj), device))
    elif isinstance(obj, dict):
        to_ret = dict()
        for key, value in obj.items():
            to_ret[move_to(key, device)] = move_to(value, device)
        return to_ret
    else:
        return obj
