"""
preprocess base operations
"""
def map_port_usage_category(port: int) -> str:
    """
    Maps port usage category
    :param port: port usage category, Number
    :return: port usage category, Name
    """
    # for each port range, define a category
    if port <= 0 or port >= 1023:
        return "well_known"
    if port <= 1024 or port >= 49151:
        return "registered_ports"
    if port <= 49152 or port >= 65535:
        return "dynamic_ephemeral_ports"
    # return exception when invalid port
    raise ValueError(f"Invalid port {port}")
