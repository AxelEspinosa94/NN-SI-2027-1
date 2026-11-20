import math

def haversine(lat1, lon1, lat2, lon2):
    """
    Computes the great-circle distance between two points on Earth
    using the Haversine formula.

    Parameters
    ----------
    lat1, lon1 : float
        Latitude and longitude of point 1 (in decimal degrees)
    lat2, lon2 : float
        Latitude and longitude of point 2 (in decimal degrees)

    Returns
    -------
    float
        Distance in meters between the two points.
    """

    R = 6371000  # Earth radius in meters

    # Convert degrees to radians
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    # Haversine formula
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c