from qfolio.utils.constants import SECTOR_MAPPING

def calculate_sector_exposure(allocation, prices):
    """
    Calculate portfolio value distribution by sector.

    Parameters:
    -----------
    allocation : dict
        {ticker: shares} mapping
    prices : dict
        {ticker: price} mapping

    Returns:
    --------
    dict : {sector: weight_percentage}
    """
    total_value = sum(allocation.get(t, 0) * prices.get(t, 0) for t in allocation)
    if total_value == 0:
        return {}

    sector_values = {}
    for ticker, shares in allocation.items():
        sector = SECTOR_MAPPING.get(ticker, 'Unknown')
        value = shares * prices.get(ticker, 0)
        sector_values[sector] = sector_values.get(sector, 0) + value

    return {sector: (value / total_value) * 100 for sector, value in sector_values.items()}

def calculate_sector_concentration(sector_weights):
    """
    Calculate Herfindahl-Hirschman Index for sector concentration.

    HHI = sum of squared weights (in decimal form) * 100
    - 100 = single sector (maximum concentration)
    - <15 = diversified (per DOJ antitrust guidelines)
    - 15-25 = moderate concentration
    - >25 = high concentration

    Parameters:
    -----------
    sector_weights : dict
        {sector: weight_percentage}

    Returns:
    --------
    float : HHI value
    """
    weights_squared = [(w/100)**2 for w in sector_weights.values() if w > 0]
    hhi = sum(weights_squared) * 100
    return hhi