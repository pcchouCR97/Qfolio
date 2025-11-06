# ============================================================================
# MEGA-CAP TECH GIANTS (Market Leaders)
# ============================================================================
tech_giants = ['AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN', 'TSLA', 'NVDA', 'NFLX']

# ============================================================================
# SEMICONDUCTORS & CHIPS (AI/Hardware Infrastructure)
# ============================================================================
semiconductors = ['NVDA', 'AMD', 'INTC']

# ============================================================================
# ENTERPRISE SOFTWARE & CLOUD
# ============================================================================
enterprise_software = ['MSFT', 'ORCL', 'CRM', 'CSCO']

# ============================================================================
# DEFENSE & AEROSPACE
# ============================================================================
defense = ['PLTR', 'LMT', 'NOC', 'RTX', 'BA']

# ============================================================================
# FINTECH & PAYMENTS
# ============================================================================
fintech = ['HOOD', 'PYPL', 'SQ']  # SQ not in your list but related

# ============================================================================
# TRADITIONAL FINANCE (Banks & Payment Networks)
# ============================================================================
finance = ['JPM', 'V', 'MA', 'PYPL', 'AXP', 'BAC', 'C', 'GS', 'MS']

# ============================================================================
# ENERGY (Oil & Gas)
# ============================================================================
energy = ['XOM', 'CVX', 'OXY']

# ============================================================================
# BIOTECHNOLOGY & PHARMACEUTICALS
# ============================================================================
biotech = ['MRNA', 'BNTX', 'REGN', 'PFE']

# ============================================================================
# CONSUMER & BLUE CHIPS
# ============================================================================
consumer = ['DIS', 'WMT', 'KO']

# ============================================================================
# INDUSTRIALS
# ============================================================================
industrials = ['GE', 'GEV', 'BA']  # BA also in defense

# ============================================================================
# ETFS - BROAD MARKET
# ============================================================================
etf_broad = ['VOO', 'VTI', 'SPY', 'QQQ']

# ============================================================================
# ETFS - SECTOR SPECIFIC
# ============================================================================
etf_sector = [
    'ARKK',  # Innovation/Growth
    'XLF',   # Financials
    'XLE',   # Energy
    'XLY',   # Consumer Discretionary
    'XLI',   # Industrials
    'XLB',   # Materials
    'XLK',   # Technology
    'XLV'    # Healthcare
]

# ============================================================================
# ALL SYMBOLS COMBINED (NO DUPLICATES)
# ============================================================================
all_symbols = [
    # Tech Giants
    'AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN', 'TSLA', 'NVDA', 'NFLX',
    
    # Semiconductors (NVDA, INTC, AMD already covered above, adding unique ones)
    'AMD', 'INTC',
    
    # Enterprise Software (MSFT, ORCL already in, adding unique)
    'ORCL', 'CRM', 'CSCO',
    
    # Defense
    'PLTR', 'LMT', 'NOC', 'RTX', 'BA',
    
    # Fintech
    'HOOD', 'PYPL',
    
    # Finance
    'JPM', 'V', 'MA', 'AXP', 'BAC', 'C', 'GS', 'MS',
    
    # Energy
    'XOM', 'CVX', 'OXY',
    
    # Biotech
    'MRNA', 'BNTX', 'REGN', 'PFE',
    
    # Consumer
    'DIS', 'WMT', 'KO',
    
    # Industrials
    'GE', 'GEV',
    
    # ETFs
    'VOO', 'VTI', 'SPY', 'QQQ', 'ARKK', 
    'XLF', 'XLE', 'XLY', 'XLI', 'XLB', 'XLK', 'XLV'
]


def return_all_symbols():
    return sorted(list(set(all_symbols)))

# Remove duplicates and sort
all_symbols = sorted(list(set(all_symbols)))

print(f"Total unique symbols: {len(all_symbols)}")
print(all_symbols)