import sys
import numpy as np

def debug_print(paras , stop  = True):
    """
    A quick debug print function for developement... =)
    """

    if type(paras) != list:
        paras = [paras]
    
    print("===[DEBUG]===")

    for i in range(len(paras)):
        print(f"{paras[i]}")
    if stop:    
        sys.exit("[DEBUG] Done")

    return None