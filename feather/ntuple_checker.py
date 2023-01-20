import uproot
import pandas as pd
with uproot.open('364250_mod.root') as f:
    print(f.keys())
    f['nominal'].show()
    scores = f['nominal'].arrays(['eventNumber', 'score'],library='pd')
    print("Head: \n", scores.head())
    print("Tail: \n", scores.tail())