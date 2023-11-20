from phonemizer import phonemize, version
from phonemizer.separator import Separator
import pandas as pd 
from datetime import datetime

"""
This script takes a csv input and phonemizes while maintaining couplet lines remain in order.
"""

  
input_file = "data/rhyme_couplets.csv"
couplets_df = pd.read_csv(input_file)

line1_g =couplets_df["line1_g"]
line2_g =couplets_df["line2_g"]

line1_p = []
line2_p = []
batch_size =100

for i in range(0,len(couplets_df),batch_size):
#prepare batches for efficiency while phoonemizing
  if i + batch_size<len(couplets_df):
    batch1 = line1_g[i:i+batch_size]
    batch2 = line2_g[i:i+batch_size]
  else:
    batch1 = line1_g[i:len(couplets_df)]
    batch2 = line2_g[i:len(couplets_df)]
  try:
    #phonemize batches
    batch1_p = phonemize(batch1, language='en-us',
    backend='festival',separator=Separator(phone="-", word=' ',
    syllable='|'), strip=True)

    batch2_p = phonemize(batch2, language='en-us', 
    backend='festival',separator=Separator(phone="-", word=' ',
    syllable='|'), strip=True)

    line1_p.extend(batch1_p)
    line2_p.extend(batch2_p)

  except:
    print(f"phonemization failed for batch:{i}")
    # Add None placeholders for failed batches
    line1_p.extend([None] * len(batch1))
    line2_p.extend([None] * len(batch2))

couplets_df["line1_p"] = line1_p
couplets_df["line2_p"] = line2_p
# phonemize_df(couplets_df, "line1_g", "line1_p")
# phonemize_df(couplets_df, "line2_g", "line2_p")

couplets_df.to_csv("data/rhyme_couplets_f-phonemized_07-30-23.csv", index=False)

# print("Phonemization Complete")
