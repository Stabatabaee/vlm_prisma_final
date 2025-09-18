# check_alignment.py
import pandas as pd
import sys

v18 = pd.read_csv("v18_full.csv")
v22s = pd.read_csv("gold_pred_v22_strict_norm.csv")  # or hybrid_norm
avail = pd.read_csv("availability_mask_dedup.csv")

def base(x): 
    return str(x).replace(".txt","")

v18['File_base']  = v18['File'].apply(base)
v22s['File_base'] = v22s['File'].apply(base)
avail['File_base']= avail['File'].apply(base) if 'File' in avail.columns else avail.iloc[:,0].apply(base).rename('File_base')

print("Counts:")
print(" v18 unique:", v18['File_base'].nunique())
print(" v22 unique:", v22s['File_base'].nunique())
print(" avail unique:", avail['File_base'].nunique())

i12 = set(v18['File_base']) & set(v22s['File_base'])
i1a = set(v18['File_base']) & set(avail['File_base'])
i2a = set(v22s['File_base']) & set(avail['File_base'])
i123= set(v18['File_base']) & set(v22s['File_base']) & set(avail['File_base'])

print("\nIntersections:")
print(" v18 ∩ v22:", len(i12))
print(" v18 ∩ avail:", len(i1a))
print(" v22 ∩ avail:", len(i2a))
print(" v18 ∩ v22 ∩ avail:", len(i123))

missing_in_v22 = sorted(set(v18['File_base']) - set(v22s['File_base']))
missing_in_av  = sorted(set(v18['File_base']) - set(avail['File_base']))
if missing_in_v22:
    print("\nPresent in v18 but missing in v22 (showing up to 20):", missing_in_v22[:20])
if missing_in_av:
    print("\nPresent in v18 but missing in availability (up to 20):", missing_in_av[:20])
