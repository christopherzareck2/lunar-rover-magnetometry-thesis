"""Print summary tables of the regolith study from y10_results.json (run from this folder)."""
import json, sys
import pandas as pd, numpy as np
pd.set_option("display.width",250); pd.set_option("display.max_columns",40); pd.set_option("display.max_rows",500)
d=json.load(open("y10_results.json")); df=pd.DataFrame(d["rows"])
fr=df[df["mode"]=="fieldrms"].copy(); df=df[df["mode"]!="fieldrms"].copy(); df["noisy"]=df["noisy"].astype(bool)
if __name__=="__main__":
    print(fr.groupby(["h","chi_reg","B"]).apply(lambda g: pd.Series({"rms_reg":g.rms_reg_nT.mean(),"rms_out":g.rms_out_nrm_nT.mean(),"ratio":(g.rms_reg_nT/g.rms_out_nrm_nT).mean(),"Mreg/Mref":g.Mreg_over_Mref.mean()})).round(3))
