
import json
from pathlib import Path
import pandas as pd
from openai import OpenAI

MODEL_NAME=TBD
VLLM_URL="http://localhost:8000/v1"

BASE_DIR=Path(".")
OUT_DIR=BASE_DIR/"outputs"
OUT_DIR.mkdir(exist_ok=True)

GT_FILE=BASE_DIR/"rts_GT.xlsx"
GEN_FILE=BASE_DIR/"rts_template.xlsx"
JSON_OUT=OUT_DIR/"validation.json"
XLSX_OUT=OUT_DIR/"validation_results.xlsx"

SYSTEM_PROMPT="""You are an expert requirements validator.
Compare all generated requirements against the ground truth.
Return ONLY a JSON array.
Each object:
{
"generated_requirement":"",
"matched_ground_truth":"",
"score":0,
"reason":""
}
score: 2=exact,1=partial,0=no match.
"""

client=OpenAI(base_url=VLLM_URL,api_key="Empty")

gt=pd.read_excel(GT_FILE)
gen=pd.read_excel(GEN_FILE)

gt_list=gt["Description"].dropna().astype(str).str.strip().tolist()
gen_list=gen["Description"].dropna().astype(str).str.strip().tolist()

prompt="Ground Truth Requirements\n\n"
prompt+="\n".join(f"{i+1}. {x}" for i,x in enumerate(gt_list))
prompt+="\n\nGenerated Requirements\n\n"
prompt+="\n".join(f"{i+1}. {x}" for i,x in enumerate(gen_list))

resp=client.chat.completions.create(
    model=MODEL_NAME,
    temperature=0,
    seed=123,
    max_tokens=32768,
    messages=[
        {"role":"system","content":SYSTEM_PROMPT},
        {"role":"user","content":prompt},
    ],
)

text=resp.choices[0].message.content.strip().replace("```json","").replace("```","").strip()
Path(JSON_OUT).write_text(text,encoding="utf-8")

matches=json.loads(text)
results=pd.DataFrame(matches)

exact=(results["score"]==2).sum()
partial=(results["score"]==1).sum()
nomatch=(results["score"]==0).sum()

matched=set(results.loc[results["score"]>0,"matched_ground_truth"])
missing=[x for x in gt_list if x not in matched]
additional=results.loc[results["score"]==0,"generated_requirement"].tolist()

precision=(exact+partial)/len(gen_list) if gen_list else 0
recall=len(matched)/len(gt_list) if gt_list else 0
f1=0 if precision+recall==0 else 2*precision*recall/(precision+recall)

summary=pd.DataFrame([
["Ground Truth",len(gt_list)],
["Generated",len(gen_list)],
["Exact Matches",exact],
["Partial Matches",partial],
["No Match",nomatch],
["Missing GT",len(missing)],
["Additional Generated",len(additional)],
["Precision",precision],
["Recall",recall],
["F1",f1],
["Exact Accuracy",exact/len(gt_list) if gt_list else 0],
["Semantic Accuracy",(exact+partial)/len(gt_list) if gt_list else 0],
["Hallucination Rate",len(additional)/len(gen_list) if gen_list else 0],
["Missing Rate",len(missing)/len(gt_list) if gt_list else 0]
],columns=["Metric","Value"])

with pd.ExcelWriter(XLSX_OUT,engine="openpyxl") as w:
    results.to_excel(w,sheet_name="Results",index=False)
    summary.to_excel(w,sheet_name="Summary",index=False)

print("Done")
