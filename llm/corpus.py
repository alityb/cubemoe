"""Builds a fixed evaluation corpus: WikiText-2 test (prose, as in Engmann et al.), Python stdlib source
(code), and license texts from installed packages (likely memorized). Saved once, reused by every model."""
import glob, json, os, random
R='/Users/alityb/projects/cubemoe/llm'
def build():
    from huggingface_hub import hf_hub_download
    import pandas as pd
    p=hf_hub_download('Salesforce/wikitext','wikitext-2-raw-v1/test-00000-of-00001.parquet',repo_type='dataset')
    wt=[t for t in pd.read_parquet(p)['text'].tolist() if len(t.split())>40 and not t.strip().startswith('=')]
    rnd=random.Random(0); rnd.shuffle(wt); prose='\n'.join(wt)
    libdir=os.path.dirname(os.__file__); files=sorted(glob.glob(f'{libdir}/*.py')); rnd.shuffle(files)
    code=[]
    for f in files:
        s=open(f,errors='ignore').read()
        if 3000<len(s)<60000: code.append(s)
    lic=[]
    for f in sorted(glob.glob(f'{libdir}/site-packages/*.dist-info/LICENSE*')+glob.glob(f'{libdir}/site-packages/*.dist-info/licenses/LICENSE*')):
        s=open(f,errors='ignore').read()
        if 800<len(s)<40000: lic.append(s)
    seen=set(); lic_u=[s for s in lic if not (s[:400] in seen or seen.add(s[:400]))]
    json.dump({'prose':prose,'code':code,'license':lic_u},open(f'{R}/corpus.json','w'))
    print(f"prose chars {len(prose)}, code files {len(code)}, distinct license texts {len(lic_u)}")
if __name__=='__main__': build()
