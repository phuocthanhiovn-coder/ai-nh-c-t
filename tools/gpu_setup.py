from tools.gpu_ssh import run

CMD = r'''
set -e
cd /workspace/autohdr
echo "== unzip code =="
rm -rf ai_engine && unzip -q -o code.zip && echo "ai_engine files: $(find ai_engine -name '*.py' | wc -l)"
echo "== unzip dataset =="
rm -rf ds_tmp && mkdir ds_tmp && unzip -q -o dataset_v2.zip -d ds_tmp
echo "dataset tree:"; find ds_tmp -maxdepth 3 -type d | head
# arrange data/pairs/{before,after}
rm -rf data/pairs && mkdir -p data/pairs
BEF=$(find ds_tmp -type d -name before | head -1)
AFT=$(find ds_tmp -type d -name after | head -1)
echo "before dir: $BEF | after dir: $AFT"
cp -r "$BEF" data/pairs/before
cp -r "$AFT" data/pairs/after
echo "pairs before: $(ls data/pairs/before | wc -l) | after: $(ls data/pairs/after | wc -l)"
touch ai_engine/__init__.py
echo "== conda python import check =="
/opt/conda/bin/python -c "import sys; sys.path.insert(0,'.'); import ai_engine.specialists.auto_enhance.model as m; print('model import OK')"
'''
rc, out, err = run(CMD, timeout=180)
print(out)
if err.strip():
    print("STDERR:", err[:1000])
print("EXIT", rc)
