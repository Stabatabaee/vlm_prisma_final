import json

# 1) Load your existing expected.json
expected = json.load(open('expected.json', 'r', encoding='utf-8'))

# 2) Collect the set of all keys you actually want to keep
#    (we assume every entry has a "file" key plus all the other real fields)
keep_keys = set()
for doc in expected:
    keep_keys.update(doc.keys())
# e.g. {'file','Title','Year',...,'Clinical Metrics'}

# 3) Prune prompts.json
prompts = json.load(open('prompts.json', 'r', encoding='utf-8'))
pruned_prompts = {k: prompts[k] for k in prompts if k in keep_keys}
with open('prompts_pruned.json', 'w', encoding='utf-8') as f:
    json.dump(pruned_prompts, f, indent=2, ensure_ascii=False)

# 4) Prune expected.json
pruned_expected = []
for doc in expected:
    pruned_expected.append({k: doc[k] for k in doc if k in keep_keys})
with open('expected_pruned.json', 'w', encoding='utf-8') as f:
    json.dump(pruned_expected, f, indent=2, ensure_ascii=False)

print("Wrote prompts_pruned.json and expected_pruned.json")
