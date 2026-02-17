import sys
sys.path.insert(0, 'rq1_dataset')
from refer import REFER

# Initialize
refer = REFER(
    data_root='rq1_dataset/',
    dataset='refcoco',
    splitBy='unc'
)

# Get number of referring expressions
print(f"Total expressions: {len(refer.Refs)}")

# Get one example
ref_ids = refer.getRefIds()
ref = refer.Refs[ref_ids[0]]
print(f"Expression: {ref['sentences'][0]['sent']}")