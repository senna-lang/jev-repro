from itertools import permutations
from model.packer import Branch

def expanded_branches(rows, branches):
    result=[]
    for row, branch in zip(rows, branches):
        if row['kind']=='noul':
            orders=((0,1),(1,0))
        else:
            orders=tuple(permutations(range(len(branch.options))))[:4]
        for order in orders:
            options=[row['options'][i] for i in order]
            result.append((row, Branch(kind=branch.kind,instructions=branch.instructions,options=options if row['kind']=='noul' else options), order))
    return result
