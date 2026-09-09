from machinezero.data.splits import make_splits,validate_disjoint,architecture_ids
def test_architecture_splits_disjoint():
 s=make_splits(10,5,3,3); validate_disjoint(s); assert not (architecture_ids(s.train)&architecture_ids(s.test))
