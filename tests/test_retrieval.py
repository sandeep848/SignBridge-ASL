import numpy as np
from asl.retrieval import PrototypeIndex

def test_prototypes_rank_correct_label():
    index=PrototypeIndex().fit([[1,0],[.9,.1],[0,1]],["A","A","B"])
    result=index.query([[0,1]],k=2)
    assert result[0][0]["label"]=="B"
    assert len(result[0])==2
