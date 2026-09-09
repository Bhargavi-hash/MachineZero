from machinezero.data.encoding import encode_transition,FEATURE_DIM
from machinezero.aliencpu.state import CPUState
from machinezero.aliencpu.instruction import Instruction
def test_encoding_no_seed_or_id():
 x=encode_transition(CPUState([1,2,3,4]),Instruction(17,0,1),None,8,4,True); assert x.shape==(FEATURE_DIM,)
