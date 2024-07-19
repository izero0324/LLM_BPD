from radon.complexity import cc_visit
from radon.metrics import h_visit
#v = ComplexityVisitor.from_code(code)

def cal_codeMatrix(code):
    h_result = h_visit(code) # Effort = Volume x Difficulty
    cc_result = cc_visit(code) # cc complexity
    #print(h_result[0][9], cc_result[0][7])
    return h_result[0][9], cc_result[0][7]

def codeMatrix_improve(code1, code2):
    h1 , c1 = cal_codeMatrix(code1)
    h2, c2 = cal_codeMatrix(code2)
    return h1>h2, c1>c2
