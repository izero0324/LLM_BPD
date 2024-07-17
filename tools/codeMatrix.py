from radon.visitors import ComplexityVisitor
from radon.complexity import cc_rank, cc_visit
from radon.metrics import h_visit
from radon.raw import analyze
#v = ComplexityVisitor.from_code(code)
code = """def is_woodall(n):
    if n == 1:
        return True
    elif n % 2 == 0:
        return False
    else:
        for i in range(2, n):
            if n % i == 0:
                return False
        return True"""
result = cc_visit(code)
print(result[0][7])
