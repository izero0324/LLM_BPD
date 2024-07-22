import os
import subprocess
import tempfile

def check_pylint(code):
    with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as f:
        func_code = code
        f.write(f"{func_code}")
        temp_filename = f.name
    result = subprocess.run(['pylint', temp_filename, '--disable=missing-docstring'], capture_output=True, text=True, timeout=3)
    print(result.stdout.split())
    os.remove(temp_filename)
    return result.stdout.split()

#check_pylint('tools/test_code.py')