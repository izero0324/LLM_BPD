import subprocess

def check_pylint(temp_filename):
    result = subprocess.run(['pylint', temp_filename, '--disable=missing-docstring'], capture_output=True, text=True, timeout=3)
    print(result.stdout)

check_pylint('tools/test_code.py')