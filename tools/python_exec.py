import tempfile
import subprocess
import resource
from time import perf_counter

# Testing data dictionary
data = {
    "text": "Write a function to count the most common words in a dictionary.",
    "code": "from collections import Counter\r\ndef count_common(words):\r\n  word_counts = Counter(words)\r\n  top_four = word_counts.most_common(4)\r\n  return (top_four)\r\n",
    "task_id": 13,
    "test_setup_code": "",
    "test_list": [
        "assert count_common(['red','green','black','pink','black','white','black','eyes','white','black','orange','pink','pink','red','red','white','orange','white','black','pink','green','green','pink','green','pink','white','orange',\"orange\",'red']) == [('pink', 6), ('black', 5), ('white', 5), ('red', 4)]",
        "assert count_common(['one', 'two', 'three', 'four', 'five', 'one', 'two', 'one', 'three', 'one']) == [('one', 4), ('two', 2), ('three', 2), ('four', 1)]",
        "assert count_common(['Facebook', 'Apple', 'Amazon', 'Netflix', 'Google', 'Apple', 'Netflix', 'Amazon']) == [('Apple', 2), ('Amazon', 2), ('Netflix', 2), ('Facebook', 1)]"
    ],
    "challenge_test_list": []
}

def test_code(data, debug = False, warmup = 0):
    # Create the temporary python file with the function and the tests
    with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as f:
        func_code = data['code']
        tests = '\n'.join(data['test_list'])
        f.write(f"{func_code}\n{tests}\n")
        temp_filename = f.name

    #run warmups
    if warmup > 0:
        subprocess.run(['python3', temp_filename], capture_output=True, text=True, timeout=10)
        warmup -= 1

    start_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0/1024.0
    start_time = perf_counter()
    # Run the temporary python file and capture output
    result = subprocess.run(['python3', temp_filename], capture_output=True, text=True, timeout=10)
    stop_time = perf_counter()
    run_time = stop_time - start_time
    memMb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0
    mem_used_kb = (memMb- start_mem) # KByte
    
    flake8_result = subprocess.run(['flake8', temp_filename, '--count'], capture_output=True, text=True, timeout=10)
    flake8_errors = flake8_result.stdout.split("\n")[-2]
    if debug:
        # Print the results
        if result.returncode == 0:
            print("All tests passed!")
        else:
            print("Tests failed:")
            print(result.stdout)
            print(result.stderr)

    # Delete temporary file 
    try:
        import os
        os.remove(temp_filename)
    except OSError as e:
        print(f"Error: {e.strerror}")

    return 1 - result.returncode, run_time, result.stderr, flake8_errors, mem_used_kb
