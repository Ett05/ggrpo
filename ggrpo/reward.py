import subprocess

def reward_function(test_cases):
    rewards = [0] * len(test_cases)
    for i, python_code in enumerate(test_cases):
        python_code = python_code[0] + "\n" + python_code[1]
        try:
            result = subprocess.run(["python", "-c", python_code], capture_output=True, text=True, timeout=1)
            if result.returncode == 0:
                rewards[i] = 1
        except subprocess.TimeoutExpired:
            rewards[i] = 0
    return rewards

