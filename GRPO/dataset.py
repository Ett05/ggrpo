problems = [
    {"prompt": "Write a function called add that takes two integers and returns their sum.",
     "test": "assert add(2, 3) == 5\nassert add(-1, 1) == 0\nassert add(0, 0) == 0"},
    
    {"prompt": "Write a function called is_even that returns True if a number is even, False otherwise.",
     "test": "assert is_even(4) == True\nassert is_even(3) == False\nassert is_even(0) == True"},
    
    {"prompt": "Write a function called reverse_string that reverses a string.",
     "test": "assert reverse_string('hello') == 'olleh'\nassert reverse_string('a') == 'a'"},
    
    {"prompt": "Write a function called max_of_three that returns the largest of three numbers.",
     "test": "assert max_of_three(1, 2, 3) == 3\nassert max_of_three(5, 5, 5) == 5"},
    
    {"prompt": "Write a function called factorial that returns the factorial of n.",
     "test": "assert factorial(0) == 1\nassert factorial(5) == 120"},
    
    {"prompt": "Write a function called is_palindrome that returns True if a string is a palindrome.",
     "test": "assert is_palindrome('racecar') == True\nassert is_palindrome('hello') == False"},
    
    {"prompt": "Write a function called count_vowels that counts vowels in a string.",
     "test": "assert count_vowels('hello') == 2\nassert count_vowels('aeiou') == 5"},
    
    {"prompt": "Write a function called flatten that flattens a list of lists into a single list.",
     "test": "assert flatten([[1,2],[3,4]]) == [1,2,3,4]\nassert flatten([[1],[2],[3]]) == [1,2,3]"},
]

system = "You are a Python coding assistant. Return ONLY the Python function, no explanation, no markdown, no code blocks. Just the raw Python code."
num_return_sequences=8
    