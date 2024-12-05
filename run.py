import json
import subprocess

input_jsonl_file = 'code.jsonl'
output_jsonl_file = 'result.jsonl'

results = []

with open(input_jsonl_file, 'r') as file:
    for line in file:
        record = json.loads(line)
        
        generated_code = record['generated_code']

        # delete the first line of generate_code
        generated_code_lines = generated_code.split('\n')[1:]
        generated_code = '\n'.join(generated_code_lines)

        public_tests_input = record['public_tests_input']
        public_tests_output = record['public_tests_output']

        with open("generated_code.py", "w") as code_file:
            code_file.write(generated_code)
        
        for test_input, expected_output in zip(public_tests_input, public_tests_output):
            process = subprocess.Popen(
                ["python3", "generated_code.py"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            stdout, stderr = process.communicate(input=test_input)

            result = {
                "index": record['index'],
                "input": test_input.strip(),
                "generated_output": stdout.strip(),
                "expected_output": expected_output.strip(),
                "matches": stdout.strip() == expected_output.strip()
            }
            
            results.append(result)

with open(output_jsonl_file, 'w') as outfile:
    for result in results:
        outfile.write(json.dumps(result) + "\n")

print("Testing and result saving completed!")
