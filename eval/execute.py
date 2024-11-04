import subprocess
import os
import json
import tempfile
import concurrent.futures
import argparse

from collections import Counter

ADD_SCRIPT = '\nif model.status == COPT.OPTIMAL:\n    print(f"Just print the best solution: {model.objval}")\nelse:\n    print("No Best Solution")'

def majority_voting(pred_answers):
    # Count occurrences of each item in the list
    count = Counter(pred_answers)
    # Find the answer with the maximum count
    max_count = max(count.values())
    # Extract all answers with the maximum count
    possible_answers = [answer for answer, cnt in count.items() if cnt == max_count]
    # Return the first answer with the maximum count
    return possible_answers[0]

def compile_script(script_content, timeout=300):
    # Ensure the target directory exists
    target_dir = './eval_execute'
    os.makedirs(target_dir, exist_ok=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix='.py', dir=target_dir) as tmp_file:
        tmp_file_name = tmp_file.name
        tmp_file.write(script_content.encode())

    try:
        # Running the Lean3 compiler on the temporary script file with a time limit
        process = subprocess.run(['python', tmp_file_name], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, check=True)
        # If compilation is successful, return the output and a success message
        execution_result = process.stdout
        execution_best_solution_start_pos = execution_result.find("Just print the best solution:")
        if execution_best_solution_start_pos != -1:
            execution_best_solution = execution_result[execution_best_solution_start_pos:].replace("Just print the best solution:", "").strip()
            execution_best_solution_end_pos = execution_best_solution.find("\n")
            if execution_best_solution_end_pos != -1:
                execution_best_solution = execution_best_solution[:execution_best_solution_end_pos]
            execution_state = "Execution Successful and Best Solution Found"
        else:
            if "No Best Solution" in execution_result:
                execution_best_solution = "No Best Solution"
                execution_state = "Execution Successful but No Best Solution Found"
            else:
                execution_best_solution = None
                execution_state = "Execution Suceessful but Out of Expectation"
    except subprocess.TimeoutExpired as e:
        # If compilation time exceeds the limit, kill the process and return a failure message
        execution_result = e.stdout
        execution_best_solution = None
        execution_state = "Execution Failed: Timeout"
    except subprocess.CalledProcessError as e:
        # If compilation fails for other reasons, return the error output
        execution_result = e.stdout
        execution_best_solution = None
        execution_state = f"Execution Failed: {e.stdout}"
    finally:
        # Clean up the temporary file
        os.remove(tmp_file_name)

    execution_output = {
        "execution_result": execution_result,
        "execution_best_solution": execution_best_solution, 
        "execution_state": execution_state
    }
    return execution_output

def main(args):
    # Check version
    process = subprocess.run(['python', '--version'], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10, check=True)
    print(process.stdout)
    process = subprocess.run(['python', '-c', 'import coptpy'], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10, check=True)
    print(process.stdout)
    print("check coptpy installed.")

    # Load scripts to compile
    early_failed = []
    to_run = []
    with open(args.input_file) as fd:
        for line in fd:
            example = json.loads(line)
            code_field = None
            for key in example.keys():
                if "coptpy_code" in key:
                    code_field = key
                    break
            assert code_field is not None

            output = example[code_field]
            example_t = {k: v for k, v in example.items()}
            start = output.find("```python")
            if start == -1:
                execution_output = {
                    "execution_result": "Execution Failed: No code",
                    "execution_best_solution": None, 
                    "execution_state": "Execution Failed: No code"
                }
                example_t.update(execution_output)
                early_failed.append(example_t)
                continue
            end = output.find("```", start + 9)
            script = output[start:end].replace("```python", "")
            if script.strip() == "":
                execution_output = {
                    "execution_result": "Execution Failed: No code",
                    "execution_best_solution": None, 
                    "execution_state": "Execution Failed: No code"
                }
                example_t.update(execution_output)
                early_failed.append(example_t)
                continue
            script += ADD_SCRIPT
            # print(script)
            # raise
            example_t["to_run_script"] = script
            to_run.append(example_t)
    # to_run = to_run[0:10]
    print(f"len(to_run): {len(to_run)}")
    # raise

    # Function to process each example
    def process_example(example):
        execution_output = compile_script(example["to_run_script"])
        # print(f"-" * 20 + "compilation result" + "-" * 20)
        # print(execution_output["execution_result"])
        # print("-" * 10)
        # print(execution_output["execution_best_solution"])
        # print("-" * 10)
        # print(execution_output["execution_state"])
        example.update(execution_output)
        return json.dumps(example, ensure_ascii=False)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Submitting all the tasks to the executor
        future_to_example = {executor.submit(process_example, example): example for example in to_run}

        # Writing the results to file as they are completed
        with open(args.output_file, "w", encoding='utf-8') as fw:
            for example in early_failed:
                dump = json.dumps(example, ensure_ascii=False)
                fw.write(dump + "\n")

            for future in concurrent.futures.as_completed(future_to_example):
                try:
                    result = future.result()
                    fw.write(result + "\n")
                except Exception as exc:
                    print(f'An error occurred: {exc}')
                    continue
    print("Execution completed.")

    if (args.question_field is not None) and (args.answer_field is not None):
        question2pred_answers = {}
        question2gt_answers = {}
        judges = []
        with open(args.output_file, "r") as fd:
            for line in fd:
                example = json.loads(line)
                question = example[args.question_field]
                if question not in question2pred_answers:
                    question2pred_answers[question] = []
                if question not in question2gt_answers:
                    question2gt_answers[question] = []
                
                gt_answer = example[args.answer_field]
                question2gt_answers[question].append(gt_answer)

                pred_answer = example["execution_best_solution"]
                question2pred_answers[question].append(pred_answer)
        
        k = -1
        for question, pred_answers in question2pred_answers.items():
            k = len(pred_answers)

            gt_answers = question2gt_answers[question]
            assert len(set(gt_answers)) == 1
            gt_answer = gt_answers[0]

            is_anyone_match = False
            for pred_answer in pred_answers:
                if gt_answer == "No Best Solution":
                    if pred_answer is not None and pred_answer == gt_answer:
                        is_anyone_match = True
                        break
                else:
                    gt_answer = round(float(gt_answer))
                    if pred_answer is not None and pred_answer != "No Best Solution":
                        pred_answer = round(float(pred_answer))
                        if gt_answer == 0:
                            close_enough = abs(pred_answer) <= args.numerical_err_tolerance
                        else:
                            close_enough = abs((pred_answer - gt_answer) / gt_answer) <= args.numerical_err_tolerance
                        if close_enough:
                            is_anyone_match = True
                            break
            
            if is_anyone_match:
                judges.append(1)
            else:
                judges.append(0)

            if args.verbose:
                print("-" * 60)
                print("-" * 20 + "question" + "-" * 20)
                print(question)
                print("-" * 20 + "pred_answers" + "-" * 20)
                print(pred_answers)
                print("-" * 20 + "gt_answer" + "-" * 20)
                print(gt_answer)
                print("-" * 20 + "judge" + "-" * 20)
                print(is_anyone_match)
        acc = sum(judges) / len(judges)
        metrics = {f"pass@{k}": acc}

        if args.majority_voting:
            mj_judges = []
            for question, pred_answers in question2pred_answers.items():
                k = len(pred_answers)

                gt_answers = question2gt_answers[question]
                assert len(set(gt_answers)) == 1
                gt_answer = gt_answers[0]

                pred_answers_t = []
                for pred_answer in pred_answers:
                    if pred_answer is None:
                        continue
                    try:
                        pred_answer = round(float(pred_answer))
                        pred_answers_t.append(pred_answer)
                    except:
                        pred_answers_t.append(pred_answer)
                if pred_answers_t != []:
                    mj_answer = majority_voting(pred_answers_t)
                else:
                    mj_answer = None

                
                is_mj_match=False
                if gt_answer == "No Best Solution":
                    if mj_answer is not None and mj_answer == gt_answer:
                        is_mj_match = True
                else:
                    gt_answer = round(float(gt_answer))
                    if mj_answer is not None and mj_answer != "No Best Solution":
                        if gt_answer == 0:
                            close_enough = abs(mj_answer) <= args.numerical_err_tolerance
                        else:
                            close_enough = abs((mj_answer - gt_answer) / gt_answer) <= args.numerical_err_tolerance
                        if close_enough:
                            is_mj_match = True
                if args.verbose:
                    print(f"gt_answer: {gt_answer}; pred_answers_t: {pred_answers_t}; mj_answer: {mj_answer}; is_mj_match: {is_mj_match}")
                
                if is_mj_match:
                    mj_judges.append(1)
                else:
                    mj_judges.append(0)

            mj_acc = sum(mj_judges) / len(mj_judges)
            metrics[f"mj@{k}"] = mj_acc

        if args.output_file.endswith(".json"):
            metrics_file = args.output_file.replace(".json", ".metrics.json")
        elif args.output_file.endswith(".jsonl"):
            metrics_file = args.output_file.replace(".jsonl", ".metrics.json")
        else:
            metrics_file = args.output_file + ".metrics.json"
        with open(metrics_file, "w") as fw:
            dump = json.dumps(metrics, indent=4)
            fw.write(dump)
            print(dump)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str) 
    parser.add_argument("--output_file", type=str, default=None) 
    parser.add_argument("--timeout", type=int, default=600) 
    parser.add_argument("--max_workers", type=int, default=16)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--majority_voting", action="store_true")
    parser.add_argument("--question_field", type=str, default=None)
    parser.add_argument("--answer_field", type=str, default=None) 
    parser.add_argument("--numerical_err_tolerance", type=float, default=0.05)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
