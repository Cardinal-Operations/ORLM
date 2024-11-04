import os
import sys
import json

model_dir = sys.argv[1]
eval_results = {
    "model": model_dir, 
    "eval_results": []
}

dir_ents = os.listdir(model_dir)
for de in dir_ents:
    if de.startswith("eval."):
        eval_path = os.path.join(model_dir, de)
        eval_ents = os.listdir(eval_path)
        for ee in eval_ents:
            if "metrics" in ee:
                metrics_path = os.path.join(eval_path, ee)
                with open(metrics_path) as fd:
                    metrics = json.load(fd)
                    eval_results["eval_results"].append({"eval_task": de, "metrics": metrics})

eval_results = json.dumps(eval_results, indent=4)
print(eval_results)