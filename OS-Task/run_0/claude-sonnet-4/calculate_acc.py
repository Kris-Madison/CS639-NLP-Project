#!/usr/bin/env python3
import json
import os

results = {}

# 读取所有 overall.json
for task_num in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    task_dir = f"os-std-aug-clear-filtered-{task_num}"
    overall_file = os.path.join(task_dir, "overall.json")
    runs_file = os.path.join(task_dir, "runs.jsonl")
    
    if os.path.exists(overall_file):
        with open(overall_file, 'r') as f:
            data = json.load(f)
            results[f"Task {task_num}"] = {
                'total': data['total'],
                'pass': data['custom']['overall']['pass'],
                'wrong': data['custom']['overall']['wrong'],
                'acc': data['custom']['overall']['acc']
            }
    elif os.path.exists(runs_file):
        # 从 runs.jsonl 计算
        total = 0
        pass_count = 0
        wrong_count = 0
        with open(runs_file, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        result = data.get('output', {}).get('result', {})
                        if isinstance(result, dict) and 'result' in result:
                            total += 1
                            if result['result'] == True:
                                pass_count += 1
                            elif result['result'] == False:
                                wrong_count += 1
                    except:
                        pass
        if total > 0:
            results[f"Task {task_num}"] = {
                'total': total,
                'pass': pass_count,
                'wrong': wrong_count,
                'acc': pass_count / total
            }

# 打印
print("=" * 100)
print("Claude Sonnet 4 准确率汇总表")
print("=" * 100)
print()
print(f"{'Task':<12} {'Total':<8} {'Pass':<8} {'Wrong':<8} {'Accuracy':<12}")
print("-" * 100)

total_all = 0
pass_all = 0
wrong_all = 0

for task_name in sorted(results.keys(), key=lambda x: int(x.split()[1])):
    r = results[task_name]
    total_all += r['total']
    pass_all += r['pass']
    wrong_all += r['wrong']
    print(f"{task_name:<12} {r['total']:<8} {r['pass']:<8} {r['wrong']:<8} {r['acc']*100:>8.2f}%")

print("-" * 100)
overall_acc = pass_all / total_all if total_all > 0 else 0
print(f"{'OVERALL':<12} {total_all:<8} {pass_all:<8} {wrong_all:<8} {overall_acc*100:>8.2f}%")
print("=" * 100)



