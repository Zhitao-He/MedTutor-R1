import json

file_path = 'Our_framework/Evaluation/Results/'

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    mps_scores = []
    ets_scores = []
    msm_scores = []

    for item in data:
        if 'MPS' in item and 'Score' in item['MPS']:
            mps_scores.append(item['MPS']['Score'])
        if 'ETS' in item and 'Score' in item['ETS']:
            ets_scores.append(item['ETS']['Score'])
        if 'MSM' in item and 'Score' in item['MSM']:
            msm_scores.append(item['MSM']['Score'])

    avg_mps = sum(mps_scores) / len(mps_scores) if mps_scores else 0
    avg_ets = sum(ets_scores) / len(ets_scores) if ets_scores else 0
    avg_msm = sum(msm_scores) / len(msm_scores) if msm_scores else 0


    print(f"ETS: {avg_ets}")
    print(f"MSM: {avg_msm}")
    print(f"MPS: {avg_mps}")

except FileNotFoundError:
    print(f"Error {file_path}")
except json.JSONDecodeError:
    print(f"Error {file_path}")
except Exception as e:
    print(f"Error: {e}")