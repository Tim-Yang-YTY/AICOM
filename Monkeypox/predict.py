import csv
from test_automation import prediction

results = []
for i in range(1, 3529):
	print(f"\033[1;32mImage testing_ds_others_img_{i}\033[0m")
	imagePath = f"6124_testing_dataset/Others/testing_ds_others_img_{i}..jpg"
	row = [f"Image testing_ds_others_img_{i}"]
	for iteration in range(1, 6):
		print(f"Iteration {iteration}")
		result = prediction(f"I{iteration}", imagePath)
		row.extend(result)
	results.append(row)
	
with open('predictions.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Image"] + ["Monkeypox", "Others", "Time"] * 5)
    writer.writerows(results)