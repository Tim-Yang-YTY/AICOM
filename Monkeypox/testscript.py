import csv
from testprediction import predict_image

resources = 'southeastasia.api.cognitive.microsoft.com'
projectId = '7db98f08-4938-4a3c-bfec-6c82b52d7fe9'
predictionKey = '1c3e003089e54d4f83ea0af548cf85b7'

resources1 = 'southcentralus.api.cognitive.microsoft.com'
projectId1 = 'abdc4481-38db-4bb0-b3d2-772cac927696'
predictionKey1 = '8890ab6c61d649688cf22b12a81515da'
it = ''

results = []
for i in range(1,4):
    print(f"\033[1;32mImage testing_ds_others_img_{i}\033[0m")
    image = f"6124_testing_dataset/Others/testing_ds_others_img_{i}..jpg"
    row = [f"Image testing_ds_others_img_{i}"]
    for iteration in range(1, 6):
        print(f"Iteration {iteration}")
        if iteration <= 3:
            resources_i = resources
            projectId_i = projectId
            predictionKey_i = predictionKey
            it = iteration
        else:
            if iteration == 4:
                resources_i = resources1
                projectId_i = projectId1
                predictionKey_i = predictionKey1
                it = 1
            if iteration == 5:
                resources_i = resources1
                projectId_i = projectId1
                predictionKey_i = predictionKey1
                it = 2
        result = predict_image(image, resources_i, projectId_i, f"Iteration{it}", predictionKey_i)
        row.extend(result)
    results.append(row)

with open('predictions.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Image"] + ["Monkeypox", "Others", "Time"] * 5)
    writer.writerows(results)

