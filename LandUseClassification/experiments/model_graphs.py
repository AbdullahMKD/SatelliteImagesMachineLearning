import matplotlib.pyplot as plt
import numpy as np
from LandUseClassification.processing.kmeans_processor import kMeans_processing as kM
from LandUseClassification.processing.image_processing import image_processor as im

image_paths = ['../Data/Al-Dhannah_2018-05-21-00_00_2018-06-21-23_59_Sentinel-2_L2A_True_color.jpg',
               '../Data/Al-Dhannah_2019-05-11-00_00_2019-06-11-23_59_Sentinel-2_L2A_True_color.jpg',
               '../Data/Al-Dhannah_2020-05-15-00_00_2020-06-15-23_59_Sentinel-2_L2A_True_color.jpg',
               '../Data/Al-Dhannah_2021-05-10-00_00_2021-06-10-23_59_Sentinel-2_L2A_True_color.jpg',
               '../Data/Al-Dhannah_2022-05-20-00_00_2022-06-20-23_59_Sentinel-2_L2A_True_color.jpg']

kprocessor = kM()
img_processor = im()
silhouette_score = []
inertia = []

for k in range(2, 11):
    local_silhouette_score = []
    local_inertia = []
    for image_path in image_paths:
        kprocessor.clustering(k, img_processor.process_data(image_path))
        local_silhouette_score.append(kprocessor.silhouette_score)
        local_inertia.append(kprocessor.inertia)
    print(f"{k} computed silhouette score")
    silhouette_score.append(np.mean(local_silhouette_score))
    inertia.append(np.mean(local_inertia))

plt.plot(range(2, 11), inertia, marker='x', linestyle="--", label='Average inertia')
plt.title('Average Inertia for Different k')
plt.xlabel('k (Number of Clusters)')
plt.ylabel('Metric Value')
plt.grid(True)
plt.show()

plt.plot(range(2, 11), silhouette_score, marker='o', label='Average Silhouette score')
plt.title('Average Silhouette Scores for Different k')
plt.xlabel('k (Number of Clusters)')
plt.ylabel('silhouette Value')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.scatter(inertia, silhouette_score, color='r', marker='x')
for i, k in enumerate(range(2, 11)):
    plt.annotate(f'k={k}', (inertia[i], silhouette_score[i]))
plt.title('Silhouette Score vs Inertia')
plt.xlabel('Inertia')
plt.ylabel('Silhouette Score')
plt.grid(True)
plt.show()