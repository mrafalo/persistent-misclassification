import SimpleITK as sitk

import work.gui as gui
from math import ceil
from downloaddata import fetch_data as fdata
import matplotlib.pyplot as plt
import numpy as np
    
img = sitk.ReadImage("C:/datasets/COI/v2/baza/source_raw/1_1579_4.png")
gui.multi_image_display2D(image_list=[img], figure_size=(8, 4));

image = sitk.Cast(img, sitk.sitkFloat32)
  
edges = sitk.CannyEdgeDetection(
    image,
    lowerThreshold=0.0,
    upperThreshold=200.0,
    variance=(5.0, 5.0, 5.0),
)
stats = sitk.StatisticsImageFilter()
stats.Execute(image)

res_img = sitk.Maximum(image * 0.5, edges * stats.GetMaximum() * 0.5)


gui.multi_image_display2D(image_list=[image,res_img], figure_size=(8, 4));
     





edge_indexes = np.where(sitk.GetArrayViewFromImage(edges) == 1.0)

# Note the reversed order of access between SimpleITK and numpy (z,y,x)
physical_points = [
    edges.TransformIndexToPhysicalPoint([int(x), int(y), int(z)])
    for z, y, x in zip(edge_indexes[0], edge_indexes[1], edge_indexes[2])
]

# Setup and solve linear equation system.
A = np.ones((len(physical_points), 4))
b = np.zeros(len(physical_points))

for row, point in enumerate(physical_points):
    A[row, 0:3] = -2 * np.array(point)
    b[row] = -linalg.norm(point) ** 2

res, _, _, _ = linalg.lstsq(A, b)

print("The sphere's location is: {0:.2f}, {1:.2f}, {2:.2f}".format(*res[0:3]))
print(f"The sphere's radius is: {np.sqrt(linalg.norm(res[0:3])**2 - res[3]):.2f}mm")




plt.figure()
plt.hist(sitk.GetArrayViewFromImage(img).flatten(), bins=100)
plt.show()


threshold_filters = {
    "Otsu": sitk.OtsuThresholdImageFilter(),
    "Triangle": sitk.TriangleThresholdImageFilter(),
    "Huang": sitk.HuangThresholdImageFilter(),
    "MaxEntropy": sitk.MaximumEntropyThresholdImageFilter(),
}

filter_selection = "Manual"
try:
    thresh_filter = threshold_filters[filter_selection]
    thresh_filter.SetInsideValue(0)
    thresh_filter.SetOutsideValue(1)
    thresh_img = thresh_filter.Execute(img)
    thresh_value = thresh_filter.GetThreshold()
except KeyError:
    thresh_value = 120
    thresh_img = img > thresh_value

print("Threshold used: " + str(thresh_value))
gui.multi_image_display2D(
    image_list=[sitk.LabelOverlay(img, thresh_img)],
    title_list=["Binary Segmentation"],
    figure_size=(8, 4),
);


stats = sitk.LabelShapeStatisticsImageFilter()
stats.Execute(sitk.ConnectedComponent(thresh_img))

# Look at the distribution of sizes of connected components (bacteria).
label_sizes = [stats.GetNumberOfPixels(l) for l in stats.GetLabels() if l != 1]

plt.figure()
plt.hist(label_sizes, bins=200)
plt.title("Distribution of Object Sizes")
plt.xlabel("size in pixels")
plt.ylabel("number of objects")
plt.show()


cleaned_thresh_img = sitk.BinaryOpeningByReconstruction(thresh_img, [10, 10, 10])
cleaned_thresh_img = sitk.BinaryClosingByReconstruction(
    cleaned_thresh_img, [10, 10, 10]
)

gui.multi_image_display2D(
    image_list=[sitk.LabelOverlay(img, cleaned_thresh_img)],
    title_list=["Cleaned Binary Segmentation"],
    figure_size=(8, 4),
);


stats = sitk.LabelShapeStatisticsImageFilter()
stats.Execute(sitk.ConnectedComponent(cleaned_thresh_img))

# Look at the distribution of sizes of connected components (bacteria).
label_sizes = [stats.GetNumberOfPixels(l) for l in stats.GetLabels() if l != 1]

plt.figure()
plt.hist(label_sizes, bins=200)
plt.title("Distribution of Object Sizes")
plt.xlabel("size in pixels")
plt.ylabel("number of objects")
plt.show()


gui.multi_image_display2D(
    image_list=[sitk.LabelOverlay(img, sitk.ConnectedComponent(cleaned_thresh_img))],
    title_list=["Cleaned Binary Segmentation"],
    figure_size=(8, 4),
);




dist_img = sitk.SignedMaurerDistanceMap(
    cleaned_thresh_img != 0,
    insideIsPositive=False,
    squaredDistance=False,
    useImageSpacing=False,
)
radius = 10
# Seeds have a distance of "radius" or more to the object boundary, they are uniquely labelled.
seeds = sitk.ConnectedComponent(dist_img < -radius)
# Relabel the seed objects using consecutive object labels while removing all objects with less than 15 pixels.
seeds = sitk.RelabelComponent(seeds, minimumObjectSize=15)
# Run the watershed segmentation using the distance map and seeds.
ws = sitk.MorphologicalWatershedFromMarkers(dist_img, seeds, markWatershedLine=True)
ws = sitk.Mask(ws, sitk.Cast(cleaned_thresh_img, sitk.sitkUInt8))


gui.multi_image_display2D(
    image_list=[dist_img, sitk.LabelOverlay(img, seeds), sitk.LabelOverlay(img, ws)],
    title_list=[
        "Segmentation Distance",
        "Watershed Seeds",
        "Binary Watershed Labeling",
    ],
  
    horizontal=False,
    figure_size=(6, 12),
);
