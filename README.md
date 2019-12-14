# Dorsal-Hand-Vein-Based-Cancellable-Biometric-Authentication-System

In this project, I had to build a biometric authentication system, and need to perform authentication based on a person's vein pattern. Since veins are present inside the body, in most cases, not visible to the naked eye so this system was more secure than others. 
This project proved very beneficial for me as I learned a lot while completing it.
And I implemented it with the help of image processing and machine learning. 
> My projects mainly consisted of five steps which are as follows,
* Data Collection, Dorsal Hand Image Acquisition
* Data processing step
* Feature extraction step
* Feature protection step
* Classification step
 

So in the first step, I collected a dataset from a device, which captures the image using infrared light. 

Unlikely to oxygenated hemoglobin, deoxidized hemoglobin absorbs light at a wavelength of about 760nm, which is in the range of the near-infrared (NIR) band. Therefore, when the dorsal hand is illuminated with near-infrared light, deoxidized hemoglobin absorbs the light and appears as a black pattern in the dorsal image.



After the preprocessing step, We extract the vein pattern from the image by applying image processing. For doing this, I used a maximum curvature method. In this method, the main steps were the following.
 [1] Extraction of the center positions of veins.
 [2] Connection of the center positions.
 [3] Labeling of the image

If we store the data on the server without protection, then when someone attacks the database, they will get real identity of everyone and will be misused, so we need a feature protection step, and I achieved it by doing X-OR operation between the original image and a random matrix of the same size.

Finally, for classification, we used kernel Fisher differential analysis and obtained 93% accuracy.
