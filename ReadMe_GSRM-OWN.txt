GSRM-OWN Model
----------------------------------------------------------------------------------------------------------
Dependencies:
	Python 3
	Python packages: pip install numpy opencv-python

----------------------------------------------------------------------------------------------------------
For Testing Model:
----------------------------------------------------------------------------------------------------------
1) Using PyCharm:
	1. Copy 'GSRM_LR_IMAGES' folder in your current PyCharm project. 
	Place your own low-resolution images in 'GSRM_LR_IMAGES' folder which you want to test. 
	(There are five sample images in 'GSRM_LR_IMAGES' folder - Baby, Flower, Lemon, Nut and Nut_1).
	2. Make 'GSRM_SR_IMAGES' folder to see output images.
	3. Copy 'OWN_MODEL.h5' model in your current PyCharm project folder.
	4. Copy 'GSRM_OWN_MODEL.py'(for building model) and 'GSRM_OWN_TESTING.py'(for testing images) 
	in your current PyCharm project folder.
	5. Run 'GSRM_OWN_TESTING.py' file.
	6. You can see output images in 'GSRM_SR_IMAGES' folder.

----------------------------------------------------------------------------------------------------------
2) Using Google Colab: 
	1. Open 'https://drive.google.com/drive/folders/1iMSaZR28xMUHBdFp4eUR1twWp6oczkeT?usp=sharing' link.
	2. Download and unzip 'GSRM-OWN_Model' from above link and add it to your Google Drive.
	3. from google.colab import drive
	   drive.mount('/content/gdrive/')
	4. % cd gdrive/My Drive/folder_path
	5. cd GSRM-OWN_MODEL
	6. ! python GSRM_OWN_TESTING.py - Run this command in colab.
	7. You can see output images in 'GSRM_SR_IMAGES' folder.
----------------------------------------------------------------------------------------------------------