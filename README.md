# Wolf in Sheep's clothes
![wolf_sheep](https://github.com/user-attachments/assets/29be3fda-b74c-4767-ab2d-5def2730f699)

Camouflage is one of the most powerful skills in multiple fields: it involves hiding or disguising traces for various purposes, whether to mislead competitors, protect secrets, or simply remain hidden. In the field of Machine Learning, this skill takes on new nuances thanks to generative models, which can build camouflages with previously unimaginable realism.

Now, imagine applying this idea to the world of cybersecurity: having the ability to disguise an attack as a benign event to go unnoticed, having the ability to be a Wolf in Sheep's Clothing.

This project was inspired by the book [Game Theory and Machine Learning for Cyber Security](https://www.amazon.com/Theory-Machine-Learning-Cyber-Security/dp/1119723922) by Charles A. KamhouaCharles A. Kamhoua.

This project has been used in the Medellín Machine Learning - Study Group (MML-SG) to understand the Machine Learning concepts. Therefore, the project was built end-to-end from scratch

You will find in this repo:
* Data Exploration using Jupyter to understand the data, its quality and define relevant features and their transformations.
* Model Training Pipeline:
  - Preprocess script used to apply validations, filtering, transformations, inputations, outliers removal, and normalization in the dataset.
  - Training script used to train generative models (Vanilla VAE, B-VAE, B-TCVAE, GANs, Discriminator-Classifier).
  - Evaluation script used to evaluate the model performance using dedicated metrics: Mutual information, KL Divergence, Total correlation, MSE, F1 and F1 Macro.
 
## Prerequisites
* Install Python 3.10
* Install the libraries using requirements.txt.
```bash
pip install -r requirements.txt
```
* Add the data.csv [dataset](https://drive.google.com/file/d/1UBWpyRqCG7QOQqQhUg--Yqj6wa6s2-l5/view?usp=drive_link) CSV file in the folder .\data

## Usage
Execute the discriminator script (classifier) main_pipeline_discriminator.py to train the classifier. 
```bash
python main_pipeline_discriminator.py
```
Execute the training pipelines script listed depend on the generative model to use main_pipeline_generator_\<MODEL\>.py. For example 
```bash
python main_pipeline_generator_b_tcvae.py
```

## External Resources 
This project was built by the Medellín Machine Learning - Study Group (MML-SG) community. In the following [link](https://drive.google.com/drive/u/0/folders/1nPMtg6caIef5o9S_J8WyNEvyEt5sO1VH) you can find the meetings records about it:
* [31. Airflow Introduction, Preprocessing challenges: Data Leakage and Power Transformations](https://drive.google.com/file/d/1BmleQlnB6jINohS92-u64Kx3O42VkLXS/view?usp=drive_link)
* [32. XGBoost Classifier as Antivirus AI Model, Bayesian Search CV, Entropy and Landscape Exploration - MML-SG Training - 2025/03/06 19:01 GMT-05:00 - Recording](https://drive.google.com/file/d/17jrE8oiqLKj8qTyIiqyEOXE1odhxE4Q2/view?usp=drive_link)
* [33. Antivirus AI model Trianing pipeline, Entropy, MLFlow tracking - MML-SG Training - 2025/03/12 19:03 GMT-05:00 - Recording](https://drive.google.com/file/d/1rz0vj1QgTHf_0LUrYsNoZowroDvQbhhJ/view?usp=drive_link)
* [34. Deploy model using MLFLow, Next steps for the Generative Model - MML-SG Training - 2025/03/19 18:31 GMT-05:00 - Recording](https://drive.google.com/file/d/1cl5gyX1lzlmlY-I8vH-bzXG2wi8Ikxek/view?usp=drive_link)
* [35. From Local to Global: Cloud ML models - MML-SG Training - 2025/03/26 18:53 GMT-05:00 - Recording](https://drive.google.com/file/d/1gdeQ4FljqIDfGpBf26vd9z0Fys1cBg8H/view?usp=drive_link)
* [36. Generative Models introduction, types and GenAI Model for Camouflage - MML-SG Training - 2025/04/02 19:00 GMT-05:00 - Recording](https://drive.google.com/file/d/1I-DyPzQ7voIXTcaQCcEdSnPHeoyrmG4o/view?usp=drive_link)
* [37. Variational Autoencoder Fundamentals to build Generative malicious traffic in servers Network MML-SG Training - 2025/04/24 19:00 GMT-05:00 - Recording](https://drive.google.com/file/d/1S5pGVOzc_p1wdevQy7w6IOLOW-JUAFaW/view?usp=drive_link)
* [38. VAE code - From Generative models VAE theory to practice (code) Pythorch + Optuna + MLFlow - MML-SG Training - 2025/05/01 19:00 GMT-05:00 - Recording](https://drive.google.com/file/d/1CvCZXciwsJyC0z5JCEdSprJB65VJ31-z/view?usp=drive_link)
* [39. Variational Autoencoder results for Servers traffic modeling. Challenges, Improvements, Next Steps - MML-SG Training - 2025/05/07 18:57 GMT-05:00 - Recording](https://drive.google.com/file/d/1T3e7jVemuJE-48tQ9Bc2EDwd1mDsUhi7/view?usp=drive_link)
* [40. Beta - Variational Autoencoder evaluation and Improvements from the Information Theory principles - MML-SG Training - 2025/05/14 19:00 GMT-05:00 - Recording](https://drive.google.com/file/d/1xR12lkA3cNm9qqbaCUGqFaP4tP8jnIrJ/view?usp=drive_link)
* [41. Beta - Total Correlational Variarional Autoencoder, theory, code, training challenges, inference, results - MML-SG Training - 2025/05/21 19:00 GMT-05:00 - Recording](https://drive.google.com/file/d/1fV2QkUmasEj-416NWtbvGfVRXfYNvB0b/view?usp=drive_link)
* [42. Generative Model for Malicious Traffic, The Wolf in Sheeps Clothing - MML-SG Training - 2025/06/04 18:59 GMT-05:00 - Recording](https://drive.google.com/file/d/1CcGaK87ecZDoJmHkyUkPJi-r0gCDrisB/view?usp=drive_link)

