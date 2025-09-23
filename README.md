Below is a list of instructions/details on how to work the overall pipeline of this project. This will go through each step of the process to building the ImageJ plugin.

# YOLO Model Training
For training the detection model, this Google Colab notebook is used: https://colab.research.google.com/drive/1WeB_pgQrNuaPgTek0PEc_UdxGd4z4vPK?usp=sharing.

For this project a YOLO object detection model is used in order to automate detection of signals within images we feed in. In the Google Colab notebook, we load in a pre-trained YOLOv8 model and fine-tune it on an annotated dataset of signals. This dataset should be uploaded to your Google Drive where it is loaded inside of the notebook. At the end of this notebook, a cell is run to export this training yolo model as a `.onnx` file. This will be imported in the java plugin script.

Before you use your dataset folder, the labels needs to be "normalized" for yolo model to process otherwise it thinks your coordinates in the label is invalid/corrupt, to normalize your label coordinates, use this code here: https://drive.google.com/file/d/1roOvMnjkHlxnCe82bqgSaC8QxddGh4s8/view?usp=sharing
This code outputs a "label normalized" folder to your original labels folder, make sure you delete your original label folder and rename "label normalized" to "label" to replace your labels to the now normalized ones

# Building the Java Plugin
The `YOLO_detector.java` file stores the entire Java script for the ImageJ plugin. The path of the ImageJ install and the `.onnx` file are explicitly assumed within the script, so these values may need to be changed locally.

To compile the `.jar` file used by ImageJ, ensure Java is installed locally. This can be checked by running the command "java -version" in the terminal. If not, it can be installed here: https://www.oracle.com/java/technologies/downloads/.

I will detail how to build the `.jar` file through building a Maven project in VSCode, since this is how I did it. First, Maven will also need to be installed onto the computer. This tutorial is a helpful guide to installing and setting up a path variable for Maven: https://www.youtube.com/watch?v=YTvlb6eny_0.

Next, we will use VSCode to create the Maven project to compile our java file. First, the Extension Pack for Java should be installed in VSCode. Next, in the explorer tab there should be a button to "Create Java Project." To build the project, follow this path of selections:
Maven > maven-archetype-quickstart.

Next input the group id and artifact id for the project, these names are not important. Specify the directory for the project. Once this is created, the `App.java` and `pom.xml` files can be replaced by `YOLO_detector.java` and `pom.xml` files in this repository. The `Apptest.java` file can be deleted. This youtube tutorial is also a good reference for setting this up: https://www.youtube.com/watch?v=zlHXH6maOR0.
Integrating the jar file into imageJ

# Integrating the jar file into imageJ
in vs code, around the bottom left corner of your page, you should see a tab called "maven", click on the arrow/drop down and you should see the java file you are working with, clicking the drop down you should see under "lifecycle" the option "clean", clean first and click on "package"

if the code is packaged properly into a jar file, you should be able to find it in the folder /repository vs code created in your files. 

copy this jar file once you find it, and go to your imageJ program folder (it should be named something like "ij154-win-java8"), then plugins, and paste this jar file there, you might want to rename the jar so the name ends with "_.", for example, I named it "Ai _..jar"
then open imageJ, under the plugin tab, you should see your program name (like "YOLOdetector" for me), click on that and it will run

The main code for this plugin is the "YOLO-detector.java" file in this repository, I also added some more dependencies in the "pom.xml" file, which I also uploaded to the repository.

There are also my ML training dataset for electron positron in "electron-positron" folder, and for the FSS data for ML training for gamma ray detection, it is inside the "gamma ray" folder

for more details about label format for annotation, please refer to my previous email that has my files and explanations of this program, however if there are something else I forgot to mention or there are other issues, feel free to contact me

