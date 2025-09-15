import ij.IJ;
import ij.ImagePlus;
import ij.gui.Roi;
import ij.plugin.frame.RoiManager;
import ij.plugin.filter.PlugInFilter;
import ij.process.ImageProcessor;
import ij.gui.ProfilePlot;

import ai.onnxruntime.*;

import java.awt.Image;
import java.awt.image.BufferedImage;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.awt.Color;
import java.awt.Graphics2D;

import org.apache.commons.math3.fitting.PolynomialCurveFitter;
import org.apache.commons.math3.fitting.WeightedObservedPoint;
import org.apache.commons.math3.analysis.polynomials.PolynomialFunction;
import ij.gui.Plot;

/**
 * Helper class to store information about a single object detection. Stores three attributes per "detection"
 * classID is useful so we can determine which detection is for which class. This is dependent on how the classes
 * are ordered in the original YOLO model, be sure to check this.
 * confidence is useful to determine which detection is most confident.
 * box stores an array of the xyxy positions of the box that yolo outputs.
 */
class Detection {
    int classId;
    float confidence;
    float[] box;

    public Detection(int classId, float confidence, float[] box) {
        this.classId = classId;
        this.confidence = confidence;
        this.box = box;
    }
}

/*
 * J-COMMENT: This is the main class for our plugin.
 * 'implements PlugInFilter' tells ImageJ that this class is a standard plugin
 * that can be run on an image. It requires us to provide the 'setup' and 'run' methods.
 */
public class YOLO_Detector implements PlugInFilter {
    // J-COMMENT: 'private' variables are only accessible within this YOLO_Detector class.
    private ImagePlus image;
    private static OrtEnvironment env;
    private static OrtSession session;

    // J-COMMENT: 'final' means this variable is a constant and cannot be changed after it's set.
    private final int MODEL_WIDTH = 1024;
    private final int MODEL_HEIGHT = 1024;

    /*
     * J-COMMENT: '@Override' is an annotation that tells the compiler we are intentionally
     * replacing a method from the 'PlugInFilter' interface. This helps catch typos.
     * This 'setup' method is called by ImageJ once when the plugin starts.
     */
    @Override
    public int setup(String arg, ImagePlus imp) {
        if (imp == null) {
            IJ.noImage();
            return DONE;
        }
        this.image = imp;
        if (session == null) {
            try {
                env = OrtEnvironment.getEnvironment();
                // --- IMPORTANT: UPDATE THIS PATH ---
                String modelPath = "C:/Users/admin/Downloads/ij154-win-java8/ImageJ/plugins/best.onnx";
                session = env.createSession(modelPath, new OrtSession.SessionOptions()); // some ONNX functionality that loads our onnx file so we can feed it in images
                IJ.log("ONNX model loaded successfully.");
            } catch (OrtException e) {
                IJ.handleException(e);
                return DONE;
            }
        }
        return DOES_ALL;
    }

    @Override
    public void run(ImageProcessor ip) {
        try {
            // == 1. PRE-PROCESSING AND INFERENCE ==
            int originalWidth = ip.getWidth();
            int originalHeight = ip.getHeight();
            float[] inputData = preprocessImage(ip, MODEL_WIDTH, MODEL_HEIGHT); //run data through preprocessing function that gets the data into a format yolo model can correctly read
            OnnxTensor inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(inputData), new long[]{1, 3, MODEL_HEIGHT, MODEL_WIDTH}); //ONNX method that takes in our image data (formatted as an array of floats) and converts it into a tensor object that yolo can take as input
            String inputName = session.getInputNames().iterator().next();
            OrtSession.Result results = session.run(Collections.singletonMap(inputName, inputTensor));
            float[][][] outputData = (float[][][]) results.get(0).getValue(); // yolo model outputs a 4d array, we're only interested in the first component (.get(0)), leaving us with a 3d array
            float[][] transposedData = transpose(outputData[0]); // our 3d array still has a batch dimension (looks like (1, x, y)), so we do data[0] to get rid of the batch dimension. then we transpose so the rows represent detections

            // == 2. FILTER DETECTIONS TO GET BEST FOR EACH CLASS ==
            Map<Integer, Detection> bestDetectionsPerClass = new HashMap<>();
            /*
             * J-COMMENT: 'Map<Integer, Detection>' is a dictionary-like data structure.
             * It maps a key (the class ID, an 'Integer') to a value (the 'Detection' object).
             * 'new HashMap<>()' creates a new, empty map.
             */

            float confidenceThreshold = 0.25f;
            for (float[] detectionData : transposedData) { //assumes that transposedData is composed of an array of arrays, where each row stores info about a single detection. We iterate over every row to iterate over all detections
                float[] classScores = new float[detectionData.length - 4];
                System.arraycopy(detectionData, 4, classScores, 0, classScores.length);
                int bestClassId = -1;
                float maxScore = 0.0f;
                for (int i = 0; i < classScores.length; i++) { 
                    if (classScores[i] > maxScore) {
                        maxScore = classScores[i];
                        bestClassId = i;
                    }
                }
                if (maxScore > confidenceThreshold) {
                    float[] box = {detectionData[0], detectionData[1], detectionData[2], detectionData[3]};
                    Detection currentDetection = new Detection(bestClassId, maxScore, box);
                    if (!bestDetectionsPerClass.containsKey(bestClassId) || maxScore > bestDetectionsPerClass.get(bestClassId).confidence) {
                        bestDetectionsPerClass.put(bestClassId, currentDetection);
                    }
                }
            }

            // == 3. CREATE SIGNAL AND BACKGROUND ROIs ==
            String[] classNames = {"electron", "positron"};
            List<Roi> signalRois = new ArrayList<>(); //empty list to store detected rois
            List<Roi> analysisRois = new ArrayList<>(); // empty list to store rois for analysis (we add 2 pixels of padding on each side for interpolation)

            // Use a sorted list to ensure consistent ordering (e.g., electron then positron), sorting by class ID.
            List<Detection> sortedDetections = new ArrayList<>(bestDetectionsPerClass.values());
            Collections.sort(sortedDetections, (d1, d2) -> Integer.compare(d1.classId, d2.classId));

            for (Detection detection : sortedDetections) {
                float scale = Math.min((float) MODEL_WIDTH / originalWidth, (float) MODEL_HEIGHT / originalHeight); //scale is used to rescale xy positions from the output of the yolo model to positions on imagej image
                int padX = (MODEL_WIDTH - Math.round(originalWidth * scale)) / 2; //calculate padding made before feeding in image to undo the preprocessing done on the image
                int padY = (MODEL_HEIGHT - Math.round(originalHeight * scale)) / 2;
                
                float[] box = detection.box;
                float centerX = box[0];
                float centerY = box[1];
                float width = box[2];
                float height = box[3];

                float unpaddedX = centerX - padX;
                float unpaddedY = centerY - padY;

                int x = Math.round(unpaddedX / scale);
                int y = Math.round(unpaddedY / scale);
                int w = Math.round(width / scale);
                int h = Math.round(height / scale);
                int x1 = x - (w / 2);
                int y1 = y - (h / 2);

                // Create and name the signal ROI
                Roi signalRoi = new Roi(x1, y1, w, h);
                String label = (detection.classId < classNames.length) ? classNames[detection.classId] : "Unknown";

                signalRoi.setName(String.format("%s: %.2f", label, detection.confidence));
                signalRois.add(signalRoi);

                // Create a slightly wider ROI for analysis (2 pixels padding on each side)
                int padding = 2;
                int w_analysis = w + (padding * 2);
                int x1_analysis = x1 - padding;
                analysisRois.add(new Roi(x1_analysis, y1, w_analysis, h));
            }

            // Add signal ROIs to the manager and display them
            RoiManager rm = RoiManager.getInstance2();
            if (rm == null) rm = new RoiManager();
            rm.reset();
            for(Roi roi : signalRois) {
                rm.addRoi(roi);
            }
            rm.setVisible(true);

            // == 4. CALCULATE AND SHOW THE E/P RATIO ==
            calculateAndShowEPRatio(this.image, signalRois, analysisRois);

            rotateAndPlotSignals(this.image, signalRois);

            // == 5. CLEAN UP RESOURCES ==
            inputTensor.close();
            results.close();

        } catch (Exception e) {
            IJ.handleException(e);
        }
    }

    /**
     * Performs background subtraction using a polynomial fit and calculates the E/P ratio.
     */
    private void calculateAndShowEPRatio(ImagePlus imp, List<Roi> signalRois, List<Roi> analysisRois) {
        if (signalRois.size() < 2) {
            IJ.log("Error: Expected at least 2 detections to calculate ratio.");
            return;
        }

        // Use a map to store signals by class name for robust ratio calculation
        Map<String, Double> averagedSignals = new HashMap<>();

        for (int i = 0; i < signalRois.size(); i++) {
            Roi signalRoi = signalRois.get(i);
            Roi analysisRoi = analysisRois.get(i);

            imp.setRoi(analysisRoi);

            double[] analysisProfile = new ProfilePlot(imp).getProfile();
            // delete later if not working
            imp.setRoi(signalRoi);
            double[] signalProfile = new ProfilePlot(imp).getProfile();
            String fullName = signalRoi.getName();
            // 2. Split the name by the colon and get the first part ("electron")
            String roiName = fullName.split(":")[0].trim();
            if (analysisProfile == null || analysisProfile.length < 3) {
                IJ.log("Could not generate a valid profile for " + roiName + ". Skipping.");
                continue;
            }

            // 1. Prepare 3 specific points for the quadratic fit
            List<WeightedObservedPoint> observations = new ArrayList<>();
            
            // Point 1: First pixel (x=0)
            observations.add(new WeightedObservedPoint(1.0, 0, analysisProfile[0]));
            // Point 2: Second pixel (x=1)
            observations.add(new WeightedObservedPoint(1.0, 1, analysisProfile[1]));
            // Point 3: Last pixel (x=n-1)
            int last_x = analysisProfile.length - 1;
            observations.add(new WeightedObservedPoint(1.0, last_x, analysisProfile[last_x]));

            // 2. Fit the quadratic polynomial to the three background points
            PolynomialCurveFitter fitter = PolynomialCurveFitter.create(2); // 2nd degree polynomial
            double[] coefficients = fitter.fit(observations);
            PolynomialFunction polynomial = new PolynomialFunction(coefficients);

            // 3. Subtract the fitted background from the signal portion of the profile
            double sumOfSubtractedSignal = 0;
            int signalWidth = signalRoi.getBounds().width;
            int signalHeight = signalRoi.getBounds().height;
            int signalStartInAnalysis = (analysisProfile.length - signalWidth) / 2;
            // change to signalWidth if not working
            for (int j = 0; j < signalWidth; j++) {
                int x_pos = j + signalStartInAnalysis;
                double signalValue = analysisProfile[x_pos];
                //double backgroundValue = polynomial.value(x_pos);
                double backgroundValue = ((signalProfile[0]+signalProfile[signalProfile.length - 1])/2);
                double subtractedValue = signalValue - backgroundValue;
                sumOfSubtractedSignal += subtractedValue;
            }
            
            // 4. Calculate and store the average signal
            averagedSignals.put(roiName, sumOfSubtractedSignal);
        }

        // Restore the original full image view by removing any set ROI
        imp.killRoi();

        // 5. Robustly calculate and display the E/P Ratio
        if (averagedSignals.containsKey("electron") && averagedSignals.containsKey("positron")) {
            double electronSignal = averagedSignals.get("electron");
            double positronSignal = averagedSignals.get("positron");
            
            // Avoid division by zero
            if (positronSignal == 0) {
                IJ.log("Error: Positron signal is zero. Cannot calculate ratio.");
                return;
            }

            double EPratio = electronSignal / positronSignal;
            
            IJ.log("Electron Avg Signal: " + String.format("%.4f", electronSignal));
            IJ.log("Positron Avg Signal: " + String.format("%.4f", positronSignal));
            IJ.log("E/P Ratio: " + String.format("%.4f", EPratio));
            IJ.showMessage("Calculation Complete", "The calculated E/P Ratio is: " + String.format("%.4f", EPratio));
        } else {
            IJ.log("Could not calculate E/P ratio. Found signals: " + averagedSignals.keySet());
        }
    }

    /**
     * Rotates the image and ROIs, creates background ROIs, performs background
     * subtraction on the profile data, and plots the final result.
     * @param verticalImage The original, vertically oriented ImagePlus.
     * @param verticalRois The list of ROIs detected on the vertical image.
     */
    private void rotateAndPlotSignals(ImagePlus verticalImage, List<Roi> verticalRois) {
        // 1. Rotate the entire image 90 degrees to the right
        ImageProcessor verticalIp = verticalImage.getProcessor();
        ImageProcessor horizontalIp = verticalIp.rotateRight();
        ImagePlus horizontalImage = new ImagePlus("Horizontal - " + verticalImage.getTitle(), horizontalIp);
        horizontalImage.show();

        // 2. Create a new RoiManager to display all our analysis ROIs
        RoiManager rm = RoiManager.getInstance2();
        if (rm == null)
            rm = new RoiManager(); // Create it only if it doesn't exist
        rm.reset(); // This is important to clear the old vertical ROIs

        int verticalImageWidth = verticalImage.getWidth();

        // 3. Loop through each detected signal to perform the analysis
        for (Roi verticalRoi : verticalRois) {
            java.awt.Rectangle r = verticalRoi.getBounds();

            // --- ROI Transformation Math ---
            int newX = r.y;
            int newY = verticalImageWidth - (r.x + r.width);
            int newWidth = r.height;
            int newHeight = r.width;
            
            // --- Create Signal and Background ROIs ---
            Roi signalRoi = new Roi(newX, newY, newWidth, newHeight);
            signalRoi.setName(verticalRoi.getName());
            signalRoi.setStrokeColor(java.awt.Color.YELLOW); // Set signal ROI color
            rm.addRoi(signalRoi);

            // Define a gap between the signal and background ROIs
            int backgroundGap = 3; // 5 pixels

            // Create ROI for background ABOVE the signal
            Roi bgRoiAbove = new Roi(newX, newY - newHeight - backgroundGap, newWidth, newHeight);
            bgRoiAbove.setName("BG_Above_" + verticalRoi.getName());
            bgRoiAbove.setStrokeColor(java.awt.Color.CYAN);
            rm.addRoi(bgRoiAbove);

            // Create ROI for background BELOW the signal
            Roi bgRoiBelow = new Roi(newX, newY + newHeight + backgroundGap, newWidth, newHeight);
            bgRoiBelow.setName("BG_Below_" + verticalRoi.getName());
            bgRoiBelow.setStrokeColor(java.awt.Color.CYAN);
            rm.addRoi(bgRoiBelow);
            
            // --- Extract Profile Data from all 3 ROIs ---
            horizontalImage.setRoi(signalRoi);
            double[] signalProfile = new ProfilePlot(horizontalImage).getProfile();
            
            horizontalImage.setRoi(bgRoiAbove);
            double[] bgProfileAbove = new ProfilePlot(horizontalImage).getProfile();
            
            horizontalImage.setRoi(bgRoiBelow);
            double[] bgProfileBelow = new ProfilePlot(horizontalImage).getProfile();
            
            // --- Process the Data ---
            // First, check if all profiles are valid and have the same length
            if (signalProfile == null || bgProfileAbove == null || bgProfileBelow == null || 
                signalProfile.length != bgProfileAbove.length || signalProfile.length != bgProfileBelow.length) {
                IJ.log("Skipping profile for " + signalRoi.getName() + " due to mismatched profile lengths.");
                continue; // Skip to the next ROI
            }
            
            // Create an array to hold the final, subtracted data
            double[] subtractedProfile = new double[signalProfile.length];
            
            // Loop through each data point to subtract the averaged background
            for (int i = 0; i < signalProfile.length; i++) {
                // Calculate the average background value for this x-position
                double avgBackground = (bgProfileAbove[i] + bgProfileBelow[i]) / 2.0;
                // Subtract it from the signal and store it
                subtractedProfile[i] = signalProfile[i] - avgBackground;
            }

            // --- Plot the Final, Processed Data ---
            // Use the generic Plot class for custom data arrays
            Plot finalPlot = new Plot("Subtracted Profile: " + signalRoi.getName(), "Distance (pixels)", "Subtracted Intensity");
            finalPlot.add("line", subtractedProfile); // Add our processed data
            finalPlot.show(); // Display the plot in a new window
        }
        
        rm.setVisible(true); // Show the RoiManager with all three ROIs
        horizontalImage.killRoi(); // Clear selection from the image
    }
    private float[] preprocessImage(ImageProcessor ip, int targetWidth, int targetHeight) {
        int originalWidth = ip.getWidth();
        int originalHeight = ip.getHeight();
        float scale = Math.min((float) targetWidth / originalWidth, (float) targetHeight / originalHeight);
        int newWidth = Math.round(originalWidth * scale);
        int newHeight = Math.round(originalHeight * scale);
        Image scaledImage = ip.getBufferedImage().getScaledInstance(newWidth, newHeight, Image.SCALE_SMOOTH);
        BufferedImage paddedImage = new BufferedImage(targetWidth, targetHeight, BufferedImage.TYPE_INT_RGB);
        Graphics2D gr = paddedImage.createGraphics();
        gr.setColor(new Color(114, 114, 114));
        gr.fillRect(0, 0, targetWidth, targetHeight);
        int padX = (targetWidth - newWidth) / 2;
        int padY = (targetHeight - newHeight) / 2;
        gr.drawImage(scaledImage, padX, padY, null);
        gr.dispose();
        float[] nchwArray = new float[1 * 3 * targetHeight * targetWidth];
        int redChannelOffset = 0;
        int greenChannelOffset = targetHeight * targetWidth;
        int blueChannelOffset = 2 * targetHeight * targetWidth;
        for (int y = 0; y < targetHeight; y++) {
            for (int x = 0; x < targetWidth; x++) {
                int pixel = paddedImage.getRGB(x, y);
                float r = ((pixel >> 16) & 0xFF) / 255.0f;
                float g = ((pixel >> 8) & 0xFF) / 255.0f;
                float b = (pixel & 0xFF) / 255.0f;
                int pixelIndex = y * targetWidth + x;
                nchwArray[redChannelOffset + pixelIndex] = r;
                nchwArray[greenChannelOffset + pixelIndex] = g;
                nchwArray[blueChannelOffset + pixelIndex] = b;
            }
        }
        return nchwArray;
    }

    private float[][] transpose(float[][] data) {
        if (data == null || data.length == 0) return new float[0][0];
        int rows = data.length;
        int cols = data[0].length;
        float[][] transposed = new float[cols][rows];
        for (int i = 0; i < cols; i++) {
            for (int j = 0; j < rows; j++) {
                transposed[i][j] = data[j][i];
            }
        }
        return transposed;
    }
}