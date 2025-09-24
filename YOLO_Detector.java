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
import ij.gui.GenericDialog;
import ij.gui.Overlay;

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

public class YOLO_Detector implements PlugInFilter {
    private ImagePlus image;
    private static OrtEnvironment env;
    
    private static OrtSession epSession; // For Electron Positron
    private static OrtSession gammaSession; // For Gamma Ray

    //to whoever use my code: make sure you change this for your own local file directory
    private static final String EP_MODEL_PATH = "C:/Users/admin/Downloads/ij154-win-java8/ImageJ/plugins/best.onnx";
    private static final String GAMMA_MODEL_PATH = "C:/Users/admin/Downloads/ij154-win-java8/ImageJ/plugins/gamma.onnx";

    private final int MODEL_WIDTH = 1024;
    private final int MODEL_HEIGHT = 1024;

    @Override
    public int setup(String arg, ImagePlus imp) {
        if (imp == null) {
            IJ.noImage();
            return DONE;
        }
        this.image = imp;

        if (env == null) {
            env = OrtEnvironment.getEnvironment();
        }
        
        return DOES_ALL;
    }
    
    @Override
    public void run(ImageProcessor ip) {
        // Create the dialog Window
        GenericDialog gd = new GenericDialog("Select Analysis Type");
        String[] choices = {"Electron/Positron", "Gamma Ray"};
        gd.addChoice("Analysis Workflow:", choices, choices[0]); // Adds a dropdown menu

        // Show the Dialog and Wait for User Input
        gd.showDialog();

        // Exit the plugin if clicks "Cancel"
        if (gd.wasCanceled()) {
            return;
        }

        // Get the User Choice and Run the Correct Workflow
        String userChoice = gd.getNextChoice();

        try {
            if (userChoice.equals("Electron/Positron")) {
                IJ.log("Running Electron/Positron");
                runElectronPositronWorkflow(ip);
            } else if (userChoice.equals("Gamma Ray")) {
                IJ.log("Running Gamma Ray");
                runGammaWorkflow(ip);
            }
        } catch (Exception e) {
            IJ.handleException(e);
        }
    }

    

/**
 * Manages the entire workflow for Electron/Positron (MS) images.
*/
    private void runElectronPositronWorkflow(ImageProcessor ip) throws OrtException {
        // load the E/P model only when needed
        if (epSession == null) {
            epSession = env.createSession(EP_MODEL_PATH, new OrtSession.SessionOptions());
        }

        // use a try with resources block to ensure the tensor is always closed
        try (OnnxTensor inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(preprocessImage(ip, MODEL_WIDTH, MODEL_HEIGHT)), new long[]{1, 3, MODEL_HEIGHT, MODEL_WIDTH})) {
            
            String inputName = epSession.getInputNames().iterator().next();

            // The "results" object is managed by the try with resources block also
            try (OrtSession.Result results = epSession.run(Collections.singletonMap(inputName, inputTensor))) {

                // process results
                float[][][] outputData = (float[][][]) results.get(0).getValue();
                float[][] transposedData = transpose(outputData[0]);

                //Filter detections to get the best from each class
                Map<Integer, Detection> bestDetectionsPerClass = new HashMap<>();
                float confidenceThreshold = 0.25f;
                for (float[] detectionData : transposedData) {
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
                
                // If no valid detections were found, exit early
                if (bestDetectionsPerClass.isEmpty()) {
                    IJ.log("No electron or positron signals detected above the confidence threshold");
                    return;
                }

                // create signal and background ROIs
                String[] classNames = {"electron", "positron"};
                List<Roi> signalRois = new ArrayList<>();
                List<Roi> analysisRois = new ArrayList<>();
                
                int originalWidth = ip.getWidth();
                int originalHeight = ip.getHeight();

                List<Detection> sortedDetections = new ArrayList<>(bestDetectionsPerClass.values());
                Collections.sort(sortedDetections, (d1, d2) -> Integer.compare(d1.classId, d2.classId));

                for (Detection detection : sortedDetections) {
                    float scale = Math.min((float) MODEL_WIDTH / originalWidth, (float) MODEL_HEIGHT / originalHeight);
                    int padX = (MODEL_WIDTH - Math.round(originalWidth * scale)) / 2;
                    int padY = (MODEL_HEIGHT - Math.round(originalHeight * scale)) / 2;
                    
                    float[] box = detection.box;
                    float centerX = box[0]; float centerY = box[1]; float width = box[2]; float height = box[3];
                    float unpaddedX = centerX - padX; float unpaddedY = centerY - padY;
                    int x = Math.round(unpaddedX / scale); int y = Math.round(unpaddedY / scale);
                    int w = Math.round(width / scale); int h = Math.round(height / scale);
                    int x1 = x - (w / 2); int y1 = y - (h / 2);

                    Roi signalRoi = new Roi(x1, y1, w, h);
                    String label = (detection.classId < classNames.length) ? classNames[detection.classId] : "Unknown";
                    signalRoi.setName(String.format("%s: %.2f", label, detection.confidence));
                    signalRois.add(signalRoi);

                    int padding = 10;
                    analysisRois.add(new Roi(x1 - padding, y1, w + (padding * 2), h));
                }

                // Perform analysis and plotting
                calculateAndShowEPRatio(this.image, signalRois, analysisRois);
                rotateAndPlotSignals(this.image, signalRois);
            }
        // The try with resources statement automatically closes the tensor and resultshere
        }
    }

    private void runGammaWorkflow(ImageProcessor ip) throws OrtException {
        if (gammaSession == null) {
            gammaSession = env.createSession(GAMMA_MODEL_PATH, new OrtSession.SessionOptions());
        }

        try (OnnxTensor inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(preprocessImage(ip, MODEL_WIDTH, MODEL_HEIGHT)), new long[]{1, 3, MODEL_HEIGHT, MODEL_WIDTH})) {
            
            String inputName = gammaSession.getInputNames().iterator().next();
            
            try (OrtSession.Result results = gammaSession.run(Collections.singletonMap(inputName, inputTensor))) {
                
                // filter detections
                float[][][] outputData = (float[][][]) results.get(0).getValue();
                float[][] transposedData = transpose(outputData[0]);
                Map<Integer, Detection> bestDetectionsPerClass = new HashMap<>();.
                float confidenceThreshold = 0.25f;
                for (float[] detectionData : transposedData) {
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
                
                // If no valid detections were found, exit early
                if (bestDetectionsPerClass.isEmpty()) {
                    IJ.log("No gamma ray signals detected above the confidence threshold.");
                    return;
                }
                

                // create rois from detections
                List<Roi> signalRois = new ArrayList<>();
                List<Roi> analysisRois = new ArrayList<>();
                String[] classNames = {"gamma"};
                int originalWidth = ip.getWidth();
                int originalHeight = ip.getHeight();

                for (Detection detection : bestDetectionsPerClass.values()) {
                    // logic to convert detection coordinates to ROI coordinates
                    float scale = Math.min((float) MODEL_WIDTH / originalWidth, (float) MODEL_HEIGHT / originalHeight);
                    int padX = (MODEL_WIDTH - Math.round(originalWidth * scale)) / 2;
                    int padY = (MODEL_HEIGHT - Math.round(originalHeight * scale)) / 2;
                    float[] box = detection.box;
                    float unpaddedX = box[0] - padX; float unpaddedY = box[1] - padY;
                    int x = Math.round(unpaddedX / scale); int y = Math.round(unpaddedY / scale);
                    int w = Math.round(box[2] / scale); int h = Math.round(box[3] / scale);
                    int x1 = x - (w / 2); int y1 = y - (h / 2);

                    Roi signalRoi = new Roi(x1, y1, w, h);
                    String label = (detection.classId < classNames.length) ? classNames[detection.classId] : "Unknown";
                    signalRoi.setName(String.format("%s: %.2f", label, detection.confidence));
                    signalRois.add(signalRoi);

                    int padding = 10;
                    analysisRois.add(new Roi(x1 - padding, y1, w + (padding * 2), h));
                }

                // DISPLAY ROIs AND ANALYZE
                Overlay gammaOverlay = new Overlay();
                for(Roi roi : signalRois) {
                    roi.setStrokeColor(java.awt.Color.GREEN);
                    gammaOverlay.add(roi);
                }
                this.image.setOverlay(gammaOverlay);

                for (int i = 0; i < signalRois.size(); i++) {
                    Roi currentSignalRoi = signalRois.get(i);
                    Roi currentAnalysisRoi = analysisRois.get(i);
                    
                    FitResult result = calculateSignalWithPolynomialFit(this.image, currentSignalRoi, currentAnalysisRoi);
                    
                    if (result != null) {
                        IJ.log(String.format("Signal for %s, Fit R^2: %.4f", currentSignalRoi.getName(),result.rSquared));
                        
                        // The subtracted Profile
                        Plot subtractedPlot = new Plot("Subtracted Profile: " + currentSignalRoi.getName(), "Distance (pixels)", "Subtracted Intensity");
                        subtractedPlot.add("line", result.subtractedProfile);
                        subtractedPlot.show();

                        // Get the raw profile data directly from the signal ROI
                        this.image.setRoi(currentSignalRoi);
                        double[] rawSignalProfile = new ProfilePlot(this.image).getProfile();
                        
                        // Check if the profile was created successfully
                        if (rawSignalProfile != null) {
                            Plot rawPlot = new Plot("Raw Profile: " + currentSignalRoi.getName(), "Distance (pixels)", "Raw Intensity");
                            rawPlot.add("line", rawSignalProfile);
                            rawPlot.show();
                        }
                    }
                }
            }
        }
    }
/**
 * Performs background subtraction using a polynomial fit and calculates the E/P ratio.
 */
    private void calculateAndShowEPRatio(ImagePlus imp, List<Roi> signalRois, List<Roi> analysisRois) {
        Map<String, Double> summedSignals = new HashMap<>();
        Map<String, Double> rSquaredValues = new HashMap<>();

        for (int i = 0; i < signalRois.size(); i++) {
            Roi signalRoi = signalRois.get(i);
            Roi analysisRoi = analysisRois.get(i);
            String roiName = signalRoi.getName().split(":")[0].trim();

            FitResult result = calculateSignalWithPolynomialFit(imp, signalRoi, analysisRoi);
            
            if (result != null) {
                summedSignals.put(roiName, result.summedSignal);
                rSquaredValues.put(roiName, result.rSquared);
            }
        }
            imp.killRoi();

        //calculate and display the EP ratio and R2 of the current polynomial
        if (summedSignals.containsKey("electron") && summedSignals.containsKey("positron")) {
            double electronSignal = summedSignals.get("electron");
            double positronSignal = summedSignals.get("positron");
            double electronR2 = rSquaredValues.get("electron");
            double positronR2 = rSquaredValues.get("positron");
            
            IJ.log(String.format("Electron Signal: %.4f (Fit R^2: %.4f)", electronSignal, electronR2));
            IJ.log(String.format("Positron Signal: %.4f (Fit R^2: %.4f)", positronSignal, positronR2));
            
            if (positronSignal != 0) {
                double EPratio = electronSignal / positronSignal;
                IJ.log("E/P Ratio: " + String.format("%.4f", EPratio));
                IJ.showMessage("Calculation finished", 
                    String.format("Electron Signal: %.4f (R^2: %.4f)\nPositron Signal: %.4f (R^2: %.4f)\nE/P Ratio: %.4f",
                    electronSignal, electronR2, positronSignal, positronR2, EPratio));
            } else {
                IJ.log("Positron signal is zero Cant calculate ratio");
            }
        } else {
            IJ.log("Could not calculate EP ratio. Found signals: " + summedSignals.keySet());
        }
    }

    /**
     * Rotates the image and ROIs, creates background ROIs, performs background
     * subtraction on the profile data, and plots the final result.
     */
    private void rotateAndPlotSignals(ImagePlus verticalImage, List<Roi> verticalRois) {
        // Rotate the entire image 90 degrees to the right
        ImageProcessor verticalIp = verticalImage.getProcessor();
        ImageProcessor horizontalIp = verticalIp.rotateRight();
        ImagePlus horizontalImage = new ImagePlus("Horizontal - " + verticalImage.getTitle(), horizontalIp);
        horizontalImage.show();

        // Create a new RoiManager to display all our analysis ROIs
        RoiManager rm = RoiManager.getInstance2();
        if (rm == null)
            rm = new RoiManager(); // Create it only if it doesn't exist
        rm.reset(); // clear the old vertical ROIs

        int verticalImageWidth = verticalImage.getWidth();

        // Loop through each detected signal to perform the analysis
        for (Roi verticalRoi : verticalRois) {
            java.awt.Rectangle r = verticalRoi.getBounds();

            // ROI transformation math
            int newX = r.y;
            int newY = verticalImageWidth - (r.x + r.width);
            int newWidth = r.height;
            int newHeight = r.width;
            
            // Create signal and background ROIs
            Roi signalRoi = new Roi(newX, newY, newWidth, newHeight);
            signalRoi.setName(verticalRoi.getName());
            signalRoi.setStrokeColor(java.awt.Color.YELLOW); // Set signal ROI color
            rm.addRoi(signalRoi);

            // Define a gap between the signal and background ROIs
            int backgroundGap = 3; // in pixels

            // Create ROI for background above the signal
            Roi bgRoiAbove = new Roi(newX, newY - newHeight - backgroundGap, newWidth, newHeight);
            bgRoiAbove.setName("BG_Above_" + verticalRoi.getName());
            bgRoiAbove.setStrokeColor(java.awt.Color.CYAN);
            rm.addRoi(bgRoiAbove);

            // Create ROI for background below the signal
            Roi bgRoiBelow = new Roi(newX, newY + newHeight + backgroundGap, newWidth, newHeight);
            bgRoiBelow.setName("BG_Below_" + verticalRoi.getName());
            bgRoiBelow.setStrokeColor(java.awt.Color.CYAN);
            rm.addRoi(bgRoiBelow);
            
            // Extract profile data from all 3 ROIs 
            horizontalImage.setRoi(signalRoi);
            double[] signalProfile = new ProfilePlot(horizontalImage).getProfile();
            
            horizontalImage.setRoi(bgRoiAbove);
            double[] bgProfileAbove = new ProfilePlot(horizontalImage).getProfile();
            
            horizontalImage.setRoi(bgRoiBelow);
            double[] bgProfileBelow = new ProfilePlot(horizontalImage).getProfile();
            
            // Process the Data
            // First check if all profiles are valid and have same length
            if (signalProfile == null || bgProfileAbove == null || bgProfileBelow == null || 
                signalProfile.length != bgProfileAbove.length || signalProfile.length != bgProfileBelow.length) {
                continue; // Skip to next ROI
            }
            
            // Create array to hold the final subtracted data
            double[] subtractedProfile = new double[signalProfile.length];
            
            // Loop through each data point to subtract the averaged background
            for (int i = 0; i < signalProfile.length; i++) {
                // Calculate the average background value for this x
                double avgBackground = (bgProfileAbove[i] + bgProfileBelow[i]) / 2.0;
                // Subtract it from the signal and store it
                subtractedProfile[i] = signalProfile[i] - avgBackground;
            }

            // Plot the final processed Data
            Plot finalPlot = new Plot("Subtracted Profile: " + signalRoi.getName(), "Distance (pixels)", "Subtracted Intensity");
            finalPlot.add("line", subtractedProfile); // Add processed data
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

     /**
     * Performs a polynomial background fit and calculates the summed signal, R^2 value,
     * and the subtracted profile data.
     */
    private FitResult calculateSignalWithPolynomialFit(ImagePlus imp, Roi signalRoi, Roi analysisRoi) {
        imp.setRoi(analysisRoi);
        double[] analysisProfile = new ProfilePlot(imp).getProfile();
        
        if (analysisProfile == null || analysisProfile.length < 5) {
            IJ.log("Could not generate a valid profile");
            return null;
        }

        // gather background points
        List<WeightedObservedPoint> observations = new ArrayList<>();
        int signalWidth = signalRoi.getBounds().width;
        int paddingWidth = (analysisProfile.length - signalWidth) / 2;
        for (int j = 0; j < paddingWidth; j++) {
            observations.add(new WeightedObservedPoint(1.0, j, analysisProfile[j]));
            int x_pos = signalWidth + paddingWidth + j;
            if (x_pos < analysisProfile.length) {
                observations.add(new WeightedObservedPoint(1.0, x_pos, analysisProfile[x_pos]));
            }
        }
        
        // fit the polynomial (Linear fit is often safest
        PolynomialCurveFitter fitter = PolynomialCurveFitter.create(2); // Using degree 1 (linear)
        double[] coefficients = fitter.fit(observations);
        PolynomialFunction polynomial = new PolynomialFunction(coefficients);

        //calculate r squared
        double sumOfSquaresTotal = 0, sumOfSquaresResidual = 0, meanY = 0;
        for (WeightedObservedPoint p : observations) { meanY += p.getY(); }
        meanY /= observations.size();
        for (WeightedObservedPoint p : observations) {
            double predictedY = polynomial.value(p.getX());
            sumOfSquaresTotal += Math.pow(p.getY() - meanY, 2);
            sumOfSquaresResidual += Math.pow(p.getY() - predictedY, 2);
        }
        double rSquared = (sumOfSquaresTotal > 0) ? (1.0 - (sumOfSquaresResidual / sumOfSquaresTotal)) : 1.0;

        //Calculate subtracted signal and profile
        double sumOfSubtractedSignal = 0;
        double[] subtractedProfile = new double[signalWidth];
        int signalStartInAnalysis = paddingWidth;
        for (int j = 0; j < signalWidth; j++) {
            int x_pos = j + signalStartInAnalysis;
            double signalValue = analysisProfile[x_pos];
            double backgroundValue = polynomial.value(x_pos);
            double subtractedValue = signalValue - backgroundValue;
            sumOfSubtractedSignal += subtractedValue;
            subtractedProfile[j] = subtractedValue;
        }

        return new FitResult(sumOfSubtractedSignal, rSquared, subtractedProfile);
    }

    private static class FitResult {
        final double summedSignal;
        final double rSquared;
        final double[] subtractedProfile;

        FitResult(double summedSignal, double rSquared, double[] subtractedProfile) {
            this.summedSignal = summedSignal;
            this.rSquared = rSquared;
            this.subtractedProfile = subtractedProfile;
        }
    }
}