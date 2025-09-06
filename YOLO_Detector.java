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

/**
 * Helper class to store information about a single object detection.
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
    private static OrtSession session;
    private final int MODEL_WIDTH = 1024;
    private final int MODEL_HEIGHT = 1024;

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
                session = env.createSession(modelPath, new OrtSession.SessionOptions());
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
            float[] inputData = preprocessImage(ip, MODEL_WIDTH, MODEL_HEIGHT);
            OnnxTensor inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(inputData), new long[]{1, 3, MODEL_HEIGHT, MODEL_WIDTH});
            String inputName = session.getInputNames().iterator().next();
            OrtSession.Result results = session.run(Collections.singletonMap(inputName, inputTensor));
            float[][][] outputData = (float[][][]) results.get(0).getValue();
            float[][] transposedData = transpose(outputData[0]);

            // == 2. FILTER DETECTIONS TO GET BEST FOR EACH CLASS ==
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

            // == 3. CREATE SIGNAL AND BACKGROUND ROIs ==
            String[] classNames = {"electron", "positron"};
            List<Roi> signalRois = new ArrayList<>();
            List<Roi> backgroundRois = new ArrayList<>();

            for (Detection detection : bestDetectionsPerClass.values()) {
                float scale = Math.min((float) MODEL_WIDTH / originalWidth, (float) MODEL_HEIGHT / originalHeight);
                int padX = (MODEL_WIDTH - Math.round(originalWidth * scale)) / 2;
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

                // Create the wider background ROI
                int w_bg = (int) (w * 1.4);
                int x1_bg = x - (w_bg / 2);
                backgroundRois.add(new Roi(x1_bg, y1, w_bg, h));
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
            calculateAndShowEPRatio(this.image, signalRois, backgroundRois);

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
    private void calculateAndShowEPRatio(ImagePlus imp, List<Roi> signalRois, List<Roi> backgroundRois) {
        if (signalRois.size() != 2) {
            IJ.log("Error: Expected exactly 2 detections (electron and positron) to calculate ratio.");
            return;
        }

        double[] avgedSignal = new double[signalRois.size()];

        for (int i = 0; i < signalRois.size(); i++) {
            Roi signalRoi = signalRois.get(i);
            Roi backgroundRoi = backgroundRois.get(i);

            // Get profile data for signal and background
            imp.setRoi(signalRoi);
            double[] signalProfile = new ProfilePlot(imp).getProfile();

            imp.setRoi(backgroundRoi);
            double[] backgroundProfile = new ProfilePlot(imp).getProfile();

            // 1. Isolate the background-only parts of the profile
            List<Double> bgOnlyValues = new ArrayList<>();
            int signalStart = (backgroundProfile.length - signalProfile.length) / 2;
            for (int j = 0; j < backgroundProfile.length; j++) {
                if (j < signalStart || j >= signalStart + signalProfile.length) {
                    bgOnlyValues.add(backgroundProfile[j]);
                }
            }

            // 2. Prepare data for polynomial fitting (x, y points)
            List<WeightedObservedPoint> observations = new ArrayList<>();
            for (int j = 0; j < bgOnlyValues.size(); j++) {
                // The 'x' is the pixel index, 'y' is the intensity
                observations.add(new WeightedObservedPoint(1.0, j, bgOnlyValues.get(j)));
            }

            // 3. Fit the polynomial to the background data
            PolynomialCurveFitter fitter = PolynomialCurveFitter.create(2); // 2nd degree polynomial
            double[] coefficients = fitter.fit(observations);
            PolynomialFunction polynomial = new PolynomialFunction(coefficients);

            // 4. Subtract the fitted background from the original signal
            double sumOfSubtractedSignal = 0;
            for (int j = 0; j < signalProfile.length; j++) {
                // The 'x' value for the polynomial is the pixel's position within the signal profile
                double backgroundValue = polynomial.value(j + signalStart);
                double subtractedValue = signalProfile[j] - backgroundValue;
                sumOfSubtractedSignal += subtractedValue;
            }
            
            // 5. Calculate the average signal after subtraction
            avgedSignal[i] = sumOfSubtractedSignal / signalProfile.length;
        }
        imp.killRoi();
        // 6. Calculate and display the E/P Ratio
        // NOTE: This assumes the first ROI is the electron and the second is the positron.
        // A more robust implementation would check the ROI names.
        double EPratio = avgedSignal[0] / avgedSignal[1];
        
        IJ.log("Electron Avg Signal: " + avgedSignal[0]);
        IJ.log("Positron Avg Signal: " + avgedSignal[1]);
        IJ.log("E/P Ratio: " + EPratio);
        IJ.showMessage("Calculation Complete", "The calculated E/P Ratio is: " + String.format("%.4f", EPratio));
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