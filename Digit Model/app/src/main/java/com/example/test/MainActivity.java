package com.example.test;

import android.annotation.SuppressLint;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.media.ThumbnailUtils;
import android.net.Uri;
import android.os.AsyncTask;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.Manifest;



import androidx.activity.EdgeToEdge;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.documentfile.provider.DocumentFile;


import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.io.File;
import java.util.Random;

import okhttp3.Call;
import okhttp3.Callback;
import okhttp3.MediaType;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

public class MainActivity extends AppCompatActivity {
    Button camera, train_gallery,gallery,send,load;
    ImageView imageView;
    TextView result;
    int imageSize = 28;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);
        camera = findViewById(R.id.take_pic_button);
        train_gallery = findViewById(R.id.train_gal_button);
        gallery=findViewById(R.id.lauch_gal_button);
        send =findViewById(R.id.send_button);
        load =findViewById(R.id.load_button);

        result=findViewById(R.id.result);

        imageView=findViewById(R.id.imageView);

        camera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if(checkSelfPermission(Manifest.permission.CAMERA)== PackageManager.PERMISSION_GRANTED){
                    Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                    startActivityForResult(cameraIntent,3);
                }
                else{
                    requestPermissions(new String[]{Manifest.permission.CAMERA},100);
                }
            }
        });
        gallery.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent galleryIntent = new Intent(Intent.ACTION_PICK,MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                startActivityForResult(galleryIntent,2);
            }
        });
        train_gallery.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                Intent folderPickerIntent = new Intent(Intent.ACTION_OPEN_DOCUMENT_TREE);
                startActivityForResult(folderPickerIntent, 6);
            }
        });
        send.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                System.out.println("Starting sending weights");
                sendWeightsToServer();
            }
        });
        load.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                loadModel();
            }
        });
    }
    private String serverIp(){
        String serverIp ="";
        try {
            InputStream is = getAssets().open("config.txt");
            int size = is.available();
            byte[] buffer = new byte[size];
            is.read(buffer);
            is.close();
            serverIp = new String(buffer);
            serverIp = serverIp.trim();


        } catch (IOException e) {
            e.printStackTrace();

        }
        return serverIp;
    }
    @SuppressLint("StaticFieldLeak")
    private void loadModel() {
        new  AsyncTask<Void, String, Boolean>() {
            @Override
            protected Boolean doInBackground(Void... voids) {

                System.out.println("Starting loadModel() method");
                try {


                    File checkpointFile = new File(getFilesDir(), "checkpoint.ckpt");
                    if (checkpointFile.exists()) {
                        boolean deleted = checkpointFile.delete();
                        if (deleted) {
                            System.out.println("Checkpoint file deleted successfully");
                        } else {
                            System.out.println("Failed to delete checkpoint file");
                        }
                    }

                    File modelFile = new File(getFilesDir(), "classifier_model.tflite");
                    if (modelFile.exists()) {
                        boolean deleted = modelFile.delete();
                        if (deleted) {
                            System.out.println("Existing classifier_model.tflite file deleted successfully");
                        } else {
                            System.out.println("Failed to delete existing classifier_model.tflite file");
                            return false;
                        }
                    }

                    File modelFile2 = new File(getFilesDir(), "feature_extractor_model.tflite");
                    if (modelFile2.exists()) {
                        boolean deleted = modelFile2.delete();
                        if (deleted) {
                            System.out.println("Existing feature_extractor_model.tflite file deleted successfully");
                        } else {
                            System.out.println("Failed to delete existing feature_extractor_model.tflite file");
                            return false;
                        }
                    }
                    System.out.println("Initializing OkHttpClient");
                    OkHttpClient client = new OkHttpClient();
                    String serverIp = serverIp();
                    String url = String.format("http://%s:5000/load_model", serverIp);
                    System.out.println("Building request to "+ url);
                    // Build the request
                    Request request = new Request.Builder()
                            .url(url)
                            .build();

                    System.out.println("Executing request...");
                    Response response = client.newCall(request).execute();

                    if (!response.isSuccessful()) {
                        throw new IOException("Unexpected code " + response);
                    }
                    System.out.println("Request successful. Response code: " + response.code());


                    System.out.println("Preparing to write model file to: " + modelFile.getAbsolutePath());
                    FileOutputStream fos = new FileOutputStream(modelFile);
                    System.out.println("Writing bytes to model file");
                    fos.write(response.body().bytes());
                    fos.close();
                    System.out.println("Model file written successfully");
                    if (modelFile.length() == 0) {
                        System.out.println("Downloaded model file is empty");
                        return false;
                    }
                    FileInputStream fis = new FileInputStream(modelFile);
                    byte[] header = new byte[10];
                    fis.read(header);
                    fis.close();
                    System.out.println("File header (first 10 bytes): " + Arrays.toString(header));

                    try {
                        Interpreter tflite = new Interpreter(modelFile);
                        System.out.println("Model loaded successfully immediately after download");
                        tflite.close();
                    } catch (Exception e) {
                        System.out.println("Error loading model immediately after download: " + e.getMessage());
                        e.printStackTrace();
                    }

                    return true;
                } catch (IOException e) {
                    System.out.println("Error: " + e.getMessage());
                    e.printStackTrace();
                    return false;
                }
            }
        }.execute();
    }

    private void sendWeightsToServer() {
        try {
            File checkpointFile = new File(getFilesDir(), "checkpoint.ckpt");
            if (!checkpointFile.exists()) {
                System.err.println("Checkpoint file not found: " + checkpointFile.getAbsolutePath());
                return;
            }
            System.out.println("Checkpoint file found: " + checkpointFile.getAbsolutePath());

            byte[] modelWeights = new byte[(int) checkpointFile.length()];
            FileInputStream fis = new FileInputStream(checkpointFile);
            fis.read(modelWeights);
            fis.close();

            System.out.println("Weights successfully read from file, size: " + modelWeights.length + " bytes.");

            OkHttpClient client = new OkHttpClient();
            RequestBody requestBody = RequestBody.create(MediaType.parse("application/octet-stream"),modelWeights);
            String serverIp = serverIp();
            String url = String.format("http://%s:5000/upload-weights", serverIp);
            System.out.println("Building request to "+ url);
            Request request = new Request.Builder()
                    .url(url)
                    .post(requestBody)
                    .build();

            client.newCall(request).enqueue(new Callback() {
                @Override
                public void onFailure(Call call, IOException e) {
                    System.err.println("Error sending weights: " + e.getMessage());
                    e.printStackTrace();
                }

                @Override
                public void onResponse(Call call, Response response) throws IOException {
                    if (response.isSuccessful()) {
                        System.out.println("Weights successfully sent to the server.");
                    } else {
                        System.err.println("Failed to send weights: " + response.message());
                    }
                }
            });

        } catch (IOException e) {
            System.err.println("IOException occurred: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private void copyModelFromAssetsIfNecessary(String modelFileName) {
        File modelFile = new File(getFilesDir(), modelFileName);

        if (!modelFile.exists()) {
            try {
                // Відкриваємо модель в активах
                InputStream inputStream = getAssets().open(modelFileName);
                FileOutputStream outputStream = new FileOutputStream(modelFile);

                byte[] buffer = new byte[1024];
                int length;
                while ((length = inputStream.read(buffer)) > 0) {
                    outputStream.write(buffer, 0, length);
                }

                outputStream.close();
                inputStream.close();
                System.out.println("Model copied to " + modelFile.getAbsolutePath());
            } catch (IOException e) {
                e.printStackTrace();
                System.out.println("Failed to copy model from assets: " + e.getMessage());
            }
        } else {
            System.out.println("Model file already exists at " + modelFile.getAbsolutePath());
        }
    }
    public void classifyImageWithInference(Bitmap image) {
        try {
            String Cl_modelFileName = "classifier_model.tflite";
            String FE_modelFileName = "feature_extractor_model.tflite";
            copyModelFromAssetsIfNecessary(Cl_modelFileName);
            copyModelFromAssetsIfNecessary(FE_modelFileName);

            File featureExtractorFile = new File(getFilesDir(), FE_modelFileName);
            File classifierFile = new File(getFilesDir(), Cl_modelFileName);

            if (!featureExtractorFile.exists()) {
                System.out.println("TFLite model file '" + FE_modelFileName + "' not found.");
                return;
            }
            if (!classifierFile.exists()) {
                System.out.println("TFLite model file '" + Cl_modelFileName + "' not found.");
                return;
            }
            System.out.println("TFLite models found, proceeding...");


            FileInputStream FEInputStream = new FileInputStream(featureExtractorFile);
            FileChannel FEChannel = FEInputStream.getChannel();
            MappedByteBuffer featureExtractorModel = FEChannel.map(FileChannel.MapMode.READ_ONLY, 0, FEChannel.size());

            FileInputStream ClInputStream = new FileInputStream(classifierFile);
            FileChannel ClChannel = ClInputStream.getChannel();
            MappedByteBuffer classifierModel = ClChannel.map(FileChannel.MapMode.READ_ONLY, 0, ClChannel.size());

            Interpreter.Options options = new Interpreter.Options();
            options.setNumThreads(4);

            Interpreter FEInterpreter = new Interpreter(featureExtractorModel, options);
            Interpreter ClInterpreter = new Interpreter(classifierModel, options);


            System.out.println("Models loaded successfully.");

            // Load checkpoint if exists
            File checkpointFile = new File(getFilesDir(), "checkpoint.ckpt");
            if (checkpointFile.exists()) {
                restoreCheckpoint(ClInterpreter);
                System.out.println("Checkpoint restored.");
            }else {
                System.out.println("No checkpoint found. Starting with initial weights.");
            }

            ByteBuffer byteBuffer = preprocessImage(image);
            System.out.println("Input ByteBuffer size: " + byteBuffer.capacity());

            performInference(FEInterpreter, ClInterpreter, byteBuffer);

            FEInterpreter.close();
            ClInterpreter.close();
            FEInputStream.close();
            ClInputStream.close();
            System.out.println("Models closed.");

        } catch (IOException e) {
            e.printStackTrace();
            System.out.println("Error: " + e.getMessage());
        } catch (Exception e) {
            e.printStackTrace();
            System.out.println("Unexpected error: " + e.getMessage());
        }
    }

    public void classifyAndTrainOnAllClasses(DocumentFile directory) {
        try {

            String Cl_modelFileName = "classifier_model.tflite";
            String FE_modelFileName = "feature_extractor_model.tflite";
            copyModelFromAssetsIfNecessary(Cl_modelFileName);
            copyModelFromAssetsIfNecessary(FE_modelFileName);

            File featureExtractorFile = new File(getFilesDir(), FE_modelFileName);
            File classifierFile = new File(getFilesDir(), Cl_modelFileName);

            if (!featureExtractorFile.exists()) {
                System.out.println("TFLite model file '" + FE_modelFileName + "' not found.");
                return;
            }
            if (!classifierFile.exists()) {
                System.out.println("TFLite model file '" + Cl_modelFileName + "' not found.");
                return;
            }
            System.out.println("TFLite models found, proceeding...");

            FileInputStream FEInputStream = new FileInputStream(featureExtractorFile);
            FileChannel FEChannel = FEInputStream.getChannel();
            MappedByteBuffer featureExtractorModel = FEChannel.map(FileChannel.MapMode.READ_ONLY, 0, FEChannel.size());

            FileInputStream ClInputStream = new FileInputStream(classifierFile);
            FileChannel ClChannel = ClInputStream.getChannel();
            MappedByteBuffer classifierModel = ClChannel.map(FileChannel.MapMode.READ_ONLY, 0, ClChannel.size());

            Interpreter.Options options = new Interpreter.Options();
            options.setNumThreads(4);

            Interpreter FEInterpreter = new Interpreter(featureExtractorModel, options);
            Interpreter ClInterpreter = new Interpreter(classifierModel, options);

            System.out.println("Models loaded successfully.");

            File checkpointFile = new File(getFilesDir(), "checkpoint.ckpt");
            if (checkpointFile.exists()) {
                System.out.println("Found checkpoint file, restoring...");
                restoreCheckpoint(ClInterpreter);
            } else {
                System.out.println("No checkpoint file found, starting fresh training.");
            }

            List<TrainingSample> allSamples = new ArrayList<>();
            DocumentFile[] classDirs = directory.listFiles();

            System.out.println("Scanning directory structure...");
            System.out.println("Found " + classDirs.length + " class directories");

            int totalImages = 0;
            for (DocumentFile classDir : classDirs) {
                if (!classDir.isDirectory()) continue;

                int correctLabel = Integer.parseInt(classDir.getName());
                DocumentFile[] imageFiles = classDir.listFiles();
                int classImages = 0;

                for (DocumentFile imageFile : classDir.listFiles()) {
                    if (!imageFile.isFile()) continue;
                    allSamples.add(new TrainingSample(imageFile, correctLabel));
                    classImages++;
                    totalImages++;
                }
                System.out.println("Class " + correctLabel + ": found " + classImages + " images");
            }

            System.out.println("Total training samples collected: " + totalImages);
            int numEpochs = 2;
            Random random = new Random();

            for (int epoch = 0; epoch < numEpochs; ++epoch) {
                // Shuffle all samples before each epoch
                Collections.shuffle(allSamples, random);
                int successfulSamples = 0;
                int failedSamples = 0;
                long epochStartTime = System.currentTimeMillis();

                for (int i = 0; i < allSamples.size(); i++) {
                    TrainingSample sample = allSamples.get(i);

                    ByteBuffer labelBuffer = ByteBuffer.allocateDirect(4 * 10).order(ByteOrder.nativeOrder());
                    // Fill with zeros first
                    for (int j = 0; j < 10; j++) {
                        labelBuffer.putFloat(0.0f);
                    }
                    // Set 1.0f for the correct class
                    labelBuffer.rewind();
                    labelBuffer.position(4 * sample.label); // Move to the correct position (4 bytes per float)
                    labelBuffer.putFloat(1.0f);
                    labelBuffer.rewind();

                    try (InputStream inputStream = getContentResolver().openInputStream(sample.imageFile.getUri())) {
                        Bitmap image = BitmapFactory.decodeStream(inputStream);
                        ByteBuffer byteBuffer = preprocessImage(image);

                        TensorBuffer extractedFeatures = TensorBuffer.createFixedSize(new int[]{1, 1024}, DataType.FLOAT32);
                        FEInterpreter.run(byteBuffer, extractedFeatures.getBuffer());

                        Map<String, Object> inputs = new HashMap<>();
                        inputs.put("x", extractedFeatures.getBuffer());
                        inputs.put("y", labelBuffer);

                        Map<String, Object> outputs = new HashMap<>();
                        outputs.put("loss", FloatBuffer.allocate(1));
                        ClInterpreter.runSignature(inputs, outputs, "train");

                        if (i % 10 == 0 || i == allSamples.size() - 1) {
                            float progress = (float) i / allSamples.size() * 100;
                            long currentTime = System.currentTimeMillis();
                            long elapsedSeconds = (currentTime - epochStartTime) / 1000;
                            System.out.println(String.format(
                                    "Epoch %d/%d: %.1f%% complete (%d/%d samples) | Time: %ds | Successfully processed: %d | Failed: %d",
                                    epoch + 1, numEpochs, progress, i + 1, allSamples.size(), elapsedSeconds,
                                    successfulSamples, failedSamples
                            ));
                        }
                    } catch (IOException e) {
                        System.out.println("Error processing sample from class " + sample.label + ": " + e.getMessage());
                        e.printStackTrace();
                        failedSamples++;
                    }
                }
                long epochEndTime = System.currentTimeMillis();
                long epochDuration = (epochEndTime - epochStartTime) / 1000;

                System.out.println(String.format(
                        "\nEpoch %d/%d completed in %d seconds",
                        epoch + 1, numEpochs, epochDuration
                ));
                System.out.println(String.format(
                        "Successfully processed: %d samples | Failed: %d samples",
                        successfulSamples, failedSamples
                ));
            }
            System.out.println("\nTraining completed. Saving checkpoint...");
            saveCheckpoint(ClInterpreter);
            FEInterpreter.close();
            ClInterpreter.close();

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // Helper class to keep image and label together
    private static class TrainingSample {
        final DocumentFile imageFile;
        final int label;

        TrainingSample(DocumentFile imageFile, int label) {
            this.imageFile = imageFile;
            this.label = label;
        }
    }

    // Main function to classify an image and update the model via training
    public void classifyAndTrain(List<Bitmap> images, String correctClass) {
        try {
            String Cl_modelFileName = "classifier_model.tflite";
            String FE_modelFileName = "feature_extractor_model.tflite";
            copyModelFromAssetsIfNecessary(Cl_modelFileName);
            copyModelFromAssetsIfNecessary(FE_modelFileName);

            File featureExtractorFile = new File(getFilesDir(), FE_modelFileName);
            File classifierFile = new File(getFilesDir(), Cl_modelFileName);

            if (!featureExtractorFile.exists()) {
                System.out.println("TFLite model file '" + FE_modelFileName + "' not found.");
                return;
            }
            if (!classifierFile.exists()) {
                System.out.println("TFLite model file '" + Cl_modelFileName + "' not found.");
                return;
            }
            System.out.println("TFLite models found, proceeding...");


            FileInputStream FEInputStream = new FileInputStream(featureExtractorFile);
            FileChannel FEChannel = FEInputStream.getChannel();
            MappedByteBuffer featureExtractorModel = FEChannel.map(FileChannel.MapMode.READ_ONLY, 0, FEChannel.size());

            FileInputStream ClInputStream = new FileInputStream(classifierFile);
            FileChannel ClChannel = ClInputStream.getChannel();
            MappedByteBuffer classifierModel = ClChannel.map(FileChannel.MapMode.READ_ONLY, 0, ClChannel.size());

            Interpreter.Options options = new Interpreter.Options();
            options.setNumThreads(4);

            Interpreter FEInterpreter = new Interpreter(featureExtractorModel, options);
            Interpreter ClInterpreter = new Interpreter(classifierModel, options);

            System.out.println("Models loaded successfully.");

            File checkpointFile = new File(getFilesDir(), "checkpoint.ckpt");
            if (checkpointFile.exists()) {
                restoreCheckpoint(ClInterpreter);
                System.out.println("Checkpoint restored.");
            }else {
                System.out.println("No checkpoint found. Starting with initial weights.");
            }

            int correctLabel = getClassIndex(correctClass);
            System.out.println("label index "+correctLabel);

            ByteBuffer labelBuffer = ByteBuffer.allocate(4*10).order(ByteOrder.nativeOrder());
            labelBuffer.putInt(correctLabel);
            labelBuffer.rewind();
            System.out.println("label buffer "+labelBuffer);

            // Prepare output loss
            FloatBuffer lossBuffer = FloatBuffer.allocate(1);

            int numEpochs =8;

            float[] losses = new float[numEpochs];
            // Training loop
            for (int epoch = 0; epoch < numEpochs; ++epoch) {
                for (Bitmap image : images) {
                    ByteBuffer byteBuffer = preprocessImage(image);

                    TensorBuffer extractedFeatures = TensorBuffer.createFixedSize(new int[]{1, 1024}, DataType.FLOAT32);
                    FEInterpreter.run(byteBuffer, extractedFeatures.getBuffer());

                    // Prepare inputs for the train signature
                    Map<String, Object> inputs = new HashMap<>();
                    inputs.put("x", extractedFeatures.getBuffer());
                    inputs.put("y", labelBuffer); // True label

                    Map<String, Object> outputs = new HashMap<>();
                    outputs.put("loss", lossBuffer);
                    ClInterpreter.runSignature(inputs, outputs, "train");
                    losses[epoch] = lossBuffer.get(0);
                    lossBuffer.rewind(); // Reset buffer after each use
                    System.out.println("Epoch: " + epoch + ", Loss: " + losses[epoch]);
                    if (epoch ==numEpochs-1){
                        performInference(FEInterpreter,ClInterpreter,byteBuffer);
                    }
                }
            }

            saveCheckpoint(ClInterpreter);
            System.out.println("Training completed. Checkpoint saved.");

//            System.out.println("Inference after training");
//            performInference(FEInterpreter,ClInterpreter,byteBuffer);

            FEInterpreter.close();
            ClInterpreter.close();

        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    private void performInference(Interpreter featureExtractor, Interpreter classifier, ByteBuffer input) {
        try {
            System.out.println("Input shape: " + Arrays.toString(featureExtractor.getInputTensor(0).shape()));

            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 28, 28, 1}, DataType.FLOAT32);
            inputFeature0.loadBuffer(input);

            System.out.println("Input tensor size: " + inputFeature0.getFlatSize());
            System.out.println("Input buffer capacity: " + input.capacity());

            TensorBuffer extractedFeatures = TensorBuffer.createFixedSize(new int[]{1, 1024}, DataType.FLOAT32);
            featureExtractor.run(inputFeature0.getBuffer(), extractedFeatures.getBuffer());

            TensorBuffer outputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 10}, DataType.FLOAT32);
            TensorBuffer outputFeature1 = TensorBuffer.createFixedSize(new int[]{1, 10}, DataType.FLOAT32);

            Map<Integer, Object> outputs = new HashMap<>();
            outputs.put(0, outputFeature0.getBuffer());
            outputs.put(1, outputFeature1.getBuffer());
            classifier.runForMultipleInputsOutputs(new Object[]{extractedFeatures.getBuffer()}, outputs);

            float[] results = outputFeature1.getFloatArray();


            String[] classes = {"0", "1", "2", "3","4","5","6","7","8","9"};
            int maxPos = 0;
            float maxConfidence = results[0];
            for (int i = 0; i < results.length; i++) {
                if (results[i] > maxConfidence) {
                    maxConfidence = results[i];
                    maxPos = i;
                }
            }

            result.setText(classes[maxPos]);

            StringBuilder probabilitiesString = new StringBuilder();
            for (int i = 0; i < results.length; i++) {
                probabilitiesString.append(classes[i]).append(": ")
                        .append(String.format("%.2f", results[i] * 100))
                        .append("%\n");
            }

            result.append("\nProbabilities:\n" + probabilitiesString);
            System.out.println("Probabilities breakdown:\n" + probabilitiesString);

        } catch (Exception e) {
            e.printStackTrace();
            System.out.println("Error during inference: " + e.getMessage());
        }
    }
    // Helper function to save model weights to checkpoint
    private void saveCheckpoint(Interpreter interpreter) {
        if (interpreter == null) {
            System.out.println("Interpreter is null. Cannot save checkpoint.");
            return;
        }
        try {
            File outputFile = new File(getFilesDir(), "checkpoint.ckpt");
            Map<String, Object> inputs = new HashMap<>();
            inputs.put("checkpoint_path", outputFile.getAbsolutePath());
            Map<String, Object> outputs = new HashMap<>();
            interpreter.runSignature(inputs, outputs, "save");
            System.out.println("Model weights saved to: " + outputFile.getAbsolutePath());
        } catch (Exception e) {
            System.err.println("Error while saving checkpoint: " + e.getMessage());
            e.printStackTrace();
        }
    }

    // Helper function to restore model weights from checkpoint
    private void restoreCheckpoint(Interpreter interpreter) {
        if (interpreter == null) {
            System.out.println("Interpreter is null. Cannot restore checkpoint.");
            return;
        }

        try {
            File outputFile = new File(getFilesDir(), "checkpoint.ckpt");
            if (!outputFile.exists()) {
                System.out.println("Checkpoint file does not exist: " + outputFile.getAbsolutePath());
                return;
            }

            Map<String, Object> inputs = new HashMap<>();
            inputs.put("checkpoint_path", outputFile.getAbsolutePath());
            Map<String, Object> outputs = new HashMap<>();
            interpreter.runSignature(inputs, outputs, "restore");
            System.out.println("Model weights restored from: " + outputFile.getAbsolutePath());
        } catch (Exception e) {
            System.err.println("Error while restoring checkpoint: " + e.getMessage());
            e.printStackTrace();
        }
    }
    // Helper function to preprocess image
    private ByteBuffer preprocessImage(Bitmap image) {
        int modelWidth = 28;
        int modelHeight = 28;
        int channels = 1;  // Grayscale image
        int pixelSize = 4;  // Float32 takes 4 bytes

        // Allocate buffer with correct size
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(modelWidth * modelHeight * channels * pixelSize);
        byteBuffer.order(ByteOrder.nativeOrder());

        // Convert to grayscale and get pixel values
        int[] pixels = new int[modelWidth * modelHeight];
        image.getPixels(pixels, 0, modelWidth, 0, 0, modelWidth, modelHeight);

        // Convert the image to float and normalize
        for (int pixel : pixels) {
            // Convert to grayscale using standard conversion formula
            float grayValue = (((pixel >> 16) & 0xFF) * 0.299f +
                    ((pixel >> 8) & 0xFF) * 0.587f +
                    (pixel & 0xFF) * 0.114f) / 255.0f;
            byteBuffer.putFloat(grayValue);
        }

        byteBuffer.rewind();

        return byteBuffer;
    }

    // Helper function to map class names to their indices
    private int getClassIndex(String className) {
        String[] classes = {"0", "1", "2", "3","4","5","6","7","8","9"};
        for (int i = 0; i < classes.length; i++) {
            if (classes[i].equals(className)) {
                return i;
            }
        }
        return -1; // Error: class not found
    }

    @Override
     protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data){
        if(resultCode==RESULT_OK){
            if (requestCode==3){
                Bitmap image = (Bitmap) data.getExtras().get("data");
                int dimension = Math.min(image.getWidth(),image.getHeight());
                image = ThumbnailUtils.extractThumbnail(image,dimension,dimension);
                imageView.setImageBitmap(image);

                image = Bitmap.createScaledBitmap(image, imageSize,imageSize,false);
                System.out.println("Scaled Image Size: Width = " + image.getWidth() + ", Height = " + image.getHeight());
                classifyImageWithInference(image);
            }
            else if (requestCode==2){
                Uri dat = data.getData();
                Bitmap image = null;
                try{
                    image = MediaStore.Images.Media.getBitmap(this.getContentResolver(),dat);

                } catch (IOException e) {
                    e.printStackTrace();
                }
                imageView.setImageBitmap(image);
                image = Bitmap.createScaledBitmap(image, imageSize,imageSize,false);
                System.out.println("Scaled Image Size: Width = " + image.getWidth() + ", Height = " + image.getHeight());
                classifyImageWithInference(image);

            }
            else if (requestCode==6){
                Uri folderUri = data.getData();
                DocumentFile directory = DocumentFile.fromTreeUri(this, folderUri);
                if (directory != null && directory.isDirectory()) {
                    classifyAndTrainOnAllClasses(directory);
                }
            }
        }
        super.onActivityResult(requestCode,resultCode,data);
     }
}