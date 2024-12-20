package com.example.test;

import android.annotation.SuppressLint;
import android.app.AlertDialog;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.media.ThumbnailUtils;
import android.net.Uri;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.provider.MediaStore;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.Manifest;



import androidx.activity.EdgeToEdge;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.documentfile.provider.DocumentFile;
import androidx.localbroadcastmanager.content.LocalBroadcastManager;


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
import java.sql.Struct;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.io.File;
import java.util.Objects;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import okhttp3.Call;
import okhttp3.Callback;
import okhttp3.MediaType;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

public class MainActivity extends AppCompatActivity {
    private static MainActivity instance;
    public static MainActivity getInstance() {
        return instance;
    }
    Button camera, train_gallery,gallery,send,load;
    ImageView imageView;
    TextView result;
    int imageSize = 28;
    private boolean isTrainingCanceled = false;
    private static final String PREFS_NAME = "app_prefs";
    private static final String SAVED_URI_KEY = "saved_directory_uri";
    String ExceptionOccurred = getString(R.string.error_generic);
    String error_title = getString(R.string.error_title);
    String success_title = getString(R.string.success_title);

    String interpreter_error = String.format(
            Locale.getDefault(),
            getString(R.string.interpreter_null)
    );
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        instance = this;
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);
        camera = findViewById(R.id.take_pic_button);
        train_gallery = findViewById(R.id.train_gal_button);
        gallery=findViewById(R.id.lauch_gal_button);
        send =findViewById(R.id.send_button);
        load =findViewById(R.id.load_button);

        result=findViewById(R.id.result);

        imageView=findViewById(R.id.imageView);

        camera.setOnClickListener(view -> {
            if(checkSelfPermission(Manifest.permission.CAMERA)== PackageManager.PERMISSION_GRANTED){
                Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                startActivityForResult(cameraIntent,3);
            }
            else{
                requestPermissions(new String[]{Manifest.permission.CAMERA},100);
            }
        });
        gallery.setOnClickListener(view -> {
            Intent galleryIntent = new Intent(Intent.ACTION_PICK,MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
            startActivityForResult(galleryIntent,2);
        });
        train_gallery.setOnClickListener(v -> {

            Intent folderPickerIntent = new Intent(Intent.ACTION_OPEN_DOCUMENT_TREE);
            startActivityForResult(folderPickerIntent, 6);
        });
        send.setOnClickListener(v -> {
            System.out.println("Starting sending weights");
            sendWeightsToServer();
        });
        load.setOnClickListener(v -> loadModel());
    }
    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (instance == this) {
            instance = null;
        }
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
            showResultDialog(error_title,ExceptionOccurred + e.getMessage());
            e.printStackTrace();
        }
        return serverIp;
    }
    @SuppressLint("StaticFieldLeak")
    private void loadModel() {
        new AsyncTask<Void, Void, Boolean>() {
            private String message;

            @Override
            protected void onPreExecute() {
                showLoadingDialog(getString(R.string.loading_model));
            }

            @Override
            protected Boolean doInBackground(Void... voids) {
                try {
                    OkHttpClient client = new OkHttpClient();
                    String serverIp = serverIp();
                    String url = String.format("http://%s:5000/load_model", serverIp);
                    Request request = new Request.Builder()
                            .url(url)
                            .build();

                    Response response = client.newCall(request).execute();
                    if (!response.isSuccessful()) {
                        message = getString(R.string.failed_to_download_model, response.message());
                        return false;
                    }

                    deleteFileIfExists(new File(getFilesDir(), "checkpoint.ckpt"));
                    deleteFileIfExists(new File(getFilesDir(), "classifier_model.tflite"));

                    File modelFile = new File(getFilesDir(), "classifier_model.tflite");
                    FileOutputStream fos = new FileOutputStream(modelFile);
                    assert response.body() != null;
                    fos.write(response.body().bytes());
                    fos.close();

                    if (modelFile.length() == 0) {
                        message = getString(R.string.downloaded_model_empty);
                        return false;
                    }

                    try (Interpreter tflite = new Interpreter(modelFile)) {
                        message = getString(R.string.model_downloaded_successfully);
                    } catch (Exception e) {
                        message = getString(R.string.error_loading_model, e.getMessage());
                        return false;
                    }

                    return true;
                } catch (IOException e) {
                    message = ExceptionOccurred + e.getMessage();
                    return false;
                }
            }

            @Override
            protected void onPostExecute(Boolean isSuccess) {
                hideLoadingDialog();

                showResultDialog(
                        isSuccess ? success_title : error_title,
                        message
                );
            }
        }.execute();
    }

    private void deleteFileIfExists(File file) {
        if (file.exists()) {
            boolean deleted = file.delete();
            System.out.println(deleted ? file.getName() + " deleted successfully" : "Failed to delete " + file.getName());
        }
    }

    private AlertDialog loadingDialog;

    private void showLoadingDialog(String message) {
        if (loadingDialog == null) {
            loadingDialog = new AlertDialog.Builder(this)
                    .setCancelable(false)
                    .setMessage(message)
                    .create();
        }
        loadingDialog.show();
    }

    private void hideLoadingDialog() {
        if (loadingDialog != null && loadingDialog.isShowing()) {
            loadingDialog.dismiss();
        }
    }

    private void showResultDialog(String title, String message) {
        new AlertDialog.Builder(this)
                .setTitle(title)
                .setMessage(message)
                .setPositiveButton("OK", null)
                .show();
    }
    private void sendWeightsToServer() {
        try {
            File checkpointFile = new File(getFilesDir(), "checkpoint.ckpt");
            if (!checkpointFile.exists()) {
                showResultDialog("Error", "Checkpoint file not found: " + checkpointFile.getAbsolutePath());
                return;
            }
            System.out.println("Checkpoint file found: " + checkpointFile.getAbsolutePath());

            byte[] modelWeights = new byte[(int) checkpointFile.length()];
            FileInputStream fis = new FileInputStream(checkpointFile);
            fis.read(modelWeights);
            fis.close();

            System.out.println("Weights successfully read from file, size: " + modelWeights.length + " bytes.");

            OkHttpClient client = new OkHttpClient();
            RequestBody requestBody = RequestBody.create(MediaType.parse("application/octet-stream"), modelWeights);
            String serverIp = serverIp();
            String url = String.format("http://%s:5000/upload-weights", serverIp);
            Request request = new Request.Builder()
                    .url(url)
                    .post(requestBody)
                    .build();

            String errorSendingWeights = getString(R.string.error_sending_weights);
            client.newCall(request).enqueue(new Callback() {
                @Override
                public void onFailure(@NonNull Call call, @NonNull IOException e) {
                    System.err.println("Error sending weights: " + e.getMessage());
                    e.printStackTrace();

                    runOnUiThread(() -> showResultDialog(error_title, errorSendingWeights + e.getMessage()));
                }

                @Override
                public void onResponse(@NonNull Call call, @NonNull Response response) {
                    if (response.isSuccessful()) {
                        System.out.println("Weights successfully sent to the server.");
                        String weightsSentSuccessfully = getString(R.string.weights_sent_successfully);
                        runOnUiThread(() -> showResultDialog(success_title, weightsSentSuccessfully));
                    } else {
                        System.err.println("Failed to send weights: " + response.message());runOnUiThread(() -> showResultDialog(error_title, errorSendingWeights + response.message()));
                    }
                }
            });

        } catch (IOException e) {
            System.err.println(ExceptionOccurred + e.getMessage());
            e.printStackTrace();
            showResultDialog(error_title, ExceptionOccurred + e.getMessage());
        }
    }
    private void copyModelFromAssetsIfNecessary(String modelFileName) {
        File modelFile = new File(getFilesDir(), modelFileName);
        if (!modelFile.exists()) {
            try {
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
                showResultDialog(error_title,ExceptionOccurred + e.getMessage());
                System.out.println("Failed to copy model from assets: " + e.getMessage());
            }
        } else {
            System.out.println("Model file already exists at " + modelFile.getAbsolutePath());
        }
    }

    private boolean checkModelFilesExistence(File featureExtractorFile, File classifierFile, String FE_modelFileName, String Cl_modelFileName, String error_title) {
        if (!featureExtractorFile.exists() || !classifierFile.exists()) {
            String file_error = "";

            if (!classifierFile.exists()) {
                System.out.println("TFLite model file '" + FE_modelFileName + "' not found.");
                file_error = String.format(getString(R.string.model_file_error), FE_modelFileName);
            } else {
                System.out.println("TFLite model file '" + Cl_modelFileName + "' not found.");
                file_error = String.format(getString(R.string.model_file_error), Cl_modelFileName);
            }

            showResultDialog(error_title, file_error);
            return false;
        }
        return true;
    }
    public void classifyImageWithInference(Bitmap image) {
        try {
            String Cl_modelFileName = "classifier_model.tflite";
            String FE_modelFileName = "feature_extractor_model.tflite";
            copyModelFromAssetsIfNecessary(Cl_modelFileName);
            copyModelFromAssetsIfNecessary(FE_modelFileName);

            File featureExtractorFile = new File(getFilesDir(), FE_modelFileName);
            File classifierFile = new File(getFilesDir(), Cl_modelFileName);

            if (!checkModelFilesExistence(featureExtractorFile, classifierFile, FE_modelFileName, Cl_modelFileName, error_title)) {
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
            showResultDialog(error_title,ExceptionOccurred + e.getMessage());
            System.out.println("Error: " + e.getMessage());
        } catch (Exception e) {
            e.printStackTrace();
            showResultDialog(error_title,ExceptionOccurred + e.getMessage());
            System.out.println("Unexpected error: " + e.getMessage());
        }
    }

    public void classifyAndTrainOnAllClasses2(DocumentFile directory) {
        ExecutorService executor = Executors.newSingleThreadExecutor();
        Handler handler = new Handler(Looper.getMainLooper());
        executor.execute(() -> {
            try {
                isTrainingCanceled = false;
                startTrainingScreen();
                String Cl_modelFileName = "classifier_model.tflite";
                String FE_modelFileName = "feature_extractor_model.tflite";
                copyModelFromAssetsIfNecessary(Cl_modelFileName);
                copyModelFromAssetsIfNecessary(FE_modelFileName);

                File featureExtractorFile = new File(getFilesDir(), FE_modelFileName);
                File classifierFile = new File(getFilesDir(), Cl_modelFileName);

                if (!checkModelFilesExistence(featureExtractorFile, classifierFile, FE_modelFileName, Cl_modelFileName, error_title)) {
                    return;
                }
                System.out.println("TFLite models found, proceeding...");

                Interpreter.Options options = new Interpreter.Options();
                options.setNumThreads(4);

                Interpreter FEInterpreter = new Interpreter(loadModelBuffer(featureExtractorFile), options);
                Interpreter ClInterpreter = new Interpreter(loadModelBuffer(classifierFile), options);

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

                Map<Integer, Integer> classImageCounts = new HashMap<>();
                for (DocumentFile classDir : classDirs) {
                    if (isTrainingCanceled) break;
                    if (!classDir.isDirectory()) continue;

                    int correctLabel = Integer.parseInt(classDir.getName());
                    DocumentFile[] imageFiles = classDir.listFiles();

                    if (imageFiles.length == 0) { //TODO: Reuse ?
                        cancelTraining();
                        String folder_length_error_message = String.format(
                                Locale.getDefault(),
                                getString(R.string.folder_length_error_message),
                                correctLabel
                        );
                        handler.post(() -> {
                            if (TrainingActivity.getInstance() != null) {
                                TrainingActivity.getInstance().checkAndShowErrorDialog(folder_length_error_message);
                            } else {
                                System.out.println("TrainingActivity instance not found.");
                            }
                        });
                        System.out.println(folder_length_error_message);
                        return;
                    }

                    int classImages = 0;
                    for (DocumentFile imageFile : imageFiles) {
                        if (isTrainingCanceled) break;
                        if (!imageFile.isFile()) continue;
                        allSamples.add(new TrainingSample(imageFile, correctLabel));
                        classImages++;
                    }
                    classImageCounts.put(correctLabel, classImages);
                    String message = String.format(
                            Locale.getDefault(),
                            getString(R.string.class_preparing),
                            correctLabel,
                            classImages
                    );
                    System.out.println(message);
                    updateTrainingProgress(message);
                }
                for (Map.Entry<Integer, Integer> entry : classImageCounts.entrySet()) {
                    if (entry.getValue() < 100) { //TODO: Reuse?
                        cancelTraining();
                        String class_length_error_message = String.format(
                                Locale.getDefault(),
                                getString(R.string.class_length_error_message),
                                entry.getKey()
                        );
                        handler.post(() -> {
                            if (TrainingActivity.getInstance() != null) {
                                TrainingActivity.getInstance().checkAndShowErrorDialog(class_length_error_message);
                            } else {
                                System.out.println("TrainingActivity instance not found.");
                            }
                        });
                        System.out.println(class_length_error_message);
                        return;
                    }
                }
                int minImages = Collections.min(classImageCounts.values());
                int maxImages = Collections.max(classImageCounts.values());
                if ((double) maxImages / minImages > 4.5) { //TODO: Reuse?
                    int difference = maxImages/ minImages;
                    cancelTraining();
                    String difference_error_message = String.format(
                            Locale.getDefault(),
                            getString(R.string.difference_error_message),
                            difference
                    );
                    handler.post(() -> {
                        if (TrainingActivity.getInstance() != null) {
                            TrainingActivity.getInstance().checkAndShowErrorDialog(difference_error_message);
                        } else {
                            System.out.println("TrainingActivity instance not found.");
                        }
                    });
                    System.out.println(difference_error_message);
                    return;
                }

                int totalSamples = allSamples.size();
                System.out.println("Total samples collected: " + totalSamples);

                int numEpochs = 2;
                Random random = new Random();

                for (int epoch = 0; epoch < numEpochs; ++epoch) {
                    if (isTrainingCanceled) break;
                    // Shuffle all samples before each epoch
                    Collections.shuffle(allSamples, random);
                    long epochStartTime = System.currentTimeMillis();

                    for (int i = 0; i < allSamples.size(); i++) {
                        if (isTrainingCanceled) break;
                        TrainingSample sample = allSamples.get(i);

                        ByteBuffer labelBuffer = ByteBuffer.allocateDirect(4 * 10).order(ByteOrder.nativeOrder());
                        for (int j = 0; j < 10; j++) {
                            if (isTrainingCanceled) break;
                            labelBuffer.putFloat(0.0f);
                        }
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

                                String message = String.format(
                                        Locale.getDefault(),
                                        getString(R.string.training_message),
                                        epoch + 1,
                                        numEpochs,
                                        progress,
                                        i + 1,
                                        allSamples.size(),
                                        elapsedSeconds
                                );
                                System.out.println(message);
                                updateTrainingProgress(message);
                            }
                        } catch (IOException e) {
                            showResultDialog(error_title,ExceptionOccurred + e.getMessage());
                            System.out.println("Error processing sample from class " + sample.label + ": " + e.getMessage());
                            e.printStackTrace();
                        }
                    }

                }

                if (!isTrainingCanceled) { //TODO: Check Alert
                    System.out.println("\nTraining completed. Saving checkpoint...");
                    saveCheckpoint(ClInterpreter);
                }
                else{
                    System.out.println("Training process is stopped");
                }
                FEInterpreter.close();
                ClInterpreter.close();
                handler.post(this::finishTraining);
            } catch (IOException e) {
                showResultDialog(error_title,ExceptionOccurred + e.getMessage());
                e.printStackTrace();
            }
            finally {
                executor.shutdown();
            }
        });
    }

    public void classifyAndTrainOnAllClasses(DocumentFile directory) {
        ExecutorService executor = Executors.newSingleThreadExecutor();
        Handler handler = new Handler(Looper.getMainLooper());

        executor.execute(() -> {
            try {
                isTrainingCanceled = false;
                startTrainingScreen();
                String Cl_modelFileName = "classifier_model.tflite";
                String FE_modelFileName = "feature_extractor_model.tflite";
                copyModelFromAssetsIfNecessary(Cl_modelFileName);
                copyModelFromAssetsIfNecessary(FE_modelFileName);

                File featureExtractorFile = new File(getFilesDir(), FE_modelFileName);
                File classifierFile = new File(getFilesDir(), Cl_modelFileName);

                if (!checkModelFilesExistence(featureExtractorFile, classifierFile, FE_modelFileName, Cl_modelFileName, error_title)) {
                    return;
                }

                // Collect all samples first
                List<TrainingSample> allSamples = new ArrayList<>();
                DocumentFile[] classDirs = directory.listFiles();
                System.out.println("Scanning directory structure...");
                System.out.println("Found " + classDirs.length + " class directories");

                Map<Integer, Integer> classImageCounts = new HashMap<>();
                for (DocumentFile classDir : classDirs) {
                    if (isTrainingCanceled) break;
                    if (!classDir.isDirectory()) continue;

                    int correctLabel = Integer.parseInt(classDir.getName());
                    DocumentFile[] imageFiles = classDir.listFiles();

                    if (imageFiles.length == 0) {
                        cancelTraining();
                        String folder_length_error_message = String.format(
                                Locale.getDefault(),
                                getString(R.string.folder_length_error_message),
                                correctLabel
                        );
                        handler.post(() -> {
                            if (TrainingActivity.getInstance() != null) {
                                TrainingActivity.getInstance().checkAndShowErrorDialog(folder_length_error_message);
                            } else {
                                System.out.println("TrainingActivity instance not found.");
                            }
                        });
                        System.out.println(folder_length_error_message);
                        return;
                    }

                    int classImages = 0;
                    for (DocumentFile imageFile : imageFiles) {
                        if (isTrainingCanceled) break;
                        if (!imageFile.isFile()) continue;
                        allSamples.add(new TrainingSample(imageFile, correctLabel));
                        classImages++;
                    }
                    classImageCounts.put(correctLabel, classImages);
                    String message = String.format(
                            Locale.getDefault(),
                            getString(R.string.class_preparing),
                            correctLabel,
                            classImages
                    );
                    System.out.println(message);
                    updateTrainingProgress(message);
                }

                for (Map.Entry<Integer, Integer> entry : classImageCounts.entrySet()) {
                    if (entry.getValue() < 100) {
                        String class_length_error_message = String.format(
                                Locale.getDefault(),
                                getString(R.string.class_length_error_message),
                                entry.getKey()
                        );
                        System.out.println(class_length_error_message);
                        cancelTraining();
                        return;
                    }
                }
                int minImages = Collections.min(classImageCounts.values());
                int maxImages = Collections.max(classImageCounts.values());
                if ((double) maxImages / minImages > 4.5) {
                    int difference = maxImages/ minImages;
                    cancelTraining();
                    String difference_error_message = String.format(
                            Locale.getDefault(),
                            getString(R.string.difference_error_message),
                            difference
                    );
                    handler.post(() -> {
                        if (TrainingActivity.getInstance() != null) {
                            TrainingActivity.getInstance().checkAndShowErrorDialog(difference_error_message);
                        } else {
                            System.out.println("TrainingActivity instance not found.");
                        }
                    });
                    System.out.println(difference_error_message);
                    return;
                }

                Collections.shuffle(allSamples, new Random());

                List<TrainingPhase> trainingPhases = new ArrayList<>();
                int totalSamples = allSamples.size();
                System.out.println("Total samples collected: " + totalSamples);

                int phase1Size = 4000;
                int phase2Size = 3000;
                int phase3Size = 3000;

                trainingPhases.add(new TrainingPhase(allSamples, 0, phase1Size, 1));
                trainingPhases.add(new TrainingPhase(allSamples, phase1Size, phase1Size + phase2Size, 2));
                trainingPhases.add(new TrainingPhase(allSamples, phase1Size + phase2Size,
                        Math.min(phase1Size + phase2Size + phase3Size, totalSamples), 3));

                // Train each phase separately
                for (TrainingPhase phase : trainingPhases) {
                    if (isTrainingCanceled) break;
                    handler.post(() -> {
                        System.out.println("Starting Phase " + phase.getPhaseNumber());
                    });
                    System.out.println("Training on " + phase.getSamples().size() + " samples");

                    // Initialize new interpreters for each phase
                    Interpreter.Options options = new Interpreter.Options();
                    options.setNumThreads(4);

                    Interpreter FEInterpreter = new Interpreter(loadModelBuffer(featureExtractorFile), options);
                    Interpreter ClInterpreter = new Interpreter(loadModelBuffer(classifierFile), options);

                    // Train on the current phase
                    trainPhase(phase, FEInterpreter, ClInterpreter);

                    handler.post(() -> System.out.println("Completed Phase " + phase.getPhaseNumber()));
                    if (!isTrainingCanceled){
                        saveCheckpoint(ClInterpreter);
                        sendWeightsToServer();
                    }

                    File checkpointFile = new File(getFilesDir(), "checkpoint.ckpt");
                    if (checkpointFile.exists()) {
                        boolean deleted = checkpointFile.delete();
                        if (deleted) {
                            System.out.println("Checkpoint file deleted successfully");
                        } else {
                            System.out.println("Failed to delete checkpoint file");
                        }
                    }
                    FEInterpreter.close();
                    ClInterpreter.close();
                }
                if (!isTrainingCanceled) {
                    System.out.println("All training phases completed");
                }
                else{
                    System.out.println("Training process is stopped");
                }
                handler.post(this::finishTraining);
            } catch (IOException e) {
                showResultDialog(error_title,ExceptionOccurred + e.getMessage());
                e.printStackTrace();
            }finally {
                executor.shutdown();
            }
        });
    }
    private void trainPhase(TrainingPhase phase, Interpreter FEInterpreter, Interpreter ClInterpreter) {
        List<TrainingSample> phaseSamples = phase.getSamples();
        int numEpochs = 2;

        for (int epoch = 0; epoch < numEpochs; ++epoch) {
            if (isTrainingCanceled) break;
            int successfulSamples = 0;
            int failedSamples = 0;
            long epochStartTime = System.currentTimeMillis();

            for (int i = 0; i < phaseSamples.size(); i++) {
                if (isTrainingCanceled) break;
                TrainingSample sample = phaseSamples.get(i);

                ByteBuffer labelBuffer = ByteBuffer.allocateDirect(4 * 10).order(ByteOrder.nativeOrder());
                for (int j = 0; j < 10; j++) {
                    if (isTrainingCanceled) break;
                    labelBuffer.putFloat(0.0f);
                }
                labelBuffer.rewind();
                labelBuffer.position(4 * sample.label);
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

                    successfulSamples++;

                    if (i % 10 == 0 || i == phaseSamples.size() - 1) {
                        float progress = (float) i / phaseSamples.size() * 100;
                        long elapsedSeconds = (System.currentTimeMillis() - epochStartTime) / 1000;

                        String message = String.format(
                                getString(R.string.training_phase_message),
                                phase.getPhaseNumber(),
                                epoch + 1,
                                numEpochs,
                                progress,
                                i + 1,
                                phaseSamples.size(),
                                elapsedSeconds
                        );
                        System.out.println(message);
                        updateTrainingProgress(message);
                    }
                } catch (IOException e) {
                    showResultDialog(error_title,ExceptionOccurred + e.getMessage());
                    System.out.println("Error processing sample from class " + sample.label + ": " + e.getMessage());
                    failedSamples++;
                }
            }

            System.out.println(String.format(
                    Locale.getDefault(),
                    getString(R.string.phase_epoch_summary),
                    phase.getPhaseNumber(),
                    epoch + 1,
                    numEpochs,
                    successfulSamples,
                    failedSamples
            ));
        }
    }
    private MappedByteBuffer loadModelBuffer(File modelFile) throws IOException {
        FileInputStream inputStream = new FileInputStream(modelFile);
        FileChannel fileChannel = inputStream.getChannel();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, 0, fileChannel.size());
    }
    public class TrainingPhase {
        private final List<TrainingSample> samples;
        private final int startIndex;
        private final int endIndex;
        private final int phaseNumber;
        private TrainingPhase(List<TrainingSample> samples, int startIndex, int endIndex, int phaseNumber) {
            this.samples = samples;
            this.startIndex = startIndex;
            this.endIndex = endIndex;
            this.phaseNumber = phaseNumber;
        }
        private List<TrainingSample> getSamples() {
            return samples.subList(startIndex, endIndex);
        }
        public int getPhaseNumber() {
            return phaseNumber;
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
    private void startTrainingScreen() {
        Intent intent = new Intent(this, TrainingActivity.class);
        startActivity(intent);
    }
    private void finishTraining() {
        Intent intent = new Intent(this, MainActivity.class);
        intent.addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP | Intent.FLAG_ACTIVITY_NEW_TASK);
        System.out.println("\nTraining completed. Saving checkpoint...");
        String message = String.format(
                Locale.getDefault(),
                getString(R.string.training_completed)
        );
        showResultDialog(success_title,message);
        startActivity(intent);
    }
    public void cancelTraining() {
        isTrainingCanceled = true;
    }
    private void updateTrainingProgress(String message) {
        Intent intent = new Intent("UPDATE_TRAINING_PROGRESS");
        intent.putExtra("progressMessage", message);
        LocalBroadcastManager.getInstance(this).sendBroadcast(intent);
    }
    private void performInference(Interpreter featureExtractor, Interpreter classifier, ByteBuffer input) {
        try {
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 28, 28, 1}, DataType.FLOAT32);
            inputFeature0.loadBuffer(input);

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

//            result.append("\nProbabilities:\n" + probabilitiesString);
            System.out.println("Probabilities breakdown:\n" + probabilitiesString);

        } catch (Exception e) {
            showResultDialog(error_title,ExceptionOccurred + e.getMessage());
            e.printStackTrace();
            System.out.println("Error during inference: " + e.getMessage());
        }
    }
    private boolean checkInterpreterInitialized(Interpreter interpreter) {
        if (interpreter == null) {
            showResultDialog(error_title,interpreter_error);
            System.out.println(interpreter_error);
        }
        return true;
    }
    private void saveCheckpoint(Interpreter interpreter) {
        if (!checkInterpreterInitialized(interpreter)) {
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
            showResultDialog(error_title,ExceptionOccurred + e.getMessage());
            System.err.println("Error while saving checkpoint: " + e.getMessage());
            e.printStackTrace();
        }
    }
    private void restoreCheckpoint(Interpreter interpreter) {
        if (!checkInterpreterInitialized(interpreter)) {
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
            showResultDialog(error_title,ExceptionOccurred + e.getMessage());
            System.err.println("Error while restoring checkpoint: " + e.getMessage());
            e.printStackTrace();
        }
    }
    private ByteBuffer preprocessImage(Bitmap image) {
        int modelWidth = 28;
        int modelHeight = 28;
        int channels = 1;  // Grayscale image
        int pixelSize = 4;  // Float32 takes 4 bytes

        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(modelWidth * modelHeight * channels * pixelSize);
        byteBuffer.order(ByteOrder.nativeOrder());

        int[] pixels = new int[modelWidth * modelHeight];
        image.getPixels(pixels, 0, modelWidth, 0, 0, modelWidth, modelHeight);

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

    @Override
     protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data){
        if(resultCode==RESULT_OK){
            if (requestCode==3){
                assert data != null;
                Bitmap image = (Bitmap) Objects.requireNonNull(data.getExtras()).get("data");
                assert image != null;
                int dimension = Math.min(image.getWidth(),image.getHeight());
                image = ThumbnailUtils.extractThumbnail(image,dimension,dimension);
                imageView.setImageBitmap(image);

                image = Bitmap.createScaledBitmap(image, imageSize,imageSize,false);
                System.out.println("Scaled Image Size: Width = " + image.getWidth() + ", Height = " + image.getHeight());
                classifyImageWithInference(image);
            }
            else if (requestCode==2){
                assert data != null;
                Uri dat = data.getData();
                Bitmap image = null;
                try{
                    image = MediaStore.Images.Media.getBitmap(this.getContentResolver(),dat);

                } catch (IOException e) {
                    showResultDialog(error_title,ExceptionOccurred + e.getMessage());
                    e.printStackTrace();
                }
                imageView.setImageBitmap(image);
                assert image != null;
                image = Bitmap.createScaledBitmap(image, imageSize,imageSize,false);
                System.out.println("Scaled Image Size: Width = " + image.getWidth() + ", Height = " + image.getHeight());
                classifyImageWithInference(image);

            }
            else if (requestCode==6){
                assert data != null;
                Uri folderUri = data.getData();
                assert folderUri != null;
                getContentResolver().takePersistableUriPermission(
                        folderUri, Intent.FLAG_GRANT_READ_URI_PERMISSION | Intent.FLAG_GRANT_WRITE_URI_PERMISSION
                );

                SharedPreferences prefs = getSharedPreferences(PREFS_NAME, MODE_PRIVATE);
                prefs.edit().putString(SAVED_URI_KEY, folderUri.toString()).apply();

                DocumentFile directory = DocumentFile.fromTreeUri(this, folderUri);
                if (directory != null && directory.isDirectory()) {
                    classifyAndTrainOnAllClasses2(directory);
                }
            }
        }
        super.onActivityResult(requestCode,resultCode,data);
     }
}