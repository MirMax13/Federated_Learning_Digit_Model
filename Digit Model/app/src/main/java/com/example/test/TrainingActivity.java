package com.example.test;

import android.app.AlertDialog;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.os.Bundle;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.TextView;

import androidx.localbroadcastmanager.content.LocalBroadcastManager;
import androidx.appcompat.app.AppCompatActivity;

import java.util.ArrayDeque;

public class TrainingActivity extends AppCompatActivity {
    private static TrainingActivity instance;
    private TextView logTextView;
    private final ArrayDeque<String> progressMessages = new ArrayDeque<>();
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_training);
        instance = this;
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        logTextView = findViewById(R.id.progressText);

        Button cancelButton = findViewById(R.id.cancel_training_button);
        cancelButton.setOnClickListener(v -> new AlertDialog.Builder(this)
                .setMessage(getString(R.string.cancel_training_message))
                .setPositiveButton(getString(R.string.positive_button_text), (dialog, which) -> {
                    Intent intent = new Intent("CANCEL_TRAINING");
                    LocalBroadcastManager.getInstance(this).sendBroadcast(intent);
                    finish();
                })
                .setNegativeButton(getString(R.string.negative_button_text), (dialog, which) -> dialog.dismiss())
                .create()
                .show());
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (instance == this) {
            instance = null;
        }
    }

    public static TrainingActivity getInstance() {
        return instance;
    }

    public void updateTrainingProgress(String progressMessage) {
        runOnUiThread(() -> {
            if (progressMessages.size() >= 10) {
                progressMessages.pollFirst();
            }
            progressMessages.addLast(progressMessage);
            StringBuilder displayedMessages = new StringBuilder();
            for (String message : progressMessages) {
                displayedMessages.append(message).append("\n");
            }
            logTextView.setText(displayedMessages.toString());
        });
    }
    @Override
    protected void onResume() {
        super.onResume();
        LocalBroadcastManager.getInstance(this).registerReceiver(progressReceiver, new IntentFilter("UPDATE_TRAINING_PROGRESS"));
        LocalBroadcastManager.getInstance(this).registerReceiver(cancelReceiver, new IntentFilter("CANCEL_TRAINING"));
    }

    @Override
    protected void onPause() {
        super.onPause();
        LocalBroadcastManager.getInstance(this).unregisterReceiver(progressReceiver);
        LocalBroadcastManager.getInstance(this).unregisterReceiver(cancelReceiver);
    }

    private final BroadcastReceiver cancelReceiver = new BroadcastReceiver() {
        @Override
        public void onReceive(Context context, Intent intent) {
            MainActivity mainActivity = MainActivity.getInstance();
            if (mainActivity != null) {
                mainActivity.cancelTraining();
            }
        }
    };

    private final BroadcastReceiver progressReceiver = new BroadcastReceiver() {
        @Override
        public void onReceive(Context context, Intent intent) {
            String progressMessage = intent.getStringExtra("progressMessage");
            updateTrainingProgress(progressMessage);
        }
    };

    public void checkAndShowErrorDialog(String message) {
        runOnUiThread(() -> new AlertDialog.Builder(this)
                .setMessage(message)
                .setPositiveButton("OK", (dialog, which) -> {
                    Intent intent = new Intent(this, MainActivity.class);
                    intent.addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP | Intent.FLAG_ACTIVITY_NEW_TASK);
                    MainActivity.getInstance().cancelTraining();
                    finish();
                })
                .create()
                .show());
    }
}