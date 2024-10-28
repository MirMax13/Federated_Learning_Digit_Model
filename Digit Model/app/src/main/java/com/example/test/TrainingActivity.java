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

public class TrainingActivity extends AppCompatActivity {
    private TextView trainingProgress;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_training);

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        trainingProgress = findViewById(R.id.progressText);

        Button cancelButton = findViewById(R.id.cancel_training_button);
        cancelButton.setOnClickListener(v -> {
            new AlertDialog.Builder(this)
                    .setMessage("Are you sure you want to cancel the workout? The training results will be lost")
                    .setPositiveButton("Yes, cancel", (dialog, which) -> {
                        Intent intent = new Intent("CANCEL_TRAINING");
                        LocalBroadcastManager.getInstance(this).sendBroadcast(intent);
                        finish();
                    })
                    .setNegativeButton("No, return", (dialog, which) -> {
                        dialog.dismiss();
                    })
                    .create()
                    .show();
        });

    }

    public void updateTrainingProgress(String progressMessage) {
        runOnUiThread(() -> trainingProgress.setText(progressMessage));
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
}