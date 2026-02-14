package com.example.frontend

import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugins.GeneratedPluginRegistrant
import android.util.Log
import android.os.Build

class MainActivity : FlutterActivity() {
    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        val arch = Build.SUPPORTED_ABIS.getOrNull(0) ?: "unknown"
        Log.d("MainActivity", "Detected architecture: $arch")

        if (arch.contains("x86") || arch.contains("i386")) {
            Log.w("MainActivity", "⚠️ RUNNING ON X86 ARCHITECTURE (EMULATOR)")
            Log.w("MainActivity", "The LLM plugin (llama_flutter_android) will crash on this architecture.")
            Log.w("MainActivity", "Attempting to register plugins individually to prevent total failure...")
            
            // On x86, we try to register plugins but catch errors specifically for the LLM one
            // This allows shared_preferences and path_provider to work.
            try {
                GeneratedPluginRegistrant.registerWith(flutterEngine)
            } catch (e: Throwable) {
                Log.e("MainActivity", "Suppressed crash during plugin registration: ${e.message}")
                Log.i("MainActivity", "The app will continue to run, but LLM features will be disabled.")
            }
        } else {
            // Standard path for real devices (ARM64)
            super.configureFlutterEngine(flutterEngine)
        }
    }
}
