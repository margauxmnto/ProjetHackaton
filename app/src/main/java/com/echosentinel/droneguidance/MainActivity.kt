package com.echosentinel.droneguidance

import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import dji.common.error.DJIError
import dji.common.error.DJISDKError
import dji.sdk.base.BaseProduct
import dji.sdk.sdkmanager.DJISDKManager
import java.util.concurrent.atomic.AtomicBoolean

class MainActivity : AppCompatActivity() {

    private val REQUIRED_PERMISSION_LIST: Array<String> = arrayOf(
        Manifest.permission.VIBRATE,
        Manifest.permission.INTERNET,
        Manifest.permission.ACCESS_WIFI_STATE,
        Manifest.permission.WAKE_LOCK,
        Manifest.permission.ACCESS_COARSE_LOCATION,
        Manifest.permission.ACCESS_NETWORK_STATE,
        Manifest.permission.ACCESS_FINE_LOCATION,
        Manifest.permission.CHANGE_WIFI_STATE,
        Manifest.permission.WRITE_EXTERNAL_STORAGE,
        Manifest.permission.BLUETOOTH,
        Manifest.permission.BLUETOOTH_ADMIN,
        Manifest.permission.READ_EXTERNAL_STORAGE,
        Manifest.permission.READ_PHONE_STATE,
    )
    private const val REQUEST_PERMISSION_CODE = 12345
    private val isRegistrationInProgress = AtomicBoolean(false)

    private lateinit var textSdkStatus: TextView
    private lateinit var textDroneStatus: TextView
    private lateinit var textTelemetry: TextView
    private lateinit var textVictimStatus: TextView

    lateinit var droneController: DroneController

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        textSdkStatus = findViewById(R.id.text_sdk_status)
        textDroneStatus = findViewById(R.id.text_drone_status)
        textTelemetry = findViewById(R.id.text_telemetry)
        textVictimStatus = findViewById(R.id.text_victim_status)

        checkAndRequestPermissions()

        droneController = DroneController(this)
        droneController.start()
    }

    private fun checkAndRequestPermissions() {
        val missingPermissions = mutableListOf<String>()
        for (permission in REQUIRED_PERMISSION_LIST) {
            if (ContextCompat.checkSelfPermission(this, permission) != PackageManager.PERMISSION_GRANTED) {
                missingPermissions.add(permission)
            }
        }
        if (missingPermissions.isNotEmpty()) {
            ActivityCompat.requestPermissions(this, missingPermissions.toTypedArray(), REQUEST_PERMISSION_CODE)
        } else {
            startSDKRegistration()
        }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_PERMISSION_CODE) {
            val allPermissionsGranted = grantResults.all { it == PackageManager.PERMISSION_GRANTED }
            if (allPermissionsGranted) {
                startSDKRegistration()
            } else {
                Toast.makeText(this, "Permissions are required for the app to function.", Toast.LENGTH_LONG).show()
                finish()
            }
        }
    }

    private fun startSDKRegistration() {
        if (isRegistrationInProgress.compareAndSet(false, true)) {
            DJISDKManager.getInstance().registerApp(this, object: DJISDKManager.SDKManagerCallback {
                override fun onRegister(error: DJIError?) {
                    if (error == DJISDKError.REGISTRATION_SUCCESS) {
                        runOnUiThread { textSdkStatus.text = getString(R.string.sdk_status_registered) }
                        DJISDKManager.getInstance().startConnectionToProduct()
                    } else {
                        runOnUiThread { textSdkStatus.text = getString(R.string.sdk_status_unregistered) }
                        Log.e("DJI_SDK", "Registration failed: ${error?.description}")
                        Toast.makeText(this@MainActivity, "DJI SDK Registration Failed", Toast.LENGTH_LONG).show()
                    }
                    isRegistrationInProgress.set(false)
                }

                override fun onProductDisconnect() {
                    runOnUiThread { textDroneStatus.text = getString(R.string.connection_status_disconnected) }
                    Log.i("DJI_SDK", "Product Disconnected")
                }

                override fun onProductConnect(product: BaseProduct?) {
                    runOnUiThread { textDroneStatus.text = getString(R.string.connection_status_connected, product?.model) }
                    Log.i("DJI_SDK", "Product Connected: ${product?.model}")
                    droneController.setProduct(product)
                }

                override fun onProductChanged(product: BaseProduct?) {}
                override fun onComponentChange(componentKey: dji.sdk.sdkmanager.ComponentKey?, oldComponent: Any?, newComponent: Any?) {}
            })
        }
    }

    fun updateTelemetry(telemetry: String) {
        runOnUiThread {
            textTelemetry.text = telemetry
        }
    }

    fun updateVictimStatus(status: String) {
        runOnUiThread {
            textVictimStatus.text = status
        }
    }
}
