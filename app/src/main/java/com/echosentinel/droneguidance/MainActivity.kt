package com.echosentinel.droneguidance

import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import dji.common.error.DJIError
import dji.common.error.DJISDKError
import dji.sdk.base.BaseProduct
import dji.sdk.sdkmanager.DJISDKManager

class MainActivity : AppCompatActivity() {
    lateinit var droneController: DroneController

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        initSDK()

        droneController = DroneController(this)
        droneController.start()
    }

    private fun initSDK() {
        val sdkManager = DJISDKManager.getInstance()
        sdkManager.registerApp(this.applicationContext, object : DJISDKManager.SDKManagerCallback {
            override fun onRegister(error: DJIError?) {
                if (error == DJISDKError.REGISTRATION_SUCCESS) {
                    Log.i("DJI_SDK", "Registration succeeded")
                    sdkManager.startConnectionToProduct()
                } else {
                    Log.e("DJI_SDK", "Registration failed: ${error?.description}")
                    runOnUiThread {
                        Toast.makeText(this@MainActivity, "DJI SDK Registration Failed", Toast.LENGTH_LONG).show()
                    }
                }
            }

            override fun onProductDisconnect() {
                Log.i("DJI_SDK", "Product Disconnected")
            }

            override fun onProductConnect(product: BaseProduct?) {
                Log.i("DJI_SDK", "Product Connected: ${product?.model}")
                droneController.setProduct(product)
            }

            override fun onProductChanged(product: BaseProduct?) {}
            override fun onComponentChange(componentKey: dji.sdk.sdkmanager.ComponentKey?, oldComponent: Any?, newComponent: Any?) {}
        })
    }
}
