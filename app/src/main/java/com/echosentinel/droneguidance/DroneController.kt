
import android.content.Context
import android.util.Log
import dji.common.camera.SettingsDefinitions
import dji.sdk.base.BaseProduct
import dji.sdk.camera.Camera

class DroneController(private val context: Context) {
    private var product: BaseProduct? = null
    private var camera: Camera? = null

    fun setProduct(product: BaseProduct?) {
        this.product = product
        if (product != null) {
            camera = product.camera
            initCamera()
        }
    }

    private fun initCamera() {
        camera?.setMode(SettingsDefinitions.CameraMode.SHOOT_PHOTO) { error ->
            if (error != null) {
                Log.e("DroneController", "Camera init error: ${error.description}")
            } else {
                Log.i("DroneController", "Camera initialized")
            }
        }
    }

    fun start() {
        // démarrage de la télémétrie, flux vidéo, etc. à implémenter ici
    }

    fun stop() {
        // libérer ressources si besoin
    }
}
