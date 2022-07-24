package com.project.FireDetector
import android.Manifest
import android.content.pm.PackageManager
import android.graphics.*
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import android.util.Size
import android.view.Surface
import android.view.ViewGroup
import android.widget.Toast
import androidx.camera.core.*
import kotlinx.android.synthetic.main.activity_main.*
import com.project.FireDetector.ml.FireDetection
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.common.TensorProcessor
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.label.TensorLabel


import java.io.ByteArrayOutputStream
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    companion object {
        const val REQUEST_CODE_PERMISSIONS = 5
        const val REQUIRED_PERMISSION = Manifest.permission.CAMERA
    }


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
//        var bitmap =
//            BitmapFactory.decodeResource(applicationContext.resources,R.drawable.picture);
//
//        processImage(bitmap)

        if (hasPermission()) {
            view_finder.post { startCamera() }
        } else {
            requestPermission()
        }

        view_finder.addOnLayoutChangeListener { _, _, _, _, _, _, _, _, _ ->
            updateTransform()
        }
    }


    private val executor = Executors.newSingleThreadExecutor()

    private fun startCamera() {
        val previewConfig = PreviewConfig.Builder().apply {
            setTargetResolution(Size(640, 480))
        }.build()


        val preview = Preview(previewConfig)

        preview.setOnPreviewOutputUpdateListener {

            val parent = view_finder.parent as ViewGroup
            parent.removeView(view_finder)
            parent.addView(view_finder, 0)

            view_finder.surfaceTexture = it.surfaceTexture
            updateTransform()
        }

        val analyzerConfig = ImageAnalysisConfig.Builder().apply {
            setImageReaderMode(
                ImageAnalysis.ImageReaderMode.ACQUIRE_LATEST_IMAGE) // analyze latest images (not every image)
        }.build()

        val analyzer = ImageAnalysis.Analyzer { image: ImageProxy, _: Int ->
            val bitmap = image.toBitmap()
            processImage(bitmap)

        }

        val analyzerUseCase = ImageAnalysis(analyzerConfig).apply {
            setAnalyzer(executor, analyzer)
        }

        CameraX.bindToLifecycle(this, preview, analyzerUseCase)
    }

    private fun ImageProxy.toBitmap(): Bitmap {
        val yBuffer = planes[0].buffer // Y
        val uBuffer = planes[1].buffer // U
        val vBuffer = planes[2].buffer // V

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)

        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        val yuvImage = YuvImage(nv21, ImageFormat.NV21, this.width, this.height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, yuvImage.width, yuvImage.height), 100, out)
        val imageBytes = out.toByteArray()
        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
    }

    private fun updateTransform() {
        val matrix = Matrix()

        // Compute the center of the view finder
        val centerX = view_finder.width / 2f
        val centerY = view_finder.height / 2f

        val rotationDegrees = when(view_finder.display.rotation) {
            Surface.ROTATION_0 -> 0
            Surface.ROTATION_90 -> 90
            Surface.ROTATION_180 -> 180
            Surface.ROTATION_270 -> 270
            else -> return
        }
        matrix.postRotate(-rotationDegrees.toFloat(), centerX, centerY)

        view_finder.setTransform(matrix)
    }


    private fun hasPermission(): Boolean =
        checkSelfPermission(REQUIRED_PERMISSION) == PackageManager.PERMISSION_GRANTED

    private fun requestPermission() =
        requestPermissions(arrayOf(REQUIRED_PERMISSION), REQUEST_CODE_PERMISSIONS)

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grantResults: IntArray
    ) {
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (hasPermission()) {
                startCamera()
            } else {
                Toast.makeText(this, "Permissions not granted by the user.", Toast.LENGTH_SHORT)
                    .show()
                finish()
            }
        }
    }


    private fun processImage(bitmap: Bitmap) {
        try {
            var tfImage = TensorImage(DataType.FLOAT32)
            tfImage.load(bitmap)

            val imageProcessor = ImageProcessor.Builder()
                .add(ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
                .build()
            tfImage = imageProcessor.process(tfImage)

            val model = FireDetection.newInstance(this@MainActivity)
            val probabilityProcessor =
                TensorProcessor.Builder().add(NormalizeOp(0f, 255f)).build()

            val outputs =
                model.process(probabilityProcessor.process(tfImage.tensorBuffer))
            val outputBuffer = outputs.outputFeature0AsTensorBuffer

            val tensorLabel =
                TensorLabel(arrayListOf("fire", "not_fire"), outputBuffer)

            val probability = tensorLabel.mapWithFloatValue["fire"]
            probability?.let {
                if (it > 0.60) {
                    tv_result.text = "Fire!"
                } else {
                    tv_result.text = "Not Fire!"
                }
            }
            Log.d("sdf", "Fire : " + probability)
        } catch (e: Exception) {
            Log.d("sdf", "Exception is " + e.localizedMessage)
        }
    }
}