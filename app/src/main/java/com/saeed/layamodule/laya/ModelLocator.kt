package com.saeed.layamodule.laya

import android.content.Context
import java.io.File

/**
 * Where the 1.29 GB graph lives.
 *
 * The weights are far too big to ship inside an APK, so they are a file the app reads. The app's
 * own media directory (`/sdcard/Android/media/<package>/laya/`) is the intended home: it needs no
 * permission and no user interaction. The other entries exist so an existing copy on the device
 * can be used as-is, and each candidate is reported as readable or not, because "the model is
 * missing" with no detail is exactly the kind of message that wastes an afternoon.
 */
object ModelLocator {

    const val RELATIVE_PATH = "laya/laya.fp32.onnx"
    private const val MIN_BYTES = 100L * 1024 * 1024

    fun candidates(context: Context): List<File> {
        val out = LinkedHashSet<File>()
        context.getExternalMediaDirs()?.forEach { dir ->
            if (dir != null) out.add(File(dir, RELATIVE_PATH))
        }
        out.add(File("/sdcard/Android/media/com.saeed.layamodule/$RELATIVE_PATH"))
        // A copy that already exists on this device. Readable or not depends on scoped storage,
        // so it is tried rather than assumed.
        out.add(File("/sdcard/Android/media/com.saeed.cosmo/$RELATIVE_PATH"))
        out.add(File("/sdcard/Download/LayaModule/laya.fp32.onnx"))
        return out.toList()
    }

    fun find(context: Context): File? = candidates(context).firstOrNull {
        it.isFile && it.canRead() && it.length() > MIN_BYTES
    }

    /** A message that names every path tried and what was wrong with it. */
    fun explain(context: Context): String {
        val lines = candidates(context).map { file ->
            val state = when {
                !file.exists() -> "missing"
                !file.canRead() -> "exists but not readable"
                file.length() <= MIN_BYTES -> "too small (${file.length() / 1024 / 1024} MB)"
                else -> "usable"
            }
            "${file.absolutePath}  →  $state"
        }
        return buildString {
            append("The Laya graph was not found. Expected about 1.29 GB at:\n")
            append(lines.joinToString("\n"))
            append("\n\nPut it at ${candidates(context).first().absolutePath} " +
                "(copying the file there is enough — no permission prompt, no import step).")
        }
    }
}