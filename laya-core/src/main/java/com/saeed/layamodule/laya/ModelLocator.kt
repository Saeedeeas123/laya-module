package com.saeed.layamodule.laya

import android.content.Context
import java.io.File

/**
 * Where the 1.29 GB graph lives.
 *
 * The weights are far too big to ship inside an APK, so they are a file the app reads. The app's
 * own media directory (`/sdcard/Android/media/<package>/laya/`) is the intended home: it needs no
 * permission and no user interaction. Callers can pass extra locations (a copy that already
 * exists elsewhere on the device, a Download folder), and every candidate is reported as readable
 * or not — "the model is missing" with no detail is exactly the message that wastes an afternoon.
 */
object ModelLocator {

    const val RELATIVE_PATH = "laya/laya.fp32.onnx"
    private const val MIN_BYTES = 100L * 1024 * 1024

    fun candidates(context: Context, extra: List<File> = emptyList()): List<File> {
        val out = LinkedHashSet<File>()
        context.getExternalMediaDirs()?.forEach { dir ->
            if (dir != null) out.add(File(dir, RELATIVE_PATH))
        }
        out.add(File("/sdcard/Android/media/${context.packageName}/$RELATIVE_PATH"))
        out.addAll(extra)
        return out.toList()
    }

    fun find(context: Context, extra: List<File> = emptyList()): File? =
        candidates(context, extra).firstOrNull {
            it.isFile && it.canRead() && it.length() > MIN_BYTES
        }

    /** A message that names every path tried and what was wrong with it. */
    fun explain(context: Context, extra: List<File> = emptyList()): String {
        val tried = candidates(context, extra)
        val lines = tried.map { file ->
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
            if (tried.isNotEmpty()) {
                append("\n\nPut it at ${tried.first().absolutePath} " +
                    "(copying the file there is enough — no permission prompt, no import step).")
            }
        }
    }
}