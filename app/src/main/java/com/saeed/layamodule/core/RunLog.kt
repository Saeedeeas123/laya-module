package com.saeed.layamodule.core

import android.app.ActivityManager
import android.content.ContentValues
import android.content.Context
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import java.io.File
import java.io.OutputStream
import java.text.SimpleDateFormat
import java.util.ArrayDeque
import java.util.Date
import java.util.Locale

/**
 * The app's own log — the only diagnostic channel this app has.
 *
 * Saeed has no Android Studio and no logcat, so the rules are hard ones:
 *   * every line is written and flushed as it happens, so a crash mid-action still leaves the
 *     file holding everything up to that instant;
 *   * one timestamped file per run, previous runs kept (~10), never overwritten;
 *   * written where it is always writable (`getExternalFilesDir/logs`) AND to a public path
 *     (`Download/LayaModule/logs`) so it can be read without a PC — through MediaStore, which
 *     needs no permission;
 *   * millisecond timestamps, and app/device/OS/ABI recorded at the start;
 *   * memory (app RSS + device available) on demand;
 *   * every exception with a full stack trace, and a global handler that writes before the
 *     process dies;
 *   * if a file cannot be opened, that is recorded and shown in the UI — never swallowed.
 */
object RunLog {

    private const val TAG = "layamodule"
    private const val RETAIN = 10

    private val timestamp = SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS", Locale.US)
    private val runStamp = SimpleDateFormat("yyyyMMdd-HHmmss", Locale.US)

    private val ring = ArrayDeque<String>()
    private val lock = Any()

    private var privateFile: File? = null
    private var privateStream: OutputStream? = null
    private var publicStream: OutputStream? = null
    private var publicUri: android.net.Uri? = null

    @Volatile
    var degraded: Boolean = false
        private set

    @Volatile
    var degradedReason: String? = null
        private set

    val fileName: String get() = fileNameValue
    private var fileNameValue: String = "layamodule-not-started.log"

    val publicPath: String? get() = publicUri?.let { "Download/LayaModule/logs/$fileName" }

    /** Call once, from Application.onCreate or the first Activity onCreate. */
    fun install(context: Context, versionName: String, versionCode: Int) {
        if (privateStream != null) return
        synchronized(lock) {
            if (privateStream != null) return
            fileNameValue = "$TAG-${runStamp.format(Date())}.log"
            openPrivate(context)
            openPublic(context)
            prune(context)
            installCrashHandler()
            i("=== run start ===")
            i("app version $versionName ($versionCode)")
            i("device ${Build.MANUFACTURER} ${Build.MODEL} (${Build.DEVICE})")
            i("android ${Build.VERSION.RELEASE} (sdk ${Build.VERSION.SDK_INT}) abis ${Build.SUPPORTED_ABIS.joinToString()}")
            i("log file $fileNameValue")
            i("private path ${privateFile?.absolutePath}")
            i("public path ${publicPath ?: "unavailable"}")
            if (degraded) i("LOGGER DEGRADED: $degradedReason")
            i("memory ${memoryLine(context)}")
        }
    }

    private fun openPrivate(context: Context) {
        try {
            val dir = File(context.getExternalFilesDir(null), "logs")
            if (!dir.exists()) dir.mkdirs()
            val file = File(dir, fileNameValue)
            privateFile = file
            privateStream = file.outputStream().buffered()
        } catch (t: Throwable) {
            degraded = true
            degradedReason = "private log file: ${t.message}"
        }
    }

    /**
     * A second copy in Download/, through MediaStore so no permission is needed. Appending is
     * done by holding the stream open for the run ("wa" mode), which is what keeps the per-line
     * flush cheap.
     */
    private fun openPublic(context: Context) {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) {
            degraded = true
            degradedReason = "public log needs Android 10+ for MediaStore appends"
            return
        }
        try {
            val values = ContentValues().apply {
                put(MediaStore.MediaColumns.DISPLAY_NAME, fileNameValue)
                put(MediaStore.MediaColumns.MIME_TYPE, "text/plain")
                put(MediaStore.MediaColumns.RELATIVE_PATH, "${Environment.DIRECTORY_DOWNLOADS}/LayaModule/logs")
            }
            val collection = MediaStore.Downloads.EXTERNAL_CONTENT_URI
            val uri = context.contentResolver.insert(collection, values)
            if (uri == null) {
                degraded = true
                degradedReason = "MediaStore refused a Download/ entry"
                return
            }
            publicUri = uri
            publicStream = context.contentResolver.openOutputStream(uri, "wa")
            if (publicStream == null) {
                degraded = true
                degradedReason = "Download/ entry created but not writable"
            }
        } catch (t: Throwable) {
            degraded = true
            degradedReason = "public log file: ${t.message}"
        }
    }

    private fun prune(context: Context) {
        try {
            val dir = File(context.getExternalFilesDir(null), "logs")
            val files = dir.listFiles { f -> f.name.startsWith("$TAG-") && f.name.endsWith(".log") }
                ?.sortedBy { it.name } ?: return
            files.dropLast(RETAIN).forEach { it.delete() }
        } catch (_: Throwable) {
        }
    }

    private fun installCrashHandler() {
        val previous = Thread.getDefaultUncaughtExceptionHandler()
        Thread.setDefaultUncaughtExceptionHandler { thread, throwable ->
            e("FATAL uncaught exception in ${thread.name}", throwable)
            i("=== run end (crash) ===")
            runCatching { privateStream?.flush(); publicStream?.flush() }
            previous?.uncaughtException(thread, throwable)
        }
    }

    // ------------------------------------------------------------------ writing
    private fun write(level: String, message: String) {
        val line = "${timestamp.format(Date())} | $level | $message"
        synchronized(lock) {
            ring.addLast(line)
            if (ring.size > 4000) ring.removeFirst()
            for (stream in listOf(privateStream, publicStream)) {
                try {
                    stream?.write((line + "\n").toByteArray())
                    stream?.flush()
                } catch (t: Throwable) {
                    degraded = true
                    degradedReason = "write failed: ${t.message}"
                }
            }
        }
    }

    fun i(message: String) = write("INFO ", message)

    fun e(message: String, throwable: Throwable? = null) {
        write("ERROR", message)
        if (throwable != null) {
            for (line in throwable.stackTraceToString().split("\n")) write("ERROR", "    $line")
        }
    }

    /** App RSS and device memory, for the "what was it doing when it died" question. */
    fun memoryLine(context: Context): String {
        val rssKb = try {
            File("/proc/self/statm").readText().split(" ")[1].toLong() * 4096 / 1024
        } catch (_: Throwable) {
            -1L
        }
        val info = ActivityManager.MemoryInfo()
        (context.getSystemService(Context.ACTIVITY_SERVICE) as? ActivityManager)?.getMemoryInfo(info)
        return "appRss=${rssKb / 1024}MB deviceAvail=${info.availMem / 1024 / 1024}MB lowMemory=${info.lowMemory}"
    }

    /** Free device memory in MB — used to refuse a load that the system would kill. */
    fun availableMb(context: Context): Long {
        val info = ActivityManager.MemoryInfo()
        (context.getSystemService(Context.ACTIVITY_SERVICE) as? ActivityManager)?.getMemoryInfo(info)
        return info.availMem / 1024 / 1024
    }

    // ------------------------------------------------------------------ viewing
    fun tail(limit: Int = 400): List<String> = synchronized(lock) {
        ring.toList().takeLast(limit)
    }

    /** Previous runs, newest first. */
    fun sessions(context: Context): List<File> {
        val dir = File(context.getExternalFilesDir(null), "logs")
        return dir.listFiles { f -> f.name.startsWith("$TAG-") && f.name.endsWith(".log") }
            ?.sortedByDescending { it.name } ?: emptyList()
    }

    fun read(file: File, limit: Int = 4000): String = try {
        file.readLines().takeLast(limit).joinToString("\n")
    } catch (t: Throwable) {
        "cannot read ${file.name}: ${t.message}"
    }
}