package com.saeed.layamodule

import android.content.Context
import com.saeed.layamodule.core.RunLog
import com.saeed.layamodule.laya.LayaEngine
import com.saeed.layamodule.laya.ModelLocator
import java.io.File

/**
 * The engine belongs to the process, not to the screen.
 *
 * The first on-device run showed why: the engine was held in Compose `remember` state, so an
 * activity recreation lost it while the native session stayed in memory — pressing "Load" again
 * built a *second* 1.5 GB session (RSS went 1501 MB → 2772 MB). Holding it here means one session
 * per process, a retry that reuses a loaded engine, and no way to double-load by tapping.
 */
object LayaRuntime {

    @Volatile
    private var instance: LayaEngine? = null

    @Volatile
    var status: String = "model not loaded"
        private set

    /** A copy of the graph that already exists on this device, used only if our own dir is empty. */
    private val extraPaths: List<File> = listOf(
        File("/sdcard/Android/media/com.saeed.cosmo/${ModelLocator.RELATIVE_PATH}"),
        File("/sdcard/Download/LayaModule/laya.fp32.onnx"),
    )

    fun engine(): LayaEngine? = instance

    fun modelFile(context: Context): File? = ModelLocator.find(context, extraPaths)

    fun missingModelMessage(context: Context): String = ModelLocator.explain(context, extraPaths)

    /**
     * Load the engine, reporting each step. Blocking — call it off the UI thread. A second call
     * while one is loaded returns the loaded engine instead of building another session.
     */
    @Synchronized
    fun load(context: Context, onStep: (String) -> Unit): LayaEngine {
        instance?.let {
            status = "ready"
            return it
        }
        val model = modelFile(context)
            ?: throw IllegalStateException(missingModelMessage(context))
        status = "loading…"
        val loaded = LayaEngine.open(
            tokenizerStream = context.assets.open("laya/tokenizer.bin"),
            configStream = context.assets.open("laya/config.json"),
            modelPath = model.absolutePath,
            probeStream = context.assets.open("laya/probe.onnx"),
            cacheDir = context.cacheDir,
            log = { line ->
                RunLog.i(line)
                onStep(line)
            },
            memory = { RunLog.memoryLine(context) },
            availableMb = { RunLog.availableMb(context) },
        )
        instance = loaded
        status = "ready · ${model.name}"
        RunLog.i("engine ready; ${RunLog.memoryLine(context)}")
        return loaded
    }

    /** Release the session (used by the self-test teardown and the "unload" action). */
    @Synchronized
    fun unload() {
        instance?.close()
        instance = null
        status = "model not loaded"
    }
}