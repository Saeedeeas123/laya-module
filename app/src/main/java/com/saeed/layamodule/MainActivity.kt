package com.saeed.layamodule

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Tab
import androidx.compose.material3.TabRow
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.saeed.layamodule.core.RunLog
import com.saeed.layamodule.laya.LayaAnswer
import com.saeed.layamodule.laya.LayaEngine
import com.saeed.layamodule.laya.LayaQuestion
import com.saeed.layamodule.laya.LayaQuestionType
import com.saeed.layamodule.laya.LayaResult
import com.saeed.layamodule.laya.ModelLocator
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        RunLog.install(this, BuildConfig.VERSION_NAME, BuildConfig.VERSION_CODE)
        setContent { LayaApp(this) }
    }
}

// ---------------------------------------------------------------------- app shell

private val ACCENT = Color(0xFF5AD1A0)
private val ACCENT2 = Color(0xFF5AA9FF)
private val WARN = Color(0xFFFFB454)
private val BAD = Color(0xFFFF6B6B)

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun LayaApp(activity: ComponentActivity) {
    var screen by remember { mutableStateOf("main") }
    var engine by remember { mutableStateOf<LayaEngine?>(null) }
    var status by remember { mutableStateOf("model not loaded") }
    var failure by remember { mutableStateOf<String?>(null) }
    var loading by remember { mutableStateOf(false) }
    var steps by remember { mutableStateOf<List<String>>(emptyList()) }

    // Loading is manual, not automatic. The 1.29 GB graph is the riskiest thing this app does; if
    // the process dies during it, the app must still come up so the log can be read and the load
    // retried, rather than dying on every launch.
    fun startLoad() {
        if (loading) return
        loading = true
        failure = null
        steps = emptyList()
        status = "loading…"
        Thread {
            fun ui(block: () -> Unit) = activity.runOnUiThread(block)
            val model = ModelLocator.find(activity)
            if (model == null) {
                val message = ModelLocator.explain(activity)
                RunLog.e("model not found. $message")
                ui { failure = message; status = "model missing"; loading = false }
                return@Thread
            }
            try {
                val loaded = LayaEngine.open(
                    tokenizerStream = activity.assets.open("laya/tokenizer.bin"),
                    configStream = activity.assets.open("laya/config.json"),
                    modelPath = model.absolutePath,
                    probeStream = activity.assets.open("laya/probe.onnx"),
                    cacheDir = activity.cacheDir,
                    log = { line ->
                        RunLog.i(line)
                        ui { steps = steps + line }
                    },
                    memory = { RunLog.memoryLine(activity) },
                    availableMb = { RunLog.availableMb(activity) },
                )
                ui { engine = loaded; status = "ready · ${model.name}"; loading = false }
            } catch (t: Throwable) {
                RunLog.e("engine load failed", t)
                ui {
                    failure = "${t::class.simpleName}: ${t.message}"
                    status = "load failed"
                    loading = false
                }
            }
        }.start()
    }

    MaterialTheme(colorScheme = darkColorScheme(primary = ACCENT, secondary = ACCENT2)) {
        Surface(modifier = Modifier.fillMaxSize()) {
            when (screen) {
                "selftest" -> SelfTestScreen(engine, activity) { screen = "main" }
                "log" -> LogScreen(activity) { screen = "main" }
                else -> MainScreen(engine, status, failure, steps, loading, activity,
                    onLoad = { startLoad() },
                    onSelfTest = { screen = "selftest" }, onLog = { screen = "log" })
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun MainScreen(
    engine: LayaEngine?,
    status: String,
    failure: String?,
    steps: List<String>,
    loading: Boolean,
    activity: ComponentActivity,
    onLoad: () -> Unit,
    onSelfTest: () -> Unit,
    onLog: () -> Unit,
) {
    var tab by remember { mutableStateOf(0) }
    val types = LayaQuestionType.values()

    Scaffold(topBar = {
        TopAppBar(
            title = {
                Column {
                    Text("Laya Module", fontWeight = FontWeight.Bold)
                    Text(status, fontSize = 11.sp, color = Color.Gray)
                }
            },
            actions = {
                TextButton(onClick = onSelfTest) { Text("Self-test") }
                TextButton(onClick = onLog) { Text("Log") }
            },
        )
    }) { padding ->
        Column(Modifier.padding(padding).fillMaxSize()) {
            TabRow(selectedTabIndex = tab) {
                types.forEachIndexed { index, type ->
                    Tab(selected = tab == index, onClick = { tab = index },
                        text = { Text(type.wire) })
                }
            }
            if (engine == null) {
                Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Button(onClick = onLoad, enabled = !loading, modifier = Modifier.fillMaxWidth()) {
                        Text(if (loading) "loading the 1.29 GB graph…" else "Load the model")
                    }
                    failure?.let {
                        Card(colors = CardDefaults.cardColors(containerColor = Color(0xFF2A1A1A))) {
                            Column(Modifier.padding(12.dp)) {
                                Text("Load failed", color = BAD, fontWeight = FontWeight.Bold)
                                Text(it, fontSize = 13.sp)
                            }
                        }
                    }
                    if (steps.isNotEmpty()) {
                        Text("loading steps (also in the log)", fontSize = 11.sp, color = Color.Gray)
                        steps.takeLast(14).forEach { line ->
                            Text(line, fontFamily = FontFamily.Monospace, fontSize = 10.sp,
                                color = if (line.contains("FAILED") || line.contains("could not"))
                                    WARN else Color(0xFFB8C6DD))
                        }
                    }
                }
            }
            PrimitiveScreen(types[tab], engine, activity)
        }
    }
}

// ---------------------------------------------------------------------- one primitive

private data class Draft(val instructions: String, val options: List<Pair<String, String>>,
                         val falseMeaning: String, val trueMeaning: String)

@Composable
private fun PrimitiveScreen(type: LayaQuestionType, engine: LayaEngine?, activity: ComponentActivity) {
    val drafts = remember(type) { mutableStateListOf(defaultDraft(type)) }
    var state by remember(type) { mutableStateOf(defaultState(type)) }
    var stateKey by remember(type) { mutableStateOf("message") }
    var result by remember(type) { mutableStateOf<LayaResult?>(null) }
    var busy by remember(type) { mutableStateOf(false) }
    var error by remember(type) { mutableStateOf<String?>(null) }

    LazyColumn(Modifier.fillMaxSize().padding(12.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item {
            InfoCard(
                title = "The ${type.wire} primitive",
                body = when (type) {
                    LayaQuestionType.CHOICE -> "Pick one label out of a closed set, and get the whole " +
                        "distribution over the labels. Use it for routing, categorising and intent."
                    LayaQuestionType.SCORE -> "Place the state on an ordered scale and get the expected " +
                        "level plus the distribution. Use it for urgency, sentiment and severity."
                    LayaQuestionType.NOUL -> "Ask a yes/no statement and get a calibrated probability, " +
                        "which you threshold yourself. Use it for spam, fraud, churn and safety gates."
                } + "\n\nEverything you write here goes into the prompt, and the wording of the question " +
                    "is part of the job — that is not a figure of speech, it changes the answer."
            )
        }
        item {
            Card {
                Column(Modifier.padding(12.dp)) {
                    Text("State to judge", fontWeight = FontWeight.Bold)
                    OutlinedTextField(
                        value = state, onValueChange = { state = it },
                        modifier = Modifier.fillMaxWidth().height(140.dp),
                        label = { Text("any text: a message, an email, a document") },
                    )
                    Spacer(Modifier.height(6.dp))
                    OutlinedTextField(
                        value = stateKey, onValueChange = { stateKey = it },
                        modifier = Modifier.fillMaxWidth(),
                        label = { Text("state field name (empty = raw text, no JSON wrapper)") },
                        singleLine = true,
                    )
                }
            }
        }
        itemsIndexed(drafts) { index, draft ->
            QuestionEditor(
                type = type, draft = draft,
                canRemove = drafts.size > 1,
                onChange = { drafts[index] = it },
                onRemove = { drafts.removeAt(index) },
            )
        }
        item {
            OutlinedButton(onClick = { drafts.add(defaultDraft(type)) }, modifier = Modifier.fillMaxWidth()) {
                Text("Add another question (all answered in one pass)")
            }
        }
        item {
            Button(
                onClick = {
                    if (engine == null) return@Button
                    busy = true
                    error = null
                    val questions = drafts.mapIndexed { index, draft ->
                        buildQuestion(type, "${type.wire}_${index + 1}", draft)
                    }
                    val stateText = if (stateKey.isBlank()) state else
                        com.google.gson.JsonObject().apply {
                            addProperty(stateKey, state)
                        }.toString()
                    // Compose has no coroutine scope here on purpose: the call blocks, so it runs
                    // on a worker thread and the UI stays responsive.
                    Thread {
                        try {
                            val run = engine.systemOne(stateText, questions)
                            activity.runOnUiThread { result = run; busy = false }
                        } catch (t: Throwable) {
                            RunLog.e("decision failed", t)
                            activity.runOnUiThread {
                                error = "${t::class.simpleName}: ${t.message}"
                                busy = false
                            }
                        }
                    }.start()
                },
                enabled = engine != null && !busy,
                modifier = Modifier.fillMaxWidth(),
            ) { Text(if (busy) "running one forward pass…" else "Run ${drafts.size} question(s)") }
        }
        error?.let { item { InfoCard("Run failed", it, BAD) } }
        result?.let { run ->
            item {
                Card {
                    Column(Modifier.padding(12.dp)) {
                        Text("Decision", fontWeight = FontWeight.Bold)
                        Text(
                            "${run.elapsedMs} ms · ${run.inputTokens} input tokens · output tokens 0 " +
                                "(nothing is generated) · ${RunLog.memoryLine(activity)}",
                            fontSize = 11.sp, color = Color.Gray,
                        )
                        if (run.truncated) {
                            Text(
                                "State was truncated for: ${run.truncatedQuestions.joinToString()} " +
                                    "(${run.stateTokensDropped} tokens dropped)",
                                color = WARN, fontSize = 12.sp,
                            )
                        }
                    }
                }
            }
            items(run.answers) { answer -> AnswerCard(answer) }
        }
    }
}

@Composable
private fun QuestionEditor(
    type: LayaQuestionType,
    draft: Draft,
    canRemove: Boolean,
    onChange: (Draft) -> Unit,
    onRemove: () -> Unit,
) {
    Card(colors = CardDefaults.cardColors(containerColor = Color(0xFF182030))) {
        Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(type.wire, fontWeight = FontWeight.Bold, color = ACCENT2)
                Spacer(Modifier.width(8.dp))
                Text("question", fontSize = 12.sp, color = Color.Gray)
                Spacer(Modifier.weight(1f))
                if (canRemove) TextButton(onClick = onRemove) { Text("remove") }
            }
            OutlinedTextField(
                value = draft.instructions, onValueChange = { onChange(draft.copy(instructions = it)) },
                modifier = Modifier.fillMaxWidth(),
                label = { Text("instructions — the question itself") },
            )
            when (type) {
                LayaQuestionType.CHOICE -> {
                    Text("Options: a label, and optionally what that label means.", fontSize = 12.sp)
                    draft.options.forEachIndexed { index, option ->
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            OutlinedTextField(
                                value = option.first,
                                onValueChange = { value ->
                                    onChange(draft.copy(options = draft.options.toMutableList().also {
                                        it[index] = value to option.second
                                    }))
                                },
                                modifier = Modifier.width(120.dp), label = { Text("label") }, singleLine = true,
                            )
                            Spacer(Modifier.width(6.dp))
                            OutlinedTextField(
                                value = option.second,
                                onValueChange = { value ->
                                    onChange(draft.copy(options = draft.options.toMutableList().also {
                                        it[index] = option.first to value
                                    }))
                                },
                                modifier = Modifier.weight(1f), label = { Text("meaning") }, singleLine = true,
                            )
                            if (draft.options.size > 2) {
                                TextButton(onClick = {
                                    onChange(draft.copy(options = draft.options.toMutableList().also {
                                        it.removeAt(index)
                                    }))
                                }) { Text("×") }
                            }
                        }
                    }
                    TextButton(onClick = {
                        onChange(draft.copy(options = draft.options + ("" to "")))
                    }) { Text("+ add option") }
                }
                LayaQuestionType.SCORE -> {
                    Text("Levels, lowest first. The answer is the expected level on this scale.", fontSize = 12.sp)
                    draft.options.forEachIndexed { index, option ->
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Text("${index}:", fontFamily = FontFamily.Monospace, modifier = Modifier.width(24.dp))
                            OutlinedTextField(
                                value = option.first,
                                onValueChange = { value ->
                                    onChange(draft.copy(options = draft.options.toMutableList().also {
                                        it[index] = value to ""
                                    }))
                                },
                                modifier = Modifier.weight(1f), label = { Text("level") }, singleLine = true,
                            )
                            if (draft.options.size > 2) {
                                TextButton(onClick = {
                                    onChange(draft.copy(options = draft.options.toMutableList().also {
                                        it.removeAt(index)
                                    }))
                                }) { Text("×") }
                            }
                        }
                    }
                    TextButton(onClick = { onChange(draft.copy(options = draft.options + ("" to ""))) }) {
                        Text("+ add level")
                    }
                }
                LayaQuestionType.NOUL -> {
                    OutlinedTextField(
                        value = draft.falseMeaning, onValueChange = { onChange(draft.copy(falseMeaning = it)) },
                        modifier = Modifier.fillMaxWidth(),
                        label = { Text("what FALSE means (optional)") },
                    )
                    OutlinedTextField(
                        value = draft.trueMeaning, onValueChange = { onChange(draft.copy(trueMeaning = it)) },
                        modifier = Modifier.fillMaxWidth(),
                        label = { Text("what TRUE means (optional)") },
                    )
                    Text("You get P(true) back; the threshold is your policy, not the model's.",
                        fontSize = 12.sp, color = Color.Gray)
                }
            }
        }
    }
}

@Composable
private fun AnswerCard(answer: LayaAnswer) {
    Card {
        Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(answer.questionId, fontFamily = FontFamily.Monospace, color = ACCENT2)
                Spacer(Modifier.width(8.dp))
                Text(answer.type.wire, fontSize = 11.sp, color = Color.Gray)
                Spacer(Modifier.weight(1f))
                Text("answer_conf ${fmt(answer.answerConfidence, 3)} · conf ${fmt(answer.confidence, 3)}",
                    fontSize = 11.sp, color = Color.Gray)
            }
            when (answer.type) {
                LayaQuestionType.CHOICE -> {
                    Text(answer.choice ?: "?", fontWeight = FontWeight.Bold, fontSize = 20.sp)
                    answer.probabilities.entries.sortedByDescending { it.value }.forEach { (label, p) ->
                        ProbabilityBar(label, p, label == answer.choice)
                    }
                }
                LayaQuestionType.SCORE -> {
                    Text("${fmt(answer.score ?: 0f, 2)} of ${(answer.legend?.size ?: 1) - 1}  · expected level",
                        fontWeight = FontWeight.Bold, fontSize = 18.sp)
                    answer.probabilities.entries.sortedBy { it.key }.forEach { (level, p) ->
                        ProbabilityBar("${level}: ${answer.legend?.get(level) ?: ""}", p, false)
                    }
                }
                LayaQuestionType.NOUL -> {
                    Text("P(true) = ${fmt(answer.noul ?: 0f, 4)}", fontWeight = FontWeight.Bold, fontSize = 20.sp)
                    ProbabilityBar("false", answer.probabilities["false"] ?: 0f, false)
                    ProbabilityBar("true", answer.probabilities["true"] ?: 0f, (answer.noul ?: 0f) > 0.5f)
                }
            }
            Text("auxiliary action head: P(act) = ${fmt(answer.actProbability, 3)}",
                fontSize = 11.sp, color = Color.Gray)
        }
    }
}

@Composable
private fun ProbabilityBar(label: String, probability: Float, highlight: Boolean) {
    Column {
        Row {
            Text(label, fontSize = 12.sp, modifier = Modifier.weight(1f))
            Text(fmt(probability, 3), fontSize = 12.sp, fontFamily = FontFamily.Monospace)
        }
        Box(
            Modifier.fillMaxWidth().height(8.dp)
                .background(Color(0xFF0A0D13)),
        ) {
            Box(
                Modifier.fillMaxWidth(probability.coerceIn(0f, 1f))
                    .height(8.dp)
                    .background(if (highlight) ACCENT else Color(0xFF4A6A94)),
            )
        }
    }
}

// ---------------------------------------------------------------------- self-test

@Composable
private fun SelfTestScreen(engine: LayaEngine?, activity: ComponentActivity, onBack: () -> Unit) {
    var report by remember { mutableStateOf<List<String>>(emptyList()) }
    var running by remember { mutableStateOf(false) }

    Column(Modifier.fillMaxSize().padding(12.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            TextButton(onClick = onBack) { Text("← back") }
            Text("Self-test", fontWeight = FontWeight.Bold)
        }
        Text(
            "Replays answers recorded from the Python implementation on this checkpoint. If every " +
                "line says OK, this module computes what the reference implementation computes — " +
                "tokenizer, prompt, head and ONNX graph together.",
            fontSize = 12.sp, color = Color.Gray,
        )
        Spacer(Modifier.height(8.dp))
        Button(
            onClick = {
                if (engine == null) return@Button
                running = true
                Thread {
                    val lines = SelfTest.run(engine, activity)
                    activity.runOnUiThread { report = lines; running = false }
                }.start()
            },
            enabled = engine != null && !running,
            modifier = Modifier.fillMaxWidth(),
        ) { Text(if (running) "running…" else "Run the parity checks") }
        Spacer(Modifier.height(8.dp))
        LazyColumn(Modifier.fillMaxSize()) {
            items(report) { line ->
                Text(
                    line,
                    fontFamily = FontFamily.Monospace,
                    fontSize = 11.sp,
                    color = when {
                        line.startsWith("FAIL") -> BAD
                        line.startsWith("OK") -> ACCENT
                        else -> Color.LightGray
                    },
                )
            }
        }
    }
}

// ---------------------------------------------------------------------- log viewer

@Composable
private fun LogScreen(activity: ComponentActivity, onBack: () -> Unit) {
    var lines by remember { mutableStateOf(RunLog.tail(400)) }
    var sessions by remember { mutableStateOf(RunLog.sessions(activity)) }
    var opened by remember { mutableStateOf<String?>(null) }
    var tick by remember { mutableStateOf(0) }

    // Live view: the log is the app's only diagnostic channel, so it refreshes itself.
    LaunchedEffect(tick) {
        while (true) {
            lines = RunLog.tail(400)
            sessions = RunLog.sessions(activity)
            kotlinx.coroutines.delay(1500)
        }
    }

    Column(Modifier.fillMaxSize().padding(12.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            TextButton(onClick = onBack) { Text("← back") }
            Text("Log", fontWeight = FontWeight.Bold)
            Spacer(Modifier.weight(1f))
            TextButton(onClick = { tick++ }) { Text("refresh") }
        }
        Text(RunLog.fileName, fontSize = 11.sp, fontFamily = FontFamily.Monospace)
        Text(
            (if (RunLog.degraded) "DEGRADED: ${RunLog.degradedReason}" else "public copy: " +
                (RunLog.publicPath ?: "unavailable")),
            fontSize = 11.sp,
            color = if (RunLog.degraded) WARN else Color.Gray,
        )
        Spacer(Modifier.height(6.dp))
        Text("previous runs (${sessions.size})", fontSize = 11.sp, color = Color.Gray)
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            sessions.take(4).forEach { file ->
                TextButton(onClick = { opened = RunLog.read(file, 400) }) {
                    Text(file.name.removePrefix("layamodule-").removeSuffix(".log"), fontSize = 10.sp)
                }
            }
        }
        HorizontalDivider()
        LazyColumn(Modifier.fillMaxSize()) {
            items(opened?.split("\n") ?: lines) { line ->
                Text(line, fontFamily = FontFamily.Monospace, fontSize = 10.sp, color = Color(0xFFB8C6DD))
            }
        }
    }
}

// ---------------------------------------------------------------------- helpers

@Composable
private fun InfoCard(title: String, body: String, accent: Color = ACCENT2) {
    Card {
        Column(Modifier.padding(12.dp)) {
            Text(title, fontWeight = FontWeight.Bold, color = accent)
            Text(body, fontSize = 13.sp)
        }
    }
}

private fun fmt(value: Float, digits: Int): String =
    String.format(java.util.Locale.US, "%.${digits}f", value)

private fun defaultDraft(type: LayaQuestionType): Draft = when (type) {
    LayaQuestionType.CHOICE -> Draft(
        "Which department should handle this?",
        listOf("billing" to "invoices, payments, refunds",
            "technical" to "bugs, outages, system errors",
            "other" to "everything else"),
        "", "",
    )
    LayaQuestionType.SCORE -> Draft(
        "How urgent is this?",
        listOf("not urgent" to "", "soon" to "", "blocking" to ""),
        "", "",
    )
    LayaQuestionType.NOUL -> Draft(
        "Does the user request a refund?", emptyList(),
        "the user is not asking for money back", "the user asks for a refund or a credit",
    )
}

private fun defaultState(type: LayaQuestionType): String = when (type) {
    LayaQuestionType.CHOICE -> "Hi, we were billed twice for March. Please refund the duplicate today."
    LayaQuestionType.SCORE -> "Your API has been returning 500 errors for the last hour and our checkout is down."
    LayaQuestionType.NOUL -> "Please refund the duplicate charge from March, or we will cancel our plan."
}

private fun buildQuestion(type: LayaQuestionType, id: String, draft: Draft): LayaQuestion =
    LayaQuestion(
        id = id,
        type = type,
        instructions = draft.instructions,
        options = when (type) {
            LayaQuestionType.NOUL -> emptyList()
            else -> draft.options.filter { it.first.isNotBlank() }
        },
        falseMeaning = draft.falseMeaning.ifBlank { null },
        trueMeaning = draft.trueMeaning.ifBlank { null },
    )
