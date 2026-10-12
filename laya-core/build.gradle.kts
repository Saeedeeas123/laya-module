// laya-core — the Laya decision engine as a reusable Android library.
//
// Everything in here is plain Kotlin plus ONNX Runtime: no UI, no app lifecycle, no logging
// framework, no dependency on the app that consumes it. That is what makes it a module rather
// than a piece of an app: another app adds one line to its build file and gets the engine.
//
//   implementation(project(":laya-core"))                       // in this repo
//   implementation("com.saeed.layamodule:laya-core:1.0.0")      // from mavenLocal (see README)
plugins {
    alias(libs.plugins.android.library)
    alias(libs.plugins.kotlin.android)
    id("maven-publish")
}

android {
    namespace = "com.saeed.layamodule.laya"
    compileSdk = 35
    buildToolsVersion = "35.0.1"

    defaultConfig {
        minSdk = 26
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }

    // The tokenizer asset is read with random access; a compressed asset cannot be mapped.
    androidResources { noCompress += "bin" }

    testOptions {
        unitTests.isReturnDefaultValues = true
    }
}

dependencies {
    // ONNX Runtime and Gson are implementation details of the module: they are packaged into the
    // consuming app but do not leak onto its compile classpath.
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.18.0")
    implementation("com.google.code.gson:gson:2.10.1")

    // The module's logic is plain Kotlin, so it is verified on the JVM against goldens produced by
    // the reference Python implementation (tools/build_assets.py in the repository root).
    testImplementation("junit:junit:4.13.2")
    testImplementation("com.google.code.gson:gson:2.10.1")
}

// Publishing, so the module can be consumed by another project on this machine without copying
// source: ./gradlew :laya-core:publishToMavenLocal, then in the consumer
//   repositories { mavenLocal() }
//   implementation("com.saeed.layamodule:laya-core:1.0.0")
// A private GitHub repo cannot be served by JitPack, so mavenLocal (or a private Maven host) is
// the supported route.
group = "com.saeed.layamodule"
version = "1.0.0"

afterEvaluate {
    publishing {
        publications {
            create<MavenPublication>("release") {
                from(components["release"])
                groupId = "com.saeed.layamodule"
                artifactId = "laya-core"
                version = "1.0.0"
            }
        }
    }
}