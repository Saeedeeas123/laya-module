// Laya Module — the Android app + the reusable Laya decision module.
// Toolchain is the FROZEN verified set (see android-build-ship skill): AGP 8.2.2 / Kotlin 1.9.22 /
// Gradle 8.5 / JDK 17 / buildTools 35.0.1. Do not bump without an on-device probe.
plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
}

android {
    namespace = "com.saeed.layamodule"
    compileSdk = 35
    buildToolsVersion = "35.0.1"

    defaultConfig {
        applicationId = "com.saeed.layamodule"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "0.1.0"
    }

    buildTypes {
        release { isMinifyEnabled = false }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }

    buildFeatures {
        compose = true
        // The log records the version, so the generated BuildConfig is needed.
        buildConfig = true
    }
    composeOptions { kotlinCompilerExtensionVersion = "1.5.8" }

    // The tokenizer asset is read with random access, and a compressed asset cannot be mapped.
    androidResources { noCompress += "bin" }

    packaging {
        resources { excludes += "/META-INF/{AL2.0,LGPL2.1}" }
        jniLibs {
            // This device is arm64-v8a; dropping the other ABIs keeps the APK small and avoids
            // shipping native libraries nothing here can load.
            excludes += setOf("lib/x86/**", "lib/x86_64/**", "lib/armeabi-v7a/**")
        }
    }

    testOptions {
        unitTests.isReturnDefaultValues = true
    }
}

dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.androidx.activity.compose)
    implementation(platform(libs.compose.bom))
    implementation(libs.compose.ui)
    implementation(libs.compose.ui.tooling.preview)
    implementation(libs.compose.material3)
    debugImplementation(libs.compose.ui.tooling)

    // Inference and JSON: both already in the local Gradle cache, no new download.
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.18.0")
    implementation("com.google.code.gson:gson:2.10.1")

    // The module's logic is plain Kotlin, so it is tested on the JVM against goldens produced by
    // the real Python implementation (tools/build_assets.py).
    testImplementation("junit:junit:4.13.2")
    testImplementation("com.google.code.gson:gson:2.10.1")
}