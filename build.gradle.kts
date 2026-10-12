// Top-level build file. Plugin versions come from gradle/libs.versions.toml.
// Every plugin a subproject applies must be declared here with `apply false`, otherwise Gradle
// resolves it a second time in the subproject and fails with "already on the classpath with an
// unknown version".
plugins {
    alias(libs.plugins.android.application) apply false
    alias(libs.plugins.android.library) apply false
    alias(libs.plugins.kotlin.android) apply false
}
