pluginManagement {
    repositories {
        google()
        // Google's mirror of Maven Central first: Maven Central answers CI runners with HTTP 429
        // (Robolectric's nativeruntime jar, 2026-11-22/23). Same artifacts, same checksums.
        maven("https://maven-central.storage-download.googleapis.com/maven2/")
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        // Google's mirror of Maven Central first: Maven Central answers CI runners with HTTP 429
        // (Robolectric's nativeruntime jar, 2026-11-22/23). Same artifacts, same checksums.
        maven("https://maven-central.storage-download.googleapis.com/maven2/")
        mavenCentral()
    }
}

rootProject.name = "wse2_android"
include(":app")
