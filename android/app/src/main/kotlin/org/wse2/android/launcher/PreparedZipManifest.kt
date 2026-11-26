package org.wse2.android.launcher

import org.json.JSONObject
import java.io.File
import java.io.InputStream
import java.security.MessageDigest
import java.util.Locale
import java.util.zip.ZipInputStream

data class PendingPreparedZip(
    val documentId: String,
    val fileName: String,
    val sizeBytes: Long,
    val lastModifiedMs: Long,
    val stamp: String
)

data class PreparedZipManifest(
    val formatVersion: Int,
    val createdUtc: String,
    val translatorFingerprint: String,
    val hashes: Map<String, String>
)

data class PreparedZipImportResult(
    val ok: Boolean,
    val partsImported: Int = 0,
    val texturesImported: Int = 0,
    val error: String = ""
)

/**
 * Verifies and extracts PC-prepared `wse2-prepared*.zip` archives (`docs/PLATFORM_PLAN.md` "P14.6").
 *
 * Because the archive contains executable `.so` code, every extracted `.so`, `.key`, and `.etc2`
 * file MUST match its SHA-256 hash in `manifest.json` before being committed to app storage.
 */
object PreparedZipVerifier {
    const val MANIFEST_NAME = "manifest.json"
    private val PREPARED_PART = Regex("""(exe|lua5164)_w\d+\.(so|key)""")
    private val PREPARED_TEXTURE = Regex("""[0-9a-f]{16}\.etc2""")

    fun parseManifest(rawJson: String): PreparedZipManifest? {
        return try {
            val obj = JSONObject(rawJson)
            val fmt = obj.optInt("format", 0)
            if (fmt != 1) return null
            val filesObj = obj.optJSONObject("files") ?: return null
            val hashes = LinkedHashMap<String, String>()
            val it = filesObj.keys()
            while (it.hasNext()) {
                val key = it.next()
                val hex = filesObj.optString(key, "").trim().lowercase(Locale.US)
                if (hex.length != 64 || !hex.all { c -> c in '0'..'9' || c in 'a'..'f' }) {
                    return null
                }
                hashes[key.substringAfterLast('/').substringAfterLast('\\')] = hex
            }
            if (hashes.isEmpty()) return null
            PreparedZipManifest(
                formatVersion = fmt,
                createdUtc = obj.optString("created_utc", ""),
                translatorFingerprint = obj.optString("translator_fingerprint", ""),
                hashes = hashes
            )
        } catch (_: Throwable) {
            null
        }
    }

    fun sha256Hex(bytes: ByteArray): String {
        val digest = MessageDigest.getInstance("SHA-256").digest(bytes)
        val sb = StringBuilder(64)
        for (b in digest) {
            sb.append(String.format(Locale.US, "%02x", b))
        }
        return sb.toString()
    }

    /**
     * Streams, SHA-256 verifies against `manifest.json`, and atomically commits all valid parts.
     * Refuses archives without a valid `manifest.json` or with any mismatched hash (leaving no
     * half-imported `.so` files behind).
     */
    fun verifyAndExtract(
        zipStream: InputStream,
        soDir: File,
        texDir: File,
        isCancelled: () -> Boolean = { false },
        onProgress: (parts: Int, textures: Int) -> Unit = { _, _ -> }
    ): PreparedZipImportResult {
        soDir.mkdirs()
        texDir.mkdirs()
        val stagingDir = File(soDir.parentFile ?: soDir, "zip_staging_${System.nanoTime()}").apply { mkdirs() }
        var manifest: PreparedZipManifest? = null
        val extractedHashes = HashMap<String, String>()
        val stagedParts = HashMap<String, File>()
        val stagedKeys = HashMap<String, ByteArray>()
        val stagedTextures = HashMap<String, File>()

        try {
            ZipInputStream(zipStream.buffered(1 shl 16)).use { zip ->
                val buf = ByteArray(1 shl 16)
                while (true) {
                    if (isCancelled()) {
                        return PreparedZipImportResult(ok = false, error = "Import cancelled.")
                    }
                    val entry = zip.nextEntry ?: break
                    if (entry.isDirectory) continue
                    val name = entry.name.substringAfterLast('/').substringAfterLast('\\')
                    if (name == MANIFEST_NAME) {
                        val raw = zip.readBytes().toString(Charsets.UTF_8)
                        manifest = parseManifest(raw)
                            ?: return PreparedZipImportResult(ok = false, error = "Invalid manifest.json in PC zip.")
                        continue
                    }
                    val isPart = PREPARED_PART.matches(name)
                    val isTex = PREPARED_TEXTURE.matches(name)
                    if (!isPart && !isTex) continue

                    val md = MessageDigest.getInstance("SHA-256")
                    if (isPart && name.endsWith(".key")) {
                        val keyBytes = zip.readBytes()
                        md.update(keyBytes)
                        extractedHashes[name] = digestHex(md.digest())
                        stagedKeys[name.substringBeforeLast('.')] = keyBytes
                    } else {
                        val staged = File(stagingDir, name)
                        staged.outputStream().use { out ->
                            while (true) {
                                val r = zip.read(buf)
                                if (r <= 0) break
                                md.update(buf, 0, r)
                                out.write(buf, 0, r)
                            }
                        }
                        extractedHashes[name] = digestHex(md.digest())
                        if (isTex) {
                            stagedTextures[name] = staged
                        } else {
                            stagedParts[name.substringBeforeLast('.')] = staged
                        }
                    }
                }
            }

            val parsedManifest = manifest
                ?: return PreparedZipImportResult(
                    ok = false,
                    error = "Refused PC zip: missing manifest.json SHA-256 hash list. Re-run tools/prepare_for_phone.ps1."
                )

            // Verify every extracted file against the manifest before committing anything.
            for ((fileName, actualHash) in extractedHashes) {
                val expectedHash = parsedManifest.hashes[fileName]
                    ?: return PreparedZipImportResult(
                        ok = false,
                        error = "Refused PC zip: '$fileName' is not listed in manifest.json."
                    )
                if (!expectedHash.equals(actualHash, ignoreCase = true)) {
                    return PreparedZipImportResult(
                        ok = false,
                        error = "Refused PC zip: SHA-256 mismatch for '$fileName' (file may be corrupted)."
                    )
                }
            }
            for ((expectedName, _) in parsedManifest.hashes) {
                if (!extractedHashes.containsKey(expectedName)) {
                    return PreparedZipImportResult(
                        ok = false,
                        error = "Refused PC zip: manifest lists '$expectedName' which is missing from the zip."
                    )
                }
            }

            var partsCommitted = 0
            var texturesCommitted = 0
            for ((part, stagedSo) in stagedParts) {
                val keyBytes = stagedKeys[part] ?: continue
                val destKey = File(soDir, "$part.key")
                val destSo = File(soDir, "$part.so")
                if (destSo.isFile && destKey.isFile && destKey.readBytes().contentEquals(keyBytes)) {
                    continue
                }
                destKey.delete()
                destSo.delete()
                val keyTmp = File(soDir, "$part.key.import")
                keyTmp.writeBytes(keyBytes)
                val movedSo = stagedSo.renameTo(destSo) || run { stagedSo.copyTo(destSo, overwrite = true); true }
                val movedKey = keyTmp.renameTo(destKey)
                if (!movedSo || !movedKey) {
                    keyTmp.delete()
                    destSo.delete()
                    return PreparedZipImportResult(ok = false, error = "Could not install verified part $part.")
                }
                ++partsCommitted
                if (partsCommitted % 10 == 0) onProgress(partsCommitted, texturesCommitted)
            }
            for ((texName, stagedTex) in stagedTextures) {
                val destTex = File(texDir, texName)
                if (!destTex.isFile) {
                    if (stagedTex.renameTo(destTex) || run { stagedTex.copyTo(destTex, overwrite = true); true }) {
                        ++texturesCommitted
                    }
                }
            }
            onProgress(partsCommitted, texturesCommitted)
            return PreparedZipImportResult(
                ok = true,
                partsImported = partsCommitted,
                texturesImported = texturesCommitted
            )
        } catch (t: Throwable) {
            return PreparedZipImportResult(ok = false, error = t.message ?: t.javaClass.simpleName)
        } finally {
            stagingDir.deleteRecursively()
        }
    }

    private fun digestHex(digest: ByteArray): String {
        val sb = StringBuilder(64)
        for (b in digest) sb.append(String.format(Locale.US, "%02x", b))
        return sb.toString()
    }
}
