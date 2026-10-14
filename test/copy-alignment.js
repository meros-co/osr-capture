// Windows-only regression for StreamCopyFromWC. Build with npx node-gyp rebuild,
// then run node test/copy-alignment.js. Each case runs in a separate process so
// an access violation reports a failure rather than terminating the test runner.
const path = require("path")
const { spawnSync } = require("child_process")
if (process.platform !== "win32") {
    console.log("SKIP: Windows-only copy alignment test")
    process.exit(0)
}
const executable = path.join(__dirname, "..", "build", "Release", "copy_alignment.exe")
const cases = []
const lengths = [0, 1, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 255, 256, 257, 4396, 7680, 15360]
for (let offset = 0; offset < 16; offset++) {
    for (const bytes of lengths) cases.push([offset, bytes, bytes, 1])
}
cases.push([0, 4396, 4396, 30], [0, 4396, 4400, 30], [0, 7680, 7680, 30], [0, 15360, 15360, 30])
for (let bytes = 0; bytes <= 65; bytes++) cases.push(["--guard", bytes])
let failures = 0
for (const args of cases) {
    const result = spawnSync(executable, args.map(String), { timeout: 10000, windowsHide: true, encoding: "utf8" })
    if (result.error) {
        console.error(result.error.message)
        process.exit(1)
    }
    if (result.status !== 0) {
        failures++
        console.error(`FAIL ${args.join(" ")}: exit=${result.status} signal=${result.signal}`)
    }
}
console.log(`${cases.length - failures}/${cases.length} copy alignment cases passed`)
process.exitCode = failures ? 1 : 0
