// Komari 分块下载：node download_file.js <remote> <local>
const { execFile } = require("child_process");
const { promisify } = require("util");
const fs = require("fs");
const exec = promisify(execFile);

async function run(cmd) {
  const { stdout } = await exec("node", ["<SSH_TOOLS_DIR>/kexec.js", "<KOMARI_NODE_ID>", cmd], { maxBuffer: 1 << 28 });
  const m = stdout.match(/=== exit: 0 ===\n([\s\S]*)$/);
  if (!m) throw new Error("exit marker missing:\n" + stdout.slice(0, 500));
  return m[1].replace(/\r/g, "");
}

async function main() {
  const remote = process.argv[2];
  const local = process.argv[3];
  const sizeStr = (await run("stat -c %s '" + remote + "'")).trim();
  const size = parseInt(sizeStr, 10);
  console.log("size", size);
  const CH = 400000; // base64 chars per chunk (300KB raw)
  const chunks = Math.ceil((size * 4 + 2) / 3 / CH);
  const parts = [];
  for (let i = 0; i < chunks; i++) {
    const skip = i * CH;
    const out = await run("base64 -w0 '" + remote + "' | tail -c +" + (skip + 1) + " | head -c " + CH);
    parts.push(out.trim());
    process.stdout.write("chunk " + (i + 1) + "/" + chunks + " ok\n");
  }
  fs.writeFileSync(local + ".b64", parts.join(""));
  const b64 = fs.readFileSync(local + ".b64", "utf8").replace(/\s+/g, "");
  fs.writeFileSync(local, Buffer.from(b64, "base64"));
  fs.unlinkSync(local + ".b64");
  const md5local = require("crypto").createHash("md5").update(fs.readFileSync(local)).digest("hex");
  const md5remote = (await run("md5sum '" + remote + "' | cut -d' ' -f1")).trim();
  console.log("md5", md5local, md5remote, md5local === md5remote ? "VERIFIED" : "MISMATCH");
}

main().catch(e => { console.error(e); process.exit(1); });
