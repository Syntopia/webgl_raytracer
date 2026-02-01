export function createLogger(statusEl) {
  const lines = [];
  const maxLines = 6;

  function push(line) {
    lines.push(line);
    while (lines.length > maxLines) {
      lines.shift();
    }
    if (statusEl) {
      statusEl.textContent = lines.join("\n");
    }
    console.log(line);
  }

  return {
    info: (msg) => push(`INFO  ${msg}`),
    warn: (msg) => push(`WARN  ${msg}`),
    error: (msg) => push(`ERROR ${msg}`)
  };
}
