// Execute the generated gallery controls in a minimal DOM; verify all local links.
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const dir = path.resolve(process.argv[2]);
const page = fs.readFileSync(path.join(dir, 'index.html'), 'utf8');
const source = page.match(/<script>([\s\S]*?)<\/script>/)[1];
const elements = {};
let callback;
function element(id) {
  if (!elements[id]) {
    elements[id] = {
      options: [], selectedIndex: 0,
      add(option) { this.options.push(option); },
      get length() { return this.options.length; },
      get value() { return this.options[this.selectedIndex]?.value; },
    };
  }
  return elements[id];
}
vm.runInNewContext(source, {
  document: { getElementById: element },
  Option: function (text, value) { this.text = text; this.value = value; },
  setInterval(fn) { callback = fn; return 1; },
  clearInterval() { callback = null; },
}, { timeout: 5000 });
let count = 0;
for (let s = 0; s < elements.scene.length; ++s) {
  elements.scene.selectedIndex = s;
  for (let c = 0; c < elements.candidate.length; ++c) {
    elements.candidate.selectedIndex = c;
    elements.candidate.onchange();
    for (const id of ['ref', 'test', 'diff', 'heat', 'sheet', 'raw']) {
      const target = elements[id].src || elements[id].href;
      if (!target || !fs.existsSync(path.join(dir, target))) throw new Error(`missing ${target}`);
    }
    if (!elements.info.textContent.includes('P95')) throw new Error('metrics missing');
    elements.blink.onclick(); callback(); callback(); elements.blink.onclick();
    count++;
  }
}
elements.prev.onclick(); elements.next.onclick();
console.log(`Gallery controls, blink and asset links verified for ${count} scene/candidate pairs.`);
