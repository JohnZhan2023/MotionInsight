const cases = window.motionInsightComparisons || [];
const video = document.querySelector('#comparison-video');
const strip = document.querySelector('#thumbnail-strip');
const videoError = document.querySelector('#video-error');
const modelLabels = document.querySelector('#comparison-labels');
const modelNames = {
  wan: 'Wan',
  dpo: 'DPO baseline',
  ours: 'MotionInsight-DPO (Ours)',
};
const initialExampleIndex = Math.max(0, cases.findIndex(item => item.id === 'q008'));

function showExample(index) {
  if (!cases[index]) return;
  const wasPlaying = !video.paused;
  video.pause();
  const item = cases[index];
  const names = item.methods.map(method => modelNames[method] || method);
  modelLabels.replaceChildren(...names.map(name => {
    const label = document.createElement('div');
    const title = document.createElement('strong');
    title.textContent = name;
    label.append(title);
    return label;
  }));
  videoError.hidden = true;
  video.poster = item.poster;
  video.src = item.video;
  video.setAttribute('aria-label', `${item.titleEn}. Three outputs, from left to right ${names.join(', ')}.`);
  video.load();
  document.querySelector('#example-title').textContent = item.titleEn;
  document.querySelector('#example-count').textContent = `${index + 1} / ${cases.length}`;
  document.querySelector('#prompt-en').textContent = item.prompt;
  [...strip.children].forEach((button, buttonIndex) => {
    button.setAttribute('aria-pressed', String(buttonIndex === index));
  });
  const thumbnail = strip.children[index];
  if (thumbnail) {
    // Keep the selected thumbnail visible without moving the page vertically.
    const left = thumbnail.offsetLeft - strip.offsetLeft;
    if (left < strip.scrollLeft) strip.scrollLeft = left;
    else if (left + thumbnail.offsetWidth > strip.scrollLeft + strip.clientWidth) {
      strip.scrollLeft = left + thumbnail.offsetWidth - strip.clientWidth;
    }
  }
  if (wasPlaying) video.play().catch(() => {});
}

cases.forEach((item, index) => {
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'thumbnail';
  button.setAttribute('aria-label', `View example ${index + 1}: ${item.titleEn}`);
  button.setAttribute('aria-pressed', String(index === initialExampleIndex));
  const image = document.createElement('img');
  image.src = item.thumbnail;
  image.alt = '';
  image.width = 240;
  image.height = 138;
  image.loading = 'lazy';
  const label = document.createElement('span');
  label.textContent = item.titleEn;
  button.append(image, label);
  button.addEventListener('click', () => showExample(index));
  strip.append(button);
});
video.addEventListener('error', () => { videoError.hidden = false; });
video.addEventListener('loadedmetadata', () => {
  videoError.hidden = true;
});
document.querySelector('#retry-video').addEventListener('click', () => {
  videoError.hidden = true;
  video.load();
});
if (cases.length) showExample(initialExampleIndex);

const dialog = document.querySelector('#figure-dialog');
const figureButton = document.querySelector('#enlarge-figure');
figureButton.addEventListener('click', () => {
  dialog.showModal();
  document.body.classList.add('dialog-open');
});
document.querySelector('#close-figure').addEventListener('click', () => dialog.close());
dialog.addEventListener('click', event => {
  const bounds = dialog.getBoundingClientRect();
  if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) dialog.close();
});
dialog.addEventListener('close', () => {
  document.body.classList.remove('dialog-open');
  figureButton.focus();
});
let toastTimer;
const toast = document.querySelector('#toast');
function notify(message) {
  clearTimeout(toastTimer);
  toast.textContent = message;
  toast.hidden = false;
  toastTimer = setTimeout(() => { toast.hidden = true; }, 3500);
}
document.querySelector('#copy-citation').addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText(document.querySelector('#citation-text').textContent);
    notify('BibTeX copied');
  } catch {
    const range = document.createRange();
    range.selectNodeContents(document.querySelector('#citation-text'));
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    notify('Citation selected. Copy manually or download the .bib file.');
  }
});
