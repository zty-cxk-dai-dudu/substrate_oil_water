'use strict';

const group = document.querySelector('#source-group');
const query = document.querySelector('#source-query');
const list = document.querySelector('#source-list');
const count = document.querySelector('#source-count');
const entries = [...list.querySelectorAll('li')].map(element => {
  const path = element.querySelector('a').textContent;
  return {element, path, directory: path.slice(0, path.lastIndexOf('/'))};
});

function filterFiles() {
  const term = query.value.trim().toLowerCase();
  let visible = 0;
  for (const entry of entries) {
    const match = (group.value === 'all' || group.value === entry.directory)
      && entry.path.toLowerCase().includes(term);
    entry.element.hidden = !match;
    entry.element.style.display = match ? '' : 'none';
    visible += Number(match);
  }
  count.textContent = `${visible} ${visible === 1 ? 'file' : 'files'} shown`;
}

group.addEventListener('change', filterFiles);
query.addEventListener('input', filterFiles);
