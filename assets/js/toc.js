// Sidebar table of contents: highlight the section currently being read.
// The active heading is the last one whose top has scrolled above a line
// a third of the way down the viewport. Without JS, the TOC is plain links.

const toc = document.querySelector(".toc");

if (toc) {
  const links = new Map();
  for (const a of toc.querySelectorAll('a[href^="#"]')) {
    const id = decodeURIComponent(a.hash.slice(1));
    const heading = id && document.getElementById(id);
    if (heading) links.set(heading, a);
  }
  const headings = [...links.keys()];
  let active = null;
  let queued = false;

  function update() {
    queued = false;
    const line = window.innerHeight / 3;
    let current = null;
    for (const h of headings) {
      if (h.getBoundingClientRect().top <= line) current = h;
      else break;
    }
    const link = current ? links.get(current) : null;
    if (link === active) return;
    active?.classList.remove("active");
    active?.removeAttribute("aria-current");
    active = link;
    if (!active) return;
    active.classList.add("active");
    active.setAttribute("aria-current", "location");

    // Keep the active entry in view when the TOC itself scrolls.
    const t = toc.getBoundingClientRect();
    const r = active.getBoundingClientRect();
    if (r.top < t.top || r.bottom > t.bottom) {
      toc.scrollTop += r.top - t.top - t.height / 3;
    }
  }

  function schedule() {
    if (!queued) {
      queued = true;
      requestAnimationFrame(update);
    }
  }

  window.addEventListener("scroll", schedule, { passive: true });
  window.addEventListener("resize", schedule);
  update();
}
