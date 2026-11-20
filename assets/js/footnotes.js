// Footnote popovers: hovering (or focusing, or tapping on touch devices) a
// footnote reference shows the footnote's content in a small popover, so the
// reader doesn't lose their place. Works on Goldmark's native footnote markup:
//   <sup><a href="#fn:1" class="footnote-ref" role="doc-noteref">1</a></sup>
//   <div class="footnotes" role="doc-endnotes"><ol><li id="fn:1">…</li></ol></div>
// Without JS, the references remain ordinary in-page links.

const SHOW_DELAY = 100;
const HIDE_DELAY = 200;
const GAP = 8; // px between reference and popover
const MARGIN = 12; // px kept clear of the viewport edges

let popover = null;
let activeRef = null;
let showTimer = 0;
let hideTimer = 0;
let lastPointerType = "mouse";
let openAtPointerDown = null; // which reference's popover was open when the tap began

function getPopover() {
  if (popover) return popover;
  popover = document.createElement("div");
  popover.className = "footnote-popover";
  popover.setAttribute("role", "tooltip");
  popover.id = "footnote-popover";
  popover.hidden = true;
  popover.addEventListener("pointerenter", (e) => {
    if (e.pointerType === "mouse") clearTimeout(hideTimer);
  });
  popover.addEventListener("pointerleave", (e) => {
    if (e.pointerType === "mouse") scheduleHide();
  });
  document.body.appendChild(popover);
  return popover;
}

function footnoteFor(ref) {
  const id = decodeURIComponent(ref.hash.slice(1));
  return id ? document.getElementById(id) : null;
}

function position(ref) {
  const pop = getPopover();
  const r = ref.getBoundingClientRect();
  const vw = document.documentElement.clientWidth;
  const vh = window.innerHeight;

  // Measure at the final width before placing.
  pop.style.left = "0px";
  pop.style.top = "0px";
  const p = pop.getBoundingClientRect();

  let left = r.left + r.width / 2 - p.width / 2;
  left = Math.max(MARGIN, Math.min(left, vw - p.width - MARGIN));

  // Prefer below the reference; flip above if it doesn't fit and there's more room there.
  let top = r.bottom + GAP;
  const below = vh - r.bottom - GAP - MARGIN;
  const above = r.top - GAP - MARGIN;
  if (p.height > below && above > below) {
    top = r.top - GAP - Math.min(p.height, above);
  }

  pop.style.left = `${left + window.scrollX}px`;
  pop.style.top = `${top + window.scrollY}px`;
}

function show(ref) {
  const note = footnoteFor(ref);
  if (!note) return;
  clearTimeout(hideTimer);
  if (activeRef === ref && !getPopover().hidden) return;

  const pop = getPopover();
  const content = note.cloneNode(true);
  content.querySelectorAll(".footnote-backref").forEach((a) => a.remove());
  // Drop the whitespace (including the &nbsp;) Goldmark leaves before the
  // backlink; when a note ends in a list it would render as a blank line.
  let last = content.lastChild;
  while (last && last.nodeType === Node.TEXT_NODE && !last.textContent.trim()) {
    last.remove();
    last = content.lastChild;
  }
  // Avoid duplicate ids from cloned content.
  content.querySelectorAll("[id]").forEach((el) => el.removeAttribute("id"));
  pop.replaceChildren(...content.childNodes);
  pop.hidden = false;

  if (activeRef) activeRef.removeAttribute("aria-describedby");
  activeRef = ref;
  ref.setAttribute("aria-describedby", pop.id);
  position(ref);
}

function hide() {
  clearTimeout(showTimer);
  clearTimeout(hideTimer);
  if (popover) popover.hidden = true;
  if (activeRef) activeRef.removeAttribute("aria-describedby");
  activeRef = null;
}

function scheduleHide() {
  clearTimeout(showTimer);
  clearTimeout(hideTimer);
  hideTimer = setTimeout(hide, HIDE_DELAY);
}

function init() {
  const refs = document.querySelectorAll("a.footnote-ref");
  if (!refs.length) return;

  document.addEventListener("pointerdown", (e) => {
    lastPointerType = e.pointerType;
    openAtPointerDown = popover && !popover.hidden ? activeRef : null;
    // Tap outside the popover and references dismisses it.
    if (
      popover &&
      !popover.hidden &&
      !popover.contains(e.target) &&
      !e.target.closest("a.footnote-ref")
    ) {
      hide();
    }
  });

  refs.forEach((ref) => {
    // Hover only for real mouse pointers; touch emulates mouse events
    // around taps, which would otherwise close the popover immediately.
    ref.addEventListener("pointerenter", (e) => {
      if (e.pointerType !== "mouse") return;
      clearTimeout(hideTimer);
      clearTimeout(showTimer);
      showTimer = setTimeout(() => show(ref), SHOW_DELAY);
    });
    ref.addEventListener("pointerleave", (e) => {
      if (e.pointerType === "mouse") scheduleHide();
    });
    ref.addEventListener("focus", () => show(ref));
    ref.addEventListener("blur", (e) => {
      if (!popover || !popover.contains(e.relatedTarget)) scheduleHide();
    });
    ref.addEventListener("click", (e) => {
      // On touch, the first tap opens the popover instead of jumping;
      // tapping the same reference again follows the link as usual.
      // (Check the state from pointerdown: the tap's focus event has
      // already opened the popover by the time click fires.)
      if (lastPointerType === "touch" && openAtPointerDown !== ref) {
        e.preventDefault();
        show(ref);
      } else {
        hide();
      }
    });
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && activeRef) {
      const ref = activeRef;
      hide();
      ref.focus();
    }
  });

  window.addEventListener("resize", () => activeRef && position(activeRef));
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
