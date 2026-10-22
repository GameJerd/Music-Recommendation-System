// Slopify: a small single-page client for the recommender API (no build step).
"use strict";

const root = document.documentElement;
const view = document.getElementById("view");
const userInput = document.getElementById("user");
const recentList = document.getElementById("recent");
const toastEl = document.getElementById("toast");

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const clamp01 = (x) => Math.min(1, Math.max(0, Number.isFinite(x) ? x : 0.5));
const icon = (id) => `<svg aria-hidden="true"><use href="#${id}"/></svg>`;
const store = {
  get: (k) => { try { return localStorage.getItem(k); } catch { return null; } },
  set: (k, v) => { try { localStorage.setItem(k, v); } catch { /* storage blocked */ } },
};

/* ---------- user ---------- */
let user = store.get("slopify.user") || "guest-" + Math.random().toString(36).slice(2, 6);
store.set("slopify.user", user);
userInput.value = user;
userInput.addEventListener("change", () => {
  user = userInput.value.trim() || user;
  userInput.value = user;
  store.set("slopify.user", user);
  refreshRecent();
  render();
});

/* ---------- api ---------- */
async function api(path, options) {
  const res = await fetch(path, options);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail ?? detail; } catch { /* not json */ }
    const err = new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    err.status = res.status;
    throw err;
  }
  return res.json();
}
const u = () => encodeURIComponent(user);
const getHistory = () => api(`/users/${u()}/history?limit=200`);
const logEvent = (trackId, eventType) => api("/events", {
  method: "POST", headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ user_id: user, track_id: trackId, event_type: eventType }),
});

/* ---------- cover colours ----------
   The dataset has no artwork. Each song gets a colour from a small curated palette, picked by its
   sound: mood (valence) and energy choose the colour, very acoustic songs get the earthy tones,
   danceability nudges the hue a little. Songs that sound alike get the same family of colour. */
const PALETTE = {
  slate: [212, 18, 46], blue: [214, 62, 52], indigo: [234, 48, 58], teal: [186, 54, 42],
  violet: [258, 36, 58], sage: [148, 24, 46], rose: [346, 62, 60], coral: [12, 72, 62],
  sand: [34, 36, 60], amber: [38, 80, 55], olive: [74, 24, 44],
};
const MOOD_ENERGY = [            // rows: sad -> happy, columns: calm -> energetic
  ["slate", "blue", "indigo"],
  ["teal", "blue", "violet"],
  ["sage", "rose", "coral"],
  ["sand", "amber", "coral"],
];
const ACOUSTIC = ["slate", "sage", "olive", "sand"];

function colours(a = {}) {
  const v = clamp01(a.valence), e = clamp01(a.energy), d = clamp01(a.danceability), ac = clamp01(a.acousticness);
  const mood = Math.min(3, Math.floor(v * 4));
  const name = ac > 0.7 ? ACOUSTIC[mood] : MOOD_ENERGY[mood][Math.min(2, Math.floor(e * 3))];
  const [h0, s, l] = PALETTE[name];
  const h = h0 + (d - 0.5) * 14;
  const hsl = (dh, sm, lm) => `hsl(${(h + dh).toFixed(0)} ${Math.min(95, s * sm).toFixed(0)}% ${Math.min(85, l * lm).toFixed(0)}%)`;
  return {
    c1: hsl(0, 1, 1), c2: hsl(20, 0.95, 0.55), c3: hsl(-16, 1.15, 1.35), cb: hsl(8, 0.85, 0.42),
  };
}
const vars = (c) => `--c1:${c.c1};--c2:${c.c2};--c3:${c.c3};--cb:${c.cb}`;
const cover = (t) => `<span class="cover" style="${vars(colours(t.audio))}" aria-hidden="true"></span>`;

function setAmbient(t) {
  const c = t ? colours(t.audio) : null;
  root.style.setProperty("--amb1", c ? c.c1 : "transparent");
  root.style.setProperty("--amb2", c ? c.c2 : "transparent");
}

/* ---------- templates ---------- */
const artistOf = (t) => t.artist_name ?? (t.artists || []).join(", ");
const trackHref = (t) => `#/track/${encodeURIComponent(t.track_id)}`;
const card = (t) => `
  <a class="card" href="${trackHref(t)}">
    ${cover(t)}
    <b>${esc(t.track_name)}</b>
    <span>${esc(artistOf(t))}</span>
  </a>`;
const shelf = (items) => `<div class="shelf">${items.map(card).join("")}</div>`;
const shelfHead = (title, sub = "") =>
  `<div class="shelf-head"><h2>${esc(title)}</h2>${sub ? `<p>${esc(sub)}</p>` : ""}</div>`;
const row = (t, end = t.year) => `
  <a class="row" href="${trackHref(t)}">
    ${cover(t)}
    <span class="t"><b>${esc(t.track_name)}</b><span>${esc(artistOf(t))}</span></span>
    <span class="end">${esc(end)}</span>
  </a>`;
const searchForm = (q = "") => `
  <form class="searchbar" role="search" data-search>
    ${icon("i-search")}
    <input name="q" type="search" value="${esc(q)}" placeholder="Songs or artists"
           aria-label="Search songs or artists" autocomplete="off">
  </form>`;
const skeletonShelf = (n = 12) => `<div class="shelf">${
  Array.from({ length: n }, () => `<div class="sk-card"><div class="sk cover-sk"></div><div class="sk sk-line" style="width:80%"></div><div class="sk sk-line" style="width:55%"></div></div>`).join("")}</div>`;
const SKELETON = {
  home: `<div class="page"><div class="sk" style="height:34px;width:220px;margin:4px 0 26px"></div>${skeletonShelf()}</div>`,
  track: `<div class="sk-hero"></div><div class="page">${skeletonShelf(6)}</div>`,
};

/* ---------- toast ---------- */
let toastTimer;
function toast(msg) {
  toastEl.textContent = msg;
  toastEl.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toastEl.classList.remove("show"), 2600);
}

/* ---------- sidebar: recently played ---------- */
async function refreshRecent() {
  try {
    const seen = new Set();
    const items = (await getHistory())
      .filter((h) => h.event_type !== "skip" && !seen.has(h.track.track_id) && seen.add(h.track.track_id))
      .slice(0, 15);
    recentList.innerHTML = items.length
      ? items.map((h) => `<li>${row(h.track, "")}</li>`).join("")
      : `<li class="empty">Songs you play or like show up here.</li>`;
  } catch {
    recentList.innerHTML = `<li class="empty">Couldn't load your history.</li>`;
  }
}

/* ---------- views ---------- */
async function homeView(isCurrent) {
  view.innerHTML = SKELETON.home;
  const listened = (await getHistory()).filter((h) => h.event_type !== "skip");
  if (!isCurrent()) return;
  if (!listened.length) {
    const popular = await api(`/recommend/user/${u()}?k=12`);
    if (!isCurrent()) return;
    setAmbient(popular.items[0]);
    view.innerHTML = `
      <div class="page">
        <section class="welcome">
          <h1>Find music by how it sounds.</h1>
          <p>Search for a song you love. Slopify suggests others with a similar energy, mood, tempo and era,
             from 170,000 tracks released between 1921 and 2020.</p>
          ${searchForm()}
        </section>
        <section>
          ${shelfHead("Popular in the catalog", "Until you play or like something")}
          ${shelf(popular.items)}
        </section>
      </div>`;
    return;
  }
  const last = listened[0].track;
  const [mix, more] = await Promise.all([
    api(`/recommend/user/${u()}?k=12`),
    api(`/recommend/similar/${encodeURIComponent(last.track_id)}?k=12`),
  ]);
  if (!isCurrent()) return;
  setAmbient(mix.items[0] || last);
  const n = new Set(listened.map((h) => h.track.track_id)).size;
  view.innerHTML = `
    <div class="page">
      <section>
        <h1 class="page-title">Your mix</h1>
        <p class="page-sub">Built from the sound of ${n === 1 ? "the song" : `the ${n} songs`} you've played or liked.</p>
        ${shelf(mix.items)}
      </section>
      <section>
        ${shelfHead(`More like ${last.track_name}`)}
        ${shelf(more.items)}
      </section>
    </div>`;
}

let searchTimer;
async function searchView(isCurrent, params) {
  setAmbient(null);
  const q = params.get("q") || "";
  view.innerHTML = `<div class="page"><h1 class="page-title" style="margin-bottom:16px">Search</h1>${searchForm(q)}<div class="results" aria-live="polite"></div></div>`;
  const input = view.querySelector("input[name=q]");
  const results = view.querySelector(".results");
  input.focus();
  input.setSelectionRange(q.length, q.length);

  const run = async (term) => {
    term = term.trim();
    history.replaceState(null, "", term ? `#/search?q=${encodeURIComponent(term)}` : "#/search");
    if (!term) {
      results.innerHTML = `<p class="note">Find a song by its title or artist.</p>`;
      return;
    }
    const found = await api(`/tracks/search?q=${encodeURIComponent(term)}&limit=30`);
    if (!isCurrent() || input.value.trim() !== term) return;
    results.innerHTML = found.length
      ? found.map((t) => row(t)).join("")
      : `<p class="note">No songs match “${esc(term)}”. Try an artist name or fewer words.</p>`;
  };
  input.addEventListener("input", () => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => run(input.value).catch(showError), 200);
  });
  await run(q);
}

const SOUND = [
  ["energy", "Energy"], ["danceability", "Danceability"], ["valence", "Positivity"],
  ["acousticness", "Acoustic"], ["instrumentalness", "Instrumental"], ["speechiness", "Spoken word"],
  ["liveness", "Live feel"],
];

async function trackView(isCurrent, _params, id) {
  view.innerHTML = SKELETON.track;
  const [t, sim, hist] = await Promise.all([
    api(`/tracks/${encodeURIComponent(id)}`),
    api(`/recommend/similar/${encodeURIComponent(id)}?k=20`),
    getHistory(),
  ]);
  if (!isCurrent()) return;
  setAmbient(t);
  const mine = hist.filter((h) => h.track.track_id === id);
  const liked = mine.some((h) => h.event_type === "like");
  const hidden = mine.some((h) => h.event_type === "skip");
  const a = t.audio;
  // "Bohemian Rhapsody - Remastered 2011" -> title + version line, as music apps show it
  const cut = t.track_name.indexOf(" - ");
  const title = cut > 0 ? t.track_name.slice(0, cut) : t.track_name;
  const version = cut > 0 ? t.track_name.slice(cut + 3) : "";
  const bar = (v) => `<span class="bar"><i style="--v:${clamp01(v).toFixed(3)}"></i></span>`;
  const sound = SOUND.map(([k, label]) =>
    `<div><dt>${label}</dt><dd>${Math.round(clamp01(a[k]) * 100)}%</dd>${bar(a[k])}</div>`).join("")
    + `<div><dt>Tempo</dt><dd>${Math.round(a.tempo)} BPM</dd>${bar((a.tempo - 60) / 140)}</div>`
    + `<div><dt>Loudness</dt><dd>${a.loudness.toFixed(1)} dB</dd>${bar((a.loudness + 60) / 60)}</div>`;

  view.innerHTML = `
    <header class="hero" style="${vars(colours(a))}">
      ${cover(t)}
      <div class="hero-text">
        <h1>${esc(title)}</h1>
        ${version ? `<p class="version">${esc(version)}</p>` : ""}
        <div class="meta">
          ${t.artists.map((x) => `<a href="#/search?q=${encodeURIComponent(x)}">${esc(x)}</a>`).join("")}
          <span>${t.year}</span>
        </div>
      </div>
    </header>
    <div class="actions">
      <a class="listen" href="https://open.spotify.com/track/${encodeURIComponent(t.track_id)}"
         target="_blank" rel="noopener" data-act="play">${icon("i-play")}Listen on Spotify</a>
      <button class="icon-btn" data-act="like" aria-pressed="${liked}" aria-label="${liked ? "Liked" : "Like"}"
              ${liked ? "disabled" : ""}>${icon(liked ? "i-heart-fill" : "i-heart")}</button>
      <button class="pill-btn" data-act="skip" ${hidden ? "disabled" : ""}>${icon("i-hide")}${hidden ? "Hidden from your mix" : "Not for me"}</button>
    </div>
    <div class="track-body">
      <section aria-labelledby="sim-h">
        <h2 id="sim-h">Similar songs</h2>
        <ol class="tracklist">
          ${sim.items.map((s, i) => `<li><a class="row" href="${trackHref(s)}">
              <span class="n">${i + 1}</span>${cover(s)}
              <span class="t"><b>${esc(s.track_name)}</b><span>${esc(s.artist_name)}</span></span>
              <span class="yr">${s.year}</span>
              <span class="match" title="How close it is in sound, genre and era">${Math.round(s.similarity * 100)}%</span>
            </a></li>`).join("")}
        </ol>
      </section>
      <aside aria-labelledby="sound-h">
        <h2 id="sound-h">How it sounds</h2>
        <dl class="sound">${sound}</dl>
        ${t.genres.length ? `<ul class="genres" aria-label="Artist genres">${t.genres.slice(0, 8).map((g) => `<li>${esc(g)}</li>`).join("")}</ul>` : ""}
      </aside>
    </div>`;
  view.dataset.track = id;
}

/* ---------- actions on the song page ---------- */
view.addEventListener("click", async (ev) => {
  const el = ev.target.closest("[data-act]");
  if (!el || !view.dataset.track) return;
  const id = view.dataset.track, act = el.dataset.act;
  try {
    if (act === "play") {
      await logEvent(id, "play");
      toast("Counted as played. Your mix now includes it.");
    } else if (act === "like") {
      await logEvent(id, "like");
      el.setAttribute("aria-pressed", "true");
      el.setAttribute("aria-label", "Liked");
      el.disabled = true;
      el.innerHTML = icon("i-heart-fill");
      toast("Liked. Your mix will lean toward songs like this.");
    } else if (act === "skip") {
      await logEvent(id, "skip");
      el.disabled = true;
      el.innerHTML = `${icon("i-hide")}Hidden from your mix`;
      toast("Got it. This song won't be suggested again.");
    }
    refreshRecent();
  } catch (err) {
    toast(`Couldn't save that: ${err.message}`);
  }
});

view.addEventListener("submit", (ev) => {
  if (!ev.target.matches("[data-search]")) return;
  ev.preventDefault();
  const q = new FormData(ev.target).get("q").trim();
  location.hash = q ? `#/search?q=${encodeURIComponent(q)}` : "#/search";
});

/* ---------- router ---------- */
const ROUTES = { "": homeView, search: searchView, track: trackView };
let renderSeq = 0;

function showError(err) {
  view.innerHTML = err.status === 404
    ? `<div class="error"><b>That song isn't in the catalog.</b><a href="#/search">Search for another one</a></div>`
    : `<div class="error"><b>Slopify can't reach the recommender.</b>${esc(err.message)}. Check that the server is running, then reload the page.</div>`;
}

async function render() {
  const seq = ++renderSeq;
  const [path, query = ""] = location.hash.replace(/^#\/?/, "").split("?");
  const [name, arg] = path.split("/");
  const viewFn = ROUTES[name] || homeView;
  document.querySelectorAll("[data-nav]").forEach((a) => {
    const on = a.dataset.nav === (name === "search" ? "search" : name === "" ? "home" : "");
    on ? a.setAttribute("aria-current", "page") : a.removeAttribute("aria-current");
  });
  delete view.dataset.track;
  view.scrollTop = 0;
  try {
    await viewFn(() => seq === renderSeq, new URLSearchParams(query), arg && decodeURIComponent(arg));
  } catch (err) {
    if (seq === renderSeq) showError(err);
  }
}

window.addEventListener("hashchange", render);
refreshRecent();
render();
