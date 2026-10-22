// Vibify: a small single-page client for the recommender API (no build step).
"use strict";

const view = document.getElementById("view");
const userInput = document.getElementById("user");
const recentList = document.getElementById("recent");
const toastEl = document.getElementById("toast");

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const clamp01 = (x) => Math.min(1, Math.max(0, Number.isFinite(x) ? x : 0.5));
const icon = (id) => `<svg aria-hidden="true"><use href="#${id}"/></svg>`;

/* ---------- user ---------- */
const USER_KEY = "vibify.user";
function loadUser() {
  try {
    const saved = localStorage.getItem(USER_KEY);
    if (saved) return saved;
  } catch { /* storage blocked: fall through */ }
  return "guest-" + Math.random().toString(36).slice(2, 6);
}
let user = loadUser();
userInput.value = user;
userInput.addEventListener("change", () => {
  user = userInput.value.trim() || user;
  userInput.value = user;
  try { localStorage.setItem(USER_KEY, user); } catch { /* per-session only */ }
  refreshRecent();
  render();
});
try { localStorage.setItem(USER_KEY, user); } catch { /* ignore */ }

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

/* ---------- generated covers ----------
   No artwork exists in the dataset, so each cover is drawn from the song's audio features:
   hue = mood (sad = deep blue, through violet and magenta, to happy = amber; never green),
   disc size = energy, disc position = danceability, groove spacing = tempo, diagonal stripes = very
   danceable, grain = acoustic, disc colour shift = instrumental. Songs that sound alike look alike. */
function coverArt(a = {}) {
  const v = clamp01(a.valence), e = clamp01(a.energy), d = clamp01(a.danceability);
  const ac = clamp01(a.acousticness), ins = clamp01(a.instrumentalness);
  const tempo = clamp01((a.tempo - 60) / 140);
  const hue = (225 + v * 160) % 360;                 // 225 blue -> 300 magenta -> 25 amber
  const sat = (38 + e * 50) * (1 - ac * 0.35);
  const lit = 22 + e * 14;
  const discHue = (hue + 25 + ins * 60) % 360;
  const cx = (28 + d * 44).toFixed(1), cy = (64 - e * 22).toFixed(1);
  const r = (18 + e * 22).toFixed(1);                 // disc radius, % of the cover
  const groove = (2.6 - tempo * 1.6).toFixed(2);      // faster = tighter grooves
  const disc = `hsl(${discHue} ${Math.min(95, sat + 22)}% ${Math.min(70, lit + 30)}%)`;
  const layers = [
    `repeating-radial-gradient(circle at ${cx}% ${cy}%, rgb(0 0 0 / .16) 0 .35%, transparent .35% ${groove}%)`,
    `radial-gradient(circle at ${cx}% ${cy}%, ${disc} 0 ${r}%, transparent calc(${r}% + .6%))`,
  ];
  if (d > 0.6) layers.push(`repeating-linear-gradient(120deg, rgb(255 255 255 / .06) 0 2px, transparent 2px ${(11 - d * 6).toFixed(1)}px)`);
  layers.push(`linear-gradient(160deg, hsl(${hue} ${sat}% ${lit}%), hsl(${(hue + 18) % 360} ${sat}% ${(lit * 0.42).toFixed(1)}%))`);
  return {
    style: `background:${layers.join(",")};--grain:${(0.12 + ac * 0.5).toFixed(2)}`,
    tint: `hsl(${hue} ${Math.max(30, sat).toFixed(0)}% ${Math.min(34, lit + 6).toFixed(0)}%)`,
  };
}
const cover = (audio) => `<div class="cover" style="${coverArt(audio).style}" aria-hidden="true"></div>`;

/* ---------- small templates ---------- */
const nameOf = (t) => t.track_name;
const artistOf = (t) => t.artist_name ?? (t.artists || []).join(", ");
const trackHref = (t) => `#/track/${encodeURIComponent(t.track_id)}`;

const card = (t) => `
  <a class="card" href="${trackHref(t)}">
    ${cover(t.audio)}
    <b>${esc(nameOf(t))}</b>
    <span>${esc(artistOf(t))}</span>
  </a>`;
const shelf = (items) => `<div class="shelf">${items.map(card).join("")}</div>`;
const shelfHead = (title, sub = "") =>
  `<div class="shelf-head"><h2>${esc(title)}</h2>${sub ? `<p>${esc(sub)}</p>` : ""}</div>`;
const row = (t, end = t.year) => `
  <a class="row" href="${trackHref(t)}">
    ${cover(t.audio)}
    <span class="t"><b>${esc(nameOf(t))}</b><span>${esc(artistOf(t))}</span></span>
    <span class="end">${esc(end)}</span>
  </a>`;
const searchForm = (q = "") => `
  <form class="searchbar" role="search" data-search>
    ${icon("i-search")}
    <input name="q" type="search" value="${esc(q)}" placeholder="Songs or artists"
           aria-label="Search songs or artists" autocomplete="off">
  </form>`;

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
      .slice(0, 12);
    recentList.innerHTML = items.length
      ? items.map((h) => `<li>${row(h.track, "")}</li>`).join("")
      : `<li class="empty">Songs you play or like show up here.</li>`;
  } catch {
    recentList.innerHTML = `<li class="empty">Couldn't load your history.</li>`;
  }
}

/* ---------- views ---------- */
async function homeView(isCurrent) {
  const listened = (await getHistory()).filter((h) => h.event_type !== "skip");
  if (!isCurrent()) return;
  if (!listened.length) {
    const popular = await api(`/recommend/user/${u()}?k=12`);
    if (!isCurrent()) return;
    view.innerHTML = `
      <div class="view-pad">
        <section class="welcome">
          <h1>Find music by how it sounds.</h1>
          <p>Search for a song you love. Vibify suggests others with a similar energy, mood, tempo and era,
             from 170,000 tracks released between 1921 and 2020.</p>
          ${searchForm()}
        </section>
        <section>
          ${shelfHead("Popular in the catalog", "Until you play or like something, these are the safest bets.")}
          ${shelf(popular.items)}
        </section>
      </div>`;
    return;
  }
  const last = listened[0].track;
  const [mix, more] = await Promise.all([
    api(`/recommend/user/${u()}?k=18`),
    api(`/recommend/similar/${encodeURIComponent(last.track_id)}?k=12`),
  ]);
  if (!isCurrent()) return;
  const n = new Set(listened.map((h) => h.track.track_id)).size;
  view.innerHTML = `
    <div class="view-pad">
      <section>
        <h1 class="mix-title">Your mix</h1>
        <p class="mix-sub">Built from the sound of the ${n} ${n === 1 ? "song" : "songs"} you've played or liked.</p>
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
  const q = params.get("q") || "";
  view.innerHTML = `<div class="view-pad">${searchForm(q)}<div class="results" aria-live="polite"></div></div>`;
  const input = view.querySelector("input[name=q]");
  const results = view.querySelector(".results");
  input.focus();
  input.setSelectionRange(q.length, q.length);

  const run = async (term) => {
    term = term.trim();
    history.replaceState(null, "", term ? `#/search?q=${encodeURIComponent(term)}` : "#/search");
    if (!term) {
      results.innerHTML = `<p class="note">Search by song title or artist.</p>`;
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
    searchTimer = setTimeout(() => run(input.value).catch(showError), 220);
  });
  await run(q);
}

const SOUND = [
  ["energy", "Energy"], ["danceability", "Danceability"], ["valence", "Positivity"],
  ["acousticness", "Acoustic"], ["instrumentalness", "Instrumental"], ["speechiness", "Spoken word"],
  ["liveness", "Live feel"],
];

async function trackView(isCurrent, _params, id) {
  const [t, sim, hist] = await Promise.all([
    api(`/tracks/${encodeURIComponent(id)}`),
    api(`/recommend/similar/${encodeURIComponent(id)}?k=20`),
    getHistory(),
  ]);
  if (!isCurrent()) return;
  const mine = hist.filter((h) => h.track.track_id === id);
  const liked = mine.some((h) => h.event_type === "like");
  const skipped = mine.some((h) => h.event_type === "skip");
  const art = coverArt(t.audio);
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
    <header class="hero" style="--tint:${art.tint}">
      <div class="cover" style="${art.style}" aria-hidden="true"></div>
      <div class="hero-text">
        <h1 class="${title.length > 22 ? "long" : ""}">${esc(title)}</h1>
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
      <button class="text-btn" data-act="skip" ${skipped ? "disabled" : ""}>${skipped ? "Hidden from your mix" : "Not for me"}</button>
    </div>
    <div class="track-body">
      <section aria-labelledby="sim-h">
        <h2 id="sim-h">Similar songs</h2>
        <ol class="tracklist">
          ${sim.items.map((s, i) => `<li><a class="row" href="${trackHref(s)}">
              <span class="n">${i + 1}</span>${cover(s.audio)}
              <span class="t"><b>${esc(s.track_name)}</b><span>${esc(s.artist_name)}</span></span>
              <span class="yr">${s.year}</span>
              <span class="match" title="Similarity of sound, genre and era">${Math.round(s.similarity * 100)}%</span>
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

/* ---------- actions on the track page ---------- */
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
      el.textContent = "Hidden from your mix";
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
    ? `<div class="error"><b>That song isn't in the catalog.</b><a href="#/search">Search for another one.</a></div>`
    : `<div class="error"><b>The recommender didn't respond.</b>${esc(err.message)}. Check that the server is running, then reload.</div>`;
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
  if (name !== "search") view.innerHTML = `<p class="loading">Loading…</p>`;
  try {
    await viewFn(() => seq === renderSeq, new URLSearchParams(query), arg && decodeURIComponent(arg));
  } catch (err) {
    if (seq === renderSeq) showError(err);
  }
}

window.addEventListener("hashchange", render);
refreshRecent();
render();
