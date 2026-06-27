const SEATS = ["E", "S", "W", "N"];
const TURN_ORDER = ["E", "N", "W", "S"];
const SEAT_NAME = { E: "East", S: "South", W: "West", N: "North" };
const TEAM_NAME = { EW: "East-West", SN: "South-North" };
const STORAGE_KEY = "guandan.webui.controllers.v1";
const HAND_TYPES = [
  ["", "Auto"],
  ["single", "Single"],
  ["pair", "Pair"],
  ["three_of_a_kind", "Trips"],
  ["full_house", "Full house"],
  ["straight", "Straight"],
  ["straight_flush", "Straight flush"],
  ["bomb", "Bomb"],
  ["four_jokers", "Four jokers"],
  ["three_pair_run", "Three-pair run"],
  ["triple_run", "Triple run"],
];

const CARD_SUIT_FILE = {
  S: "spades",
  H: "hearts",
  D: "diamonds",
  C: "clubs",
};

const CARD_RANK_FILE = {
  2: "02",
  3: "03",
  4: "04",
  5: "05",
  6: "06",
  7: "07",
  8: "08",
  9: "09",
  10: "10",
  J: "j",
  Q: "q",
  K: "k",
  A: "a",
};

const BASE_RANK_VALUE = {
  3: 1,
  4: 2,
  5: 3,
  6: 4,
  7: 5,
  8: 6,
  9: 7,
  10: 8,
  J: 9,
  Q: 10,
  K: 11,
  A: 12,
  2: 13,
  SJ: 15,
  BJ: 16,
};

const el = {
  tableSelect: document.getElementById("tableSelect"),
  refreshTables: document.getElementById("refreshTables"),
  createTable: document.getElementById("createTable"),
  quickStart: document.getElementById("quickStart"),
  statusLine: document.getElementById("statusLine"),
  scoreStrip: document.getElementById("scoreStrip"),
  seatTop: document.getElementById("seatTop"),
  seatRight: document.getElementById("seatRight"),
  seatBottom: document.getElementById("seatBottom"),
  seatLeft: document.getElementById("seatLeft"),
  trickZone: document.getElementById("trickZone"),
  seatTabs: document.getElementById("seatTabs"),
  seatControls: document.getElementById("seatControls"),
  eventLog: document.getElementById("eventLog"),
};

const state = {
  tables: [],
  tableId: null,
  snapshot: null,
  privateSnapshots: {},
  controllers: loadControllers(),
  viewSeat: localStorage.getItem("guandan.webui.viewSeat") || "S",
  selectedCards: new Set(),
  seatActions: {},
  seatActionDealId: null,
  events: [],
  lastSelectedCardId: null,
  busy: false,
  pollId: null,
  autoInFlight: false,
  lastAutoKey: "",
};

function loadControllers() {
  try {
    const parsed = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch {
    return {};
  }
}

function saveControllers() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(state.controllers));
}

function controllersForTable(tableId = state.tableId) {
  if (!tableId) return {};
  if (!state.controllers[tableId]) state.controllers[tableId] = {};
  return state.controllers[tableId];
}

function setLocalController(tableId, seat, controller) {
  if (!state.controllers[tableId]) state.controllers[tableId] = {};
  state.controllers[tableId][seat] = controller;
  saveControllers();
}

function localHumanSeats() {
  const controllers = controllersForTable();
  return SEATS.filter((seat) => controllers[seat]?.kind === "human");
}

function localOwnedSeats() {
  const controllers = controllersForTable();
  return SEATS.filter((seat) => controllers[seat]?.controllerId);
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function status(message, isError = false) {
  el.statusLine.textContent = message;
  el.statusLine.classList.toggle("error-text", isError);
}

function errorMessage(payload, fallback) {
  if (!payload || typeof payload !== "object") return fallback;
  if (payload.rejection?.message) return payload.rejection.message;
  if (payload.detail) return typeof payload.detail === "string" ? payload.detail : JSON.stringify(payload.detail);
  if (payload.error) return payload.error;
  return fallback;
}

async function request(path, options = {}) {
  const headers = { Accept: "application/json" };
  const init = {
    method: options.method || "GET",
    headers,
  };
  if (options.body !== undefined) {
    headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(options.body);
  }
  const response = await fetch(path, init);
  let payload = {};
  const text = await response.text();
  if (text) {
    try {
      payload = JSON.parse(text);
    } catch {
      payload = { error: text };
    }
  }
  if (!response.ok) {
    const message = errorMessage(payload, `${init.method} ${path} failed with ${response.status}`);
    const error = new Error(message);
    error.payload = payload;
    error.status = response.status;
    throw error;
  }
  if (Array.isArray(payload.events)) appendEvents(payload.events);
  if (payload.snapshot) {
    state.snapshot = payload.snapshot;
    syncSeatActionsFromSnapshot(payload.snapshot);
  }
  return payload;
}

async function runAction(label, action) {
  if (state.busy) return;
  state.busy = true;
  status(label);
  render();
  try {
    await action();
  } catch (error) {
    status(error.message || String(error), true);
  } finally {
    state.busy = false;
    render();
  }
}

function appendEvents(events) {
  for (const event of events) {
    applySeatActionEvent(event);
    state.events.unshift(event);
  }
  state.events = state.events.slice(0, 80);
}

function resetSeatActions(dealId = null) {
  state.seatActions = {};
  state.seatActionDealId = dealId;
}

function syncSeatActionsFromSnapshot(snapshot) {
  if (!snapshot) return;
  const dealId = Number.isFinite(snapshot.deal_id) ? snapshot.deal_id : null;
  if (state.seatActionDealId !== dealId) resetSeatActions(dealId);
  const trick = snapshot.current_trick;
  if (!trick || !Array.isArray(trick.card_ids) || trick.card_ids.length === 0) return;
  if (!SEATS.includes(trick.last_play_seat)) return;
  state.seatActions[trick.last_play_seat] = {
    kind: "cards",
    cardIds: trick.card_ids,
    handType: trick.hand_type,
  };
}

function applySeatActionEvent(event) {
  const payload = event?.payload || {};
  if (["MatchStarted", "DealStarted", "CardsDealt"].includes(event?.type)) {
    resetSeatActions(null);
    return;
  }
  if (event?.type === "CardsPlayed" && SEATS.includes(payload.seat) && Array.isArray(payload.card_ids)) {
    state.seatActions[payload.seat] = {
      kind: "cards",
      cardIds: payload.card_ids,
      handType: payload.hand_type,
    };
    return;
  }
  if (event?.type === "PlayerPassed" && SEATS.includes(payload.seat)) {
    state.seatActions[payload.seat] = { kind: "pass" };
    return;
  }
  if (event?.type === "TributePaid" && SEATS.includes(payload.giver) && payload.card_id) {
    state.seatActions[payload.giver] = {
      kind: "cards",
      cardIds: [payload.card_id],
      handType: "tribute",
      label: "Tribute",
    };
    return;
  }
  if (event?.type === "TributeReturned" && SEATS.includes(payload.receiver) && payload.card_id) {
    state.seatActions[payload.receiver] = {
      kind: "cards",
      cardIds: [payload.card_id],
      handType: "return",
      label: "Return",
    };
  }
}

async function refreshTables() {
  const payload = await request("/tables");
  state.tables = Array.isArray(payload.tables) ? payload.tables : [];
  const urlTable = new URLSearchParams(window.location.search).get("table");
  if (!state.tableId && urlTable && state.tables.includes(urlTable)) {
    state.tableId = urlTable;
  }
  if (!state.tableId && state.tables.length > 0) {
    state.tableId = state.tables[0];
  }
  if (state.tableId && !state.tables.includes(state.tableId)) {
    state.tableId = null;
    state.snapshot = null;
    state.privateSnapshots = {};
    resetSeatActions(null);
  }
  render();
  if (state.tableId) await syncSnapshot();
}

async function createTable() {
  const payload = await request("/tables", {
    method: "POST",
    body: { action_timeout_seconds: 45, timeout_fallback: "auto_pass" },
  });
  state.tableId = payload.table_id;
  state.tables = Array.from(new Set([payload.table_id, ...state.tables]));
  state.snapshot = payload.snapshot || null;
  state.privateSnapshots = {};
  state.selectedCards.clear();
  resetSeatActions(null);
  updateUrlTable();
  startPolling();
  status(`Created ${payload.table_id}`);
  await syncSnapshot();
}

async function quickStart() {
  await createTable();
  const tableId = state.tableId;
  await joinSeat("S", "human", "You");
  for (const seat of ["E", "N", "W"]) {
    await joinSeat(seat, "bot", `RL Agent ${seat}`);
  }
  for (const seat of SEATS) {
    const controller = controllersForTable(tableId)[seat];
    if (controller) {
      await request(`/tables/${tableId}/ready`, {
        method: "POST",
        body: { seat, controller_id: controller.controllerId },
      });
    }
  }
  await request(`/tables/${tableId}/start`, { method: "POST", body: {} });
  state.viewSeat = "S";
  localStorage.setItem("guandan.webui.viewSeat", state.viewSeat);
  status(`Started ${tableId}`);
  await syncSnapshot();
}

async function selectTable(tableId) {
  state.tableId = tableId || null;
  state.snapshot = null;
  state.privateSnapshots = {};
  state.selectedCards.clear();
  resetSeatActions(null);
  updateUrlTable();
  startPolling();
  if (state.tableId) await syncSnapshot();
  render();
}

function updateUrlTable() {
  const url = new URL(window.location.href);
  if (state.tableId) url.searchParams.set("table", state.tableId);
  else url.searchParams.delete("table");
  window.history.replaceState({}, "", url);
}

function startPolling() {
  if (state.pollId) window.clearInterval(state.pollId);
  if (!state.tableId) return;
  state.pollId = window.setInterval(() => {
    syncSnapshot({ quiet: true }).catch((error) => status(error.message || String(error), true));
  }, 1500);
}

async function syncSnapshot(options = {}) {
  if (!state.tableId) return;
  const payload = await request(`/tables/${state.tableId}`);
  state.snapshot = payload;
  syncSeatActionsFromSnapshot(payload);
  pruneMissingLocalControllers();
  ensureViewSeat();
  await fetchHumanPrivateSnapshots();
  if (!options.quiet) status(`${state.tableId} synced`);
  render();
  maybeAutoAct();
}

function pruneMissingLocalControllers() {
  if (!state.snapshot) return;
  const local = controllersForTable();
  for (const seat of Object.keys(local)) {
    if (!state.snapshot.seats?.[seat]) {
      delete local[seat];
      delete state.privateSnapshots[seat];
    }
  }
  saveControllers();
}

function ensureViewSeat() {
  const humans = localHumanSeats();
  if (humans.includes(state.viewSeat)) return;
  if (humans.length > 0) state.viewSeat = humans[0];
  else if (!SEATS.includes(state.viewSeat)) state.viewSeat = "S";
  localStorage.setItem("guandan.webui.viewSeat", state.viewSeat);
}

async function fetchHumanPrivateSnapshots() {
  const local = controllersForTable();
  const humans = localHumanSeats();
  for (const seat of humans) {
    const controller = local[seat];
    try {
      const query = new URLSearchParams({ controller_id: controller.controllerId });
      state.privateSnapshots[seat] = await request(
        `/tables/${state.tableId}/seats/${seat}/snapshot?${query.toString()}`,
      );
    } catch {
      delete state.privateSnapshots[seat];
    }
  }
}

async function joinSeat(seat, kind, displayName = null) {
  if (!state.tableId) return;
  const action = kind === "human" ? "join-human" : "join-local-bot";
  const storedKind = kind === "human" ? "human" : "rl_agent";
  const label = displayName || (kind === "human" ? `Human ${seat}` : `RL Agent ${seat}`);
  const payload = await request(`/tables/${state.tableId}/${action}`, {
    method: "POST",
    body: { seat, display_name: label },
  });
  setLocalController(state.tableId, seat, {
    controllerId: payload.controller_id,
    playerId: payload.player_id,
    kind: storedKind,
    displayName: label,
  });
  if (kind === "human") {
    state.viewSeat = seat;
    localStorage.setItem("guandan.webui.viewSeat", seat);
  }
  status(`${label} seated`);
  await syncSnapshot();
}

async function readySeat(seat) {
  const controller = controllersForTable()[seat];
  if (!state.tableId || !controller) return;
  await request(`/tables/${state.tableId}/ready`, {
    method: "POST",
    body: { seat, controller_id: controller.controllerId },
  });
  status(`${SEAT_NAME[seat]} ready`);
  await syncSnapshot();
}

async function startDeal() {
  if (!state.tableId) return;
  await request(`/tables/${state.tableId}/start`, { method: "POST", body: {} });
  state.selectedCards.clear();
  resetSeatActions(null);
  status("Deal started");
  await syncSnapshot();
}

async function submitHumanAction(kind) {
  const seat = state.viewSeat;
  const privateSnapshot = state.privateSnapshots[seat];
  const controller = controllersForTable()[seat];
  if (!state.tableId || !privateSnapshot || !controller) return;

  if (kind === "pass") {
    await sendSeatAction(seat, controller.controllerId, { kind: "pass" });
  } else if (kind === "play") {
    const cardIds = [...state.selectedCards];
    await sendSeatAction(seat, controller.controllerId, {
      kind: "play_cards",
      cardIds,
    });
  } else if (kind === "tribute") {
    const cardId = [...state.selectedCards][0];
    await sendSeatAction(seat, controller.controllerId, { kind: "submit_tribute", cardId });
  } else if (kind === "return") {
    const cardId = [...state.selectedCards][0];
    await sendSeatAction(seat, controller.controllerId, { kind: "return_tribute", cardId });
  }
  state.selectedCards.clear();
  await syncSnapshot();
}

async function sendSeatAction(seat, controllerId, action) {
  if (action.kind === "pass") {
    await request(`/tables/${state.tableId}/pass`, {
      method: "POST",
      body: { seat, controller_id: controllerId },
    });
    return;
  }
  if (action.kind === "play_cards") {
    const body = {
      seat,
      controller_id: controllerId,
      card_ids: action.cardIds,
    };
    await request(`/tables/${state.tableId}/play`, { method: "POST", body });
    return;
  }
  if (action.kind === "submit_tribute") {
    await request(`/tables/${state.tableId}/tribute`, {
      method: "POST",
      body: { seat, controller_id: controllerId, card_id: action.cardId },
    });
    return;
  }
  if (action.kind === "return_tribute") {
    await request(`/tables/${state.tableId}/return-tribute`, {
      method: "POST",
      body: { seat, controller_id: controllerId, card_id: action.cardId },
    });
  }
}

async function maybeAutoAct() {
  if (!state.tableId || !state.snapshot || state.busy || state.autoInFlight) return;
  if (!["PLAYING", "TRIBUTE"].includes(state.snapshot.phase)) return;
  const seat = state.snapshot.acting_seat || state.snapshot.current_turn;
  const controller = controllersForTable()[seat];
  if (!seat || !controller || controller.kind === "human") return;

  const key = `${state.tableId}:${seat}:${state.snapshot.event_seq}:${state.snapshot.phase}`;
  if (state.lastAutoKey === key) return;
  state.lastAutoKey = key;
  state.autoInFlight = true;
  try {
    status(`${controller.displayName || seat} thinking`);
    await wait(420);
    await request(`/tables/${state.tableId}/bot-action`, {
      method: "POST",
      body: {
        seat,
        controller_id: controller.controllerId,
        request_id: key,
      },
    });
    await syncSnapshot({ quiet: true });
  } catch (error) {
    status(error.message || String(error), true);
  } finally {
    state.autoInFlight = false;
  }
}

function wait(ms) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function render() {
  renderTableSelect();
  renderScoreStrip();
  renderTable();
  renderSeatTabs();
  renderSeatControls();
  renderEvents();
  const disabled = state.busy;
  el.createTable.disabled = disabled;
  el.quickStart.disabled = disabled;
  el.refreshTables.disabled = disabled;
}

function renderTableSelect() {
  if (state.tables.length === 0) {
    el.tableSelect.innerHTML = `<option value="">No tables</option>`;
    el.tableSelect.value = "";
    return;
  }
  el.tableSelect.innerHTML = state.tables
    .map((tableId) => `<option value="${escapeHtml(tableId)}">${escapeHtml(tableId)}</option>`)
    .join("");
  el.tableSelect.value = state.tableId || state.tables[0] || "";
}

function renderScoreStrip() {
  const snapshot = state.snapshot;
  if (!snapshot) {
    el.scoreStrip.innerHTML = `<div class="score-pill"><strong>No table</strong></div>`;
    return;
  }
  const levelByTeam = snapshot.level_by_team || {};
  const deadline = snapshot.action_deadline_epoch_ms ? Math.max(0, snapshot.action_deadline_epoch_ms - Date.now()) : 0;
  const seconds = deadline ? Math.ceil(deadline / 1000) : null;
  el.scoreStrip.innerHTML = [
    pill("Phase", formatPhase(snapshot.phase)),
    pill("Level", snapshot.current_level || "2"),
    pill("EW", levelByTeam.EW || "2"),
    pill("SN", levelByTeam.SN || "2"),
    pill("Turn", snapshot.acting_seat || snapshot.current_turn || "-"),
    pill("Clock", seconds === null ? "-" : `${seconds}s`),
  ].join("");
}

function pill(label, value) {
  return `<div class="score-pill"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`;
}

function renderTable() {
  const snapshot = state.snapshot;
  const positions = {
    top: el.seatTop,
    right: el.seatRight,
    bottom: el.seatBottom,
    left: el.seatLeft,
  };
  for (const target of Object.values(positions)) target.innerHTML = "";

  if (!snapshot) {
    el.trickZone.innerHTML = `
      <div class="trick-title">No active table</div>
      <div class="trick-subtitle">Create or select a table</div>
    `;
    return;
  }

  for (const seat of SEATS) {
    const position = positionForSeat(seat);
    positions[position].innerHTML = renderSeatZone(seat, position);
  }
  renderTrickZone();
  bindCardButtons();
}

function positionForSeat(seat) {
  const viewIndex = TURN_ORDER.indexOf(state.viewSeat);
  const seatIndex = TURN_ORDER.indexOf(seat);
  const relative = (seatIndex - viewIndex + TURN_ORDER.length) % TURN_ORDER.length;
  return ["bottom", "right", "top", "left"][relative];
}

function renderSeatZone(seat, position) {
  const snapshot = state.snapshot;
  const player = snapshot.seats?.[seat];
  const count = snapshot.hand_counts?.[seat] || 0;
  const isTurn = (snapshot.acting_seat || snapshot.current_turn) === seat;
  const privateSnapshot = seat === state.viewSeat ? state.privateSnapshots[seat] : null;
  const exactCards = privateSnapshot?.hand || null;
  const controller = controllersForTable()[seat];
  const metaParts = [];
  if (player?.kind) metaParts.push(player.kind);
  if (controller?.kind) metaParts.push(`local ${controller.kind}`);
  const meta = metaParts.length ? metaParts.join(" / ") : "open";
  const handClass = position === "bottom" ? "hand-strip" : "hand-strip compact-hand";
  const tableActions = position === "bottom" ? renderTableActionDock() : "";
  const played = renderSeatPlayed(seat, position);
  const cards = exactCards
    ? renderCards(sortCards(exactCards, snapshot.current_level), {
        selectable: controller?.kind === "human",
        eligible: privateSnapshot.eligible_card_ids || [],
        level: snapshot.current_level,
      })
    : renderCardBacks(count, position === "bottom" ? 18 : 8);
  const visibleCardCount = exactCards ? exactCards.length : Math.min(count, position === "bottom" ? 18 : 8);
  const playerPlate = `
    <div class="${isTurn ? "turn-ring" : ""}">
      <div class="player-plate">
        <div class="seat-avatar">${escapeHtml(seat)}</div>
        <div class="player-copy">
          <span class="seat-name">${escapeHtml(player?.display_name || `${SEAT_NAME[seat]} seat`)}</span>
          <span class="seat-meta">${escapeHtml(meta)}</span>
        </div>
      </div>
    </div>
  `;
  if (position === "bottom") {
    return `
      ${played}
      <div class="${handClass}" style="--card-count: ${visibleCardCount || 1}">
        ${cards}
        ${!exactCards && count ? `<span class="count-badge">${count}</span>` : ""}
      </div>
      <div class="human-seat-row">
        ${playerPlate}
        ${tableActions}
      </div>
    `;
  }
  return `
    ${playerPlate}
    <div class="${handClass}" style="--card-count: ${visibleCardCount || 1}">
      ${cards}
      ${!exactCards && count ? `<span class="count-badge">${count}</span>` : ""}
    </div>
    ${played}
  `;
}

function renderSeatPlayed(seat, position) {
  const action = state.seatActions[seat];
  const classes = ["seat-played", `seat-played-${position}`];
  if (!action) {
    return `<div class="${classes.concat("empty").join(" ")}" aria-hidden="true"></div>`;
  }
  if (action.kind === "pass") {
    return `<div class="${classes.concat("pass-marker").join(" ")}"><span>Pass</span></div>`;
  }
  const cardIds = Array.isArray(action.cardIds) ? action.cardIds : [];
  const label = action.label || formatHandType(action.handType);
  const playedCardCount = Math.max(cardIds.length, 1);
  return `
    <div class="${classes.join(" ")}">
      <div class="played-label">${escapeHtml(label)}</div>
      <div class="played-cards" style="--played-card-count: ${playedCardCount}">
        ${renderCards(cardIds, { selectable: false, level: state.snapshot.current_level })}
      </div>
    </div>
  `;
}

function renderTrickZone() {
  const snapshot = state.snapshot;
  const trick = snapshot.current_trick;
  const activeSeat = snapshot.acting_seat || snapshot.current_turn || "-";
  if (!trick || !Array.isArray(trick.card_ids) || trick.card_ids.length === 0) {
    el.trickZone.innerHTML = `
      <div class="trick-title">${escapeHtml(formatPhase(snapshot.phase))}</div>
      <div class="trick-subtitle">Turn ${escapeHtml(activeSeat)}</div>
    `;
    return;
  }
  const title = `${SEAT_NAME[trick.last_play_seat] || trick.last_play_seat} played ${formatHandType(trick.hand_type)}`;
  el.trickZone.innerHTML = `
    <div class="trick-title">${escapeHtml(title)}</div>
    <div class="trick-subtitle">Primary ${escapeHtml(trick.primary_rank || "-")} / turn ${escapeHtml(activeSeat)}</div>
  `;
}

function renderCards(cards, options = {}) {
  const eligible = new Set(options.eligible || []);
  return cards
    .map((cardId, index) => {
      const selected = state.selectedCards.has(cardId);
      const classes = [
        "card-shell",
        options.selectable ? "selectable" : "",
        selected ? "selected" : "",
        eligible.has(cardId) ? "eligible" : "",
        isWildCard(cardId, options.level) ? "wild" : "",
      ]
        .filter(Boolean)
        .join(" ");
      const action = options.selectable
        ? `data-card-id="${escapeHtml(cardId)}" data-card-index="${index}" aria-pressed="${selected ? "true" : "false"}"`
        : "";
      return `
        <button class="${classes}" type="button" ${action} title="${escapeHtml(cardLabel(cardId))}">
          <img class="card-image" src="${escapeHtml(cardImage(cardId))}" alt="${escapeHtml(cardLabel(cardId))}">
        </button>
      `;
    })
    .join("");
}

function renderCardBacks(count, maxCards) {
  const visible = Math.min(count, maxCards);
  return Array.from({ length: visible })
    .map(
      (_, index) => `
        <span class="card-shell" title="Hidden card ${index + 1}">
          <img class="card-image" src="/ui/assets/cards/card_back.png" alt="Hidden card">
        </span>
      `,
    )
    .join("");
}

function bindCardButtons() {
  document.querySelectorAll("[data-card-id]").forEach((button) => {
    button.addEventListener("click", (event) => {
      const cardId = button.getAttribute("data-card-id");
      if (!cardId) return;
      const allCards = [...document.querySelectorAll("[data-card-id]")].map((item) => item.getAttribute("data-card-id"));
      if (event.shiftKey && state.lastSelectedCardId && allCards.includes(state.lastSelectedCardId)) {
        const start = allCards.indexOf(state.lastSelectedCardId);
        const end = allCards.indexOf(cardId);
        const [from, to] = start <= end ? [start, end] : [end, start];
        for (const selectedCardId of allCards.slice(from, to + 1)) {
          if (selectedCardId) state.selectedCards.add(selectedCardId);
        }
      } else if (state.selectedCards.has(cardId)) {
        state.selectedCards.delete(cardId);
      } else {
        state.selectedCards.add(cardId);
      }
      state.lastSelectedCardId = cardId;
      renderTable();
    });
  });
}

function renderSeatTabs() {
  const humans = localHumanSeats();
  if (humans.length === 0) {
    el.seatTabs.innerHTML = `
      <div class="section-title">View</div>
      <div class="empty-state">Observer</div>
    `;
    return;
  }
  el.seatTabs.innerHTML = `
    <div class="section-title">View</div>
    <div class="seat-tabs">
      ${humans
        .map(
          (seat) => `
            <button class="seat-tab ${seat === state.viewSeat ? "active" : ""}" type="button" data-action="view-seat" data-seat="${seat}">
              ${escapeHtml(SEAT_NAME[seat])}
            </button>
          `,
        )
        .join("")}
    </div>
  `;
}

function renderSeatControls() {
  if (!state.tableId || !state.snapshot) {
    el.seatControls.innerHTML = `
      <div class="section-title">Seats</div>
      <div class="empty-state">No table selected</div>
    `;
    return;
  }
  const startDisabled = state.busy || !["READY_CHECK", "DEAL_COMPLETE", "MATCH_COMPLETE"].includes(state.snapshot.phase);
  el.seatControls.innerHTML = `
    <div class="section-title">Seats</div>
    ${SEATS.map(renderSeatControl).join("")}
    <div class="command-row">
      <button type="button" data-action="start" ${startDisabled ? "disabled" : ""}>Start Deal</button>
    </div>
  `;
}

function renderSeatControl(seat) {
  const player = state.snapshot.seats?.[seat];
  const controller = controllersForTable()[seat];
  const occupied = Boolean(player);
  const buttons = occupied
    ? controller
      ? `<button class="seat-action" type="button" data-action="ready" data-seat="${seat}" ${state.busy ? "disabled" : ""}>Ready</button>`
      : `<span class="seat-control-meta">Observed</span>`
    : `
      <button class="seat-action" type="button" data-action="join-human" data-seat="${seat}" ${state.busy ? "disabled" : ""}>Human</button>
      <button class="seat-action" type="button" data-action="join-bot" data-seat="${seat}" ${state.busy ? "disabled" : ""}>RL Agent</button>
    `;
  return `
    <div class="seat-control">
      <div class="seat-letter">${escapeHtml(seat)}</div>
      <div class="seat-control-copy">
        <div class="seat-control-title">${escapeHtml(player?.display_name || `${SEAT_NAME[seat]} seat`)}</div>
        <div class="seat-control-meta">${escapeHtml(player?.kind || "empty")}</div>
        <div class="seat-actions">${buttons}</div>
      </div>
    </div>
  `;
}

function renderTableActionDock() {
  if (!state.tableId || !state.snapshot) {
    return "";
  }
  const controller = controllersForTable()[state.viewSeat];
  const privateSnapshot = state.privateSnapshots[state.viewSeat];
  if (!controller || controller.kind !== "human") {
    return "";
  }
  if (!privateSnapshot) {
    return `
      <div class="table-action-dock">
        <div class="command-row table-command-row">
          <button class="action-command primary" type="button" data-action="play" disabled>Play</button>
          <button class="action-command" type="button" data-action="pass" disabled>Pass</button>
          <button class="action-command" type="button" data-action="tribute" disabled>Tribute</button>
          <button class="action-command" type="button" data-action="return" disabled>Return</button>
          <button class="action-command quiet" type="button" data-action="clear-selection" ${state.selectedCards.size === 0 ? "disabled" : ""}>Clear</button>
        </div>
      </div>
    `;
  }
  const activeSeat = state.snapshot.acting_seat || state.snapshot.current_turn;
  const isTurn = activeSeat === state.viewSeat;
  const selected = [...state.selectedCards];
  const isPlay = ["lead", "play_or_pass"].includes(privateSnapshot.legal_action);
  const isTribute = privateSnapshot.legal_action === "tribute";
  const isReturn = privateSnapshot.legal_action === "return_tribute";
  const eligible = new Set(privateSnapshot.eligible_card_ids || []);
  const selectedEligible = selected.length === 1 && (eligible.size === 0 || eligible.has(selected[0]));
  const playDisabled = state.busy || !isTurn || !isPlay || selected.length === 0;
  const passDisabled = state.busy || !isTurn || privateSnapshot.legal_action !== "play_or_pass";
  const tributeDisabled = state.busy || !isTurn || !isTribute || !selectedEligible;
  const returnDisabled = state.busy || !isTurn || !isReturn || !selectedEligible;
  return `
    <div class="table-action-dock">
      <div class="command-row table-command-row">
        <button class="action-command primary" type="button" data-action="play" ${playDisabled ? "disabled" : ""}>Play</button>
        <button class="action-command" type="button" data-action="pass" ${passDisabled ? "disabled" : ""}>Pass</button>
        <button class="action-command" type="button" data-action="tribute" ${tributeDisabled ? "disabled" : ""}>Tribute</button>
        <button class="action-command" type="button" data-action="return" ${returnDisabled ? "disabled" : ""}>Return</button>
        <button class="action-command quiet" type="button" data-action="clear-selection" ${state.selectedCards.size === 0 ? "disabled" : ""}>Clear</button>
      </div>
    </div>
  `;
}

function renderEvents() {
  if (state.events.length === 0) {
    el.eventLog.innerHTML = `<div class="empty-state">No events</div>`;
    return;
  }
  el.eventLog.innerHTML = state.events
    .map(
      (event) => `
        <div class="event-item">
          <div class="event-type">#${escapeHtml(event.seq)} ${escapeHtml(event.type)}</div>
          <div class="event-payload">${escapeHtml(JSON.stringify(event.payload || {}))}</div>
        </div>
      `,
    )
    .join("");
}

function cardImage(cardId) {
  const parts = String(cardId).split("-");
  if (parts.length === 2) {
    return parts[1] === "BJ" ? "/ui/assets/cards/card_joker_02.png" : "/ui/assets/cards/card_joker_01.png";
  }
  const suit = CARD_SUIT_FILE[parts[1]] || "spades";
  const rank = CARD_RANK_FILE[parts[2]] || "02";
  return `/ui/assets/cards/card_${suit}_${rank}.png`;
}

function cardLabel(cardId) {
  const parts = String(cardId).split("-");
  if (parts.length === 2) {
    return `${parts[0]} ${parts[1] === "BJ" ? "Big Joker" : "Small Joker"}`;
  }
  return `${parts[0]} ${parts[1]} ${parts[2]}`;
}

function cardRank(cardId) {
  const parts = String(cardId).split("-");
  return parts.length === 2 ? parts[1] : parts[2];
}

function rankValue(rank, level) {
  if (rank === "BJ") return 16;
  if (rank === "SJ") return 15;
  if (rank === level) return 14;
  return BASE_RANK_VALUE[rank] || 0;
}

function sortCards(cards, level = "2") {
  return [...cards].sort((left, right) => {
    const diff = rankValue(cardRank(right), level) - rankValue(cardRank(left), level);
    if (diff !== 0) return diff;
    return String(left).localeCompare(String(right));
  });
}

function isWildCard(cardId, level = "2") {
  const parts = String(cardId).split("-");
  return parts.length === 3 && parts[1] === "H" && parts[2] === level;
}

function formatPhase(value) {
  return String(value || "-")
    .toLowerCase()
    .split("_")
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function formatHandType(value) {
  const found = HAND_TYPES.find(([key]) => key === value);
  return found ? found[1] : formatPhase(value);
}

document.addEventListener("click", (event) => {
  const button = event.target.closest("[data-action]");
  if (!button) return;
  const action = button.getAttribute("data-action");
  const seat = button.getAttribute("data-seat");
  if (action === "join-human") runAction("Joining human", () => joinSeat(seat, "human"));
  else if (action === "join-bot") runAction("Joining bot", () => joinSeat(seat, "bot"));
  else if (action === "ready") runAction("Ready", () => readySeat(seat));
  else if (action === "start") runAction("Starting deal", startDeal);
  else if (action === "view-seat") {
    state.viewSeat = seat;
    localStorage.setItem("guandan.webui.viewSeat", seat);
    state.selectedCards.clear();
    render();
  } else if (action === "play") runAction("Playing cards", () => submitHumanAction("play"));
  else if (action === "pass") runAction("Passing", () => submitHumanAction("pass"));
  else if (action === "tribute") runAction("Submitting tribute", () => submitHumanAction("tribute"));
  else if (action === "return") runAction("Returning tribute", () => submitHumanAction("return"));
  else if (action === "clear-selection") {
    state.selectedCards.clear();
    state.lastSelectedCardId = null;
    render();
  }
});

el.tableSelect.addEventListener("change", () => {
  runAction("Loading table", () => selectTable(el.tableSelect.value));
});
el.refreshTables.addEventListener("click", () => runAction("Refreshing tables", refreshTables));
el.createTable.addEventListener("click", () => runAction("Creating table", createTable));
el.quickStart.addEventListener("click", () => runAction("Starting quick match", quickStart));

refreshTables()
  .then(() => {
    startPolling();
    render();
  })
  .catch((error) => {
    status(error.message || String(error), true);
    render();
  });
