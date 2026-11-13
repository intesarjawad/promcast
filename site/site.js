// No dependencies. Without this file the page is a complete, static document: the sheet is drawn
// up to the second stamp, both arrivals stamped and the bar between them. With it, the clock hand
// moves, the forecast is drawn up to the hand, and stamps print as the hand reaches them.
const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const pad = (n) => String(n).padStart(2, "0");
const clock = (seconds) => {
	const total = Math.max(0, Math.round(seconds / 60));
	const h = Math.floor(total / 60);
	return h ? `${h}h ${pad(total % 60)}m` : `${total % 60}m`;
};
// The same wording as timetable.minutes in scripts/timetable.py.
const minutes = (seconds) =>
	seconds < 3570 ? `${Math.round(seconds / 60)} min` : clock(seconds);

const data = JSON.parse($("#sheet-data").textContent);
const input = $("#clock");
const SPAN = Number(input.max);
const hm = (seconds) => {
	const d = new Date((data.origin + seconds) * 1000);
	return `${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}`;
};

// A path's value `seconds` after now, linear between steps; the same rule as scripts/timetable.py.
const levelAt = (start, path, seconds) => {
	if (seconds <= 0) return start;
	const position = seconds / data.step;
	const index = Math.min(Math.floor(position), path.length - 1);
	const before = index === 0 ? start : path[index - 1];
	const after = position < path.length ? path[index] : path[path.length - 1];
	return before + (after - before) * (Math.min(position, path.length) - index);
};

const frames = $$(".frame").map((node) => {
	const svg = $("svg", node);
	const n = (key) => Number(svg.dataset[key]);
	const geo = {
		w: n("w"),
		left: n("left"),
		right: n("right"),
		top: n("top"),
		bottom: n("bottom"),
		before: n("before"),
		after: n("after"),
	};
	const x = (seconds) =>
		geo.left +
		((seconds + geo.before) / (geo.before + geo.after)) *
			(geo.right - geo.left);
	return {
		node,
		svg,
		geo,
		x,
		hand: $(".hand", svg),
		handLabel: $(".hand-l", svg),
		sweep: $(".sweep", svg),
		marks: $$(".arr [data-t]", svg),
		counts: $$(".verdict[data-from]", svg).map((text) => [
			$(".n", text),
			Number(text.dataset.from),
			Number(text.dataset.to),
		]),
		labels: $(".train", svg),
		ticks: $$(".tick", svg).map((tick) => [
			tick,
			Number(tick.getAttribute("x")),
		]),
		groups: $$("[data-i]", svg),
		arrivals: $$(".arr", svg),
		visible: () => node.offsetParent !== null && node.clientWidth > 0,
	};
});

const figure = $(".tt");
const events = $$("#events li");
const buttons = $$(".arrivals button");
const rd = {
	name: $("#rd-name"),
	p: $("#rd-p"),
	d: $("#rd-d"),
	say: $("#rd-say"),
	left: $("#rd-left"),
};

const state = { t: reduced ? Number(input.value) : 0, sel: 0 };
const series = () => data.series[state.sel];
// After a predicted crossing, display its predicted time.
const free = (s, path, arrived, t) =>
	arrived !== null && t >= arrived
		? hm(arrived)
		: `${(levelAt(s.now, path, t) * 100).toFixed(1)} % free`;

const placeInput = () => {
	const frame = frames.find((f) => f.visible());
	if (!frame) return;
	const scale = frame.node.clientWidth / frame.geo.w;
	Object.assign(input.style, {
		left: `${frame.node.offsetLeft + frame.x(0) * scale}px`,
		width: `${(frame.geo.right - frame.x(0)) * scale}px`,
		top: `${frame.node.offsetTop + frame.geo.top * scale}px`,
		height: `${(frame.geo.bottom - frame.geo.top + 30) * scale}px`,
	});
};

const render = () => {
	const { t } = state;
	const s = series();
	const promcast = free(s, s.q[0], s.p, t);
	const lin = s.lin ? free(s, s.lin, s.d, t) : null;
	for (const f of frames) {
		f.hand.style.transform = `translateX(${(f.x(t) - f.x(0)).toFixed(1)}px)`;
		f.handLabel.textContent = hm(t);
		const near = Number.parseFloat(f.svg.style.fontSize) * 3.8;
		for (const [tick, at] of f.ticks)
			tick.style.opacity = Math.abs(at - f.x(t)) < near ? 0 : 1;
		f.sweep.setAttribute("width", Math.max(0, f.x(t) - f.x(0) + 1).toFixed(1));
		for (const mark of f.marks)
			mark.classList.toggle("in", Number(mark.dataset.t) <= t + 1);
		for (const [n, from, to] of f.counts) n.textContent = minutes(to - from);
	}
	rd.name.firstChild.textContent = `${s.name} `;
	$("small", rd.name).textContent = s.place;
	rd.p.textContent = promcast;
	rd.d.textContent = lin ?? "no forecast";
	const [say, left] =
		s.p === null || s.d === null
			? [
					"Predicted crossing",
					s.p === null ? "none for Promcast" : "none for predict_linear",
				]
			: [
					"Predicted crossing gap",
					`${minutes(Math.abs(s.d - s.p))} (${s.d > s.p ? "linear later" : "linear earlier"})`,
				];
	rd.say.textContent = say;
	rd.left.textContent = left;
	let current = -1;
	events.forEach((li, i) => {
		const lit = Number(li.dataset.t) <= t + 1;
		li.classList.toggle("lit", lit);
		if (lit) current = i;
	});
	events.forEach((li, i) => {
		li.classList.toggle("cur", i === current);
	});
	input.value = String(t);
	input.setAttribute(
		"aria-valuetext",
		`${hm(t)} UTC. ${s.name}: Promcast ${promcast}${lin === null ? "" : `, predict_linear ${lin}`}. ${say ? `${say} ` : ""}${left}.`,
	);
};

const select = (index) => {
	state.sel = index;
	buttons.forEach((b, i) => {
		b.setAttribute("aria-pressed", String(i === index));
	});
	for (const f of frames) {
		for (const g of f.groups)
			g.classList.toggle("dim", Number(g.dataset.i) !== index);
		// The chosen series is drawn over the rest.
		const chosen = $(`.trn[data-i="${index}"]`, f.svg);
		chosen.parentNode.insertBefore(chosen, f.labels);
		for (const g of f.arrivals)
			g.classList.toggle("on", Number(g.dataset.i) === index);
	}
	render();
};
// The arrivals sit under the sheet, so choosing one brings the sheet back into view.
buttons.forEach((button, i) => {
	button.addEventListener("click", () => {
		select(i);
		figure.scrollIntoView({
			block: "nearest",
			behavior: reduced ? "auto" : "smooth",
		});
	});
});

// The opening: the hand sweeps from now to Promcast's arrival, where the first stamp prints and
// the hand holds; then on to predict_linear's, where the second prints and the bar is complete.
let intro = 0;
const stopIntro = () => {
	cancelAnimationFrame(intro);
	intro = 0;
};
const ease = (p) => (p < 0.5 ? 4 * p * p * p : 1 - (-2 * p + 2) ** 3 / 2);
const playIntro = () => {
	const first = data.series[0].p ?? 0;
	const second = data.series[0].d ?? first;
	const keys = [
		[0, 0],
		[2400, first],
		[3300, first],
		[4600, second],
	];
	const start = performance.now();
	const step = (now) => {
		const ms = Math.max(0, now - start);
		const k = keys.findIndex(([at]) => at > ms);
		if (k === -1) {
			state.t = second;
			stopIntro();
			render();
			return;
		}
		const [a0, t0] = keys[k - 1];
		const [a1, t1] = keys[k];
		state.t = t0 + (t1 - t0) * ease((ms - a0) / (a1 - a0));
		render();
		intro = requestAnimationFrame(step);
	};
	intro = requestAnimationFrame(step);
};

input.addEventListener("input", () => {
	stopIntro();
	state.t = Number(input.value);
	render();
});
// The native range snaps to whole steps; a pointer on the sheet moves the hand continuously.
input.addEventListener("pointermove", (event) => {
	if (!(event.buttons & 1)) return;
	const rect = input.getBoundingClientRect();
	stopIntro();
	state.t =
		Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width)) * SPAN;
	render();
});

placeInput();
render();
addEventListener("resize", placeInput);
document.fonts?.ready.then(placeInput);

if (!reduced) {
	const watch = new IntersectionObserver(([entry]) => {
		if (!entry.isIntersecting) return;
		watch.disconnect();
		playIntro();
	});
	watch.observe(figure);
}

// A lane under the pointer or focus says what its gap means, in the figure's caption.
for (const fig of $$(".fig")) {
	const caption = $("figcaption", fig);
	if (!caption || !$(".hit", fig)) continue;
	const resting = caption.innerHTML;
	for (const hit of $$(".hit", fig)) {
		const say = () =>
			(caption.textContent = `${hit.getAttribute("aria-label")}. ${hit.dataset.gap}.`);
		const rest = () => (caption.innerHTML = resting);
		hit.addEventListener("pointerenter", say);
		hit.addEventListener("focus", say);
		hit.addEventListener("pointerleave", rest);
		hit.addEventListener("blur", rest);
	}
}

// States: one board at a time, switched by tabs; without script all four stay stacked.
const tabs = $(".tabs");
if (tabs) {
	const panels = $$(".state");
	const pick = (index) => {
		panels.forEach((panel, i) => {
			panel.classList.toggle("on", i === index);
		});
		$$("button", tabs).forEach((button, i) => {
			button.setAttribute("aria-selected", String(i === index));
			button.tabIndex = i === index ? 0 : -1;
		});
	};
	panels.forEach((panel, i) => {
		const button = document.createElement("button");
		button.textContent = $("figcaption b", panel).textContent.replace(
			/\.$/,
			"",
		);
		button.type = "button";
		button.role = "tab";
		button.id = `state-tab-${i}`;
		button.setAttribute("aria-controls", `state-panel-${i}`);
		panel.id = `state-panel-${i}`;
		panel.setAttribute("role", "tabpanel");
		panel.setAttribute("aria-labelledby", button.id);
		button.addEventListener("click", () => pick(i));
		button.addEventListener("keydown", (event) => {
			const step = { ArrowRight: 1, ArrowLeft: -1 }[event.key];
			if (!step) return;
			const to = (i + step + panels.length) % panels.length;
			pick(to);
			$$("button", tabs)[to].focus();
		});
		tabs.append(button);
	});
	pick(0);
	document.documentElement.classList.add("js");
}

for (const button of $$("[data-copy]")) {
	button.addEventListener("click", async () => {
		const label = $(".copied", button);
		try {
			await navigator.clipboard.writeText(button.dataset.copy);
			label.textContent = "copied";
		} catch {
			label.textContent = "select and copy";
		}
		setTimeout(() => (label.textContent = ""), 1600);
	});
}
