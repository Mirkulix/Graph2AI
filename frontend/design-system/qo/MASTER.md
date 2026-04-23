# Clarity — QO Design System (Master)

> **Logic:** Page-specific specs in `design-system/pages/[page-name].md` override
> this file. Otherwise, this is the single source of truth for QO frontend
> styling. Tokens live in `frontend/src/styles.css` — when this doc and
> `styles.css` disagree, `styles.css` wins and this file is updated.

---

## 1. Identity

- **Name:** Clarity
- **Mode:** Light-first (a dark theme is shipped via `[data-theme='dark']`,
  but light is the default and the design baseline).
- **Mission:** *Glasklar, ruhig, AI-native.* Calm surfaces, one accent, colour
  only where it carries meaning. Data-forward, never decorative.
- **Stack:** React 19 + Vite 6, served embedded by the `qo` server (port 4646).

---

## 2. Color tokens

All values are in `:root` of `frontend/src/styles.css`. Use the CSS variable,
not the literal hex, in components.

### Surfaces

| Token | Hex | Usage |
|-------|-----|-------|
| `--bg` | `#F8F9FB` | App background (off-white, never pure white) |
| `--surface` | `#FFFFFF` | Cards, sidebar, header, panels |
| `--surface-sunken` | `#F2F4F7` | Inputs, hovered chips, code blocks |
| `--surface-elevated` | `#FFFFFF` | Modals, popovers (paired with shadow) |

### Text

| Token | Hex | Usage |
|-------|-----|-------|
| `--text` | `#0F1117` | Primary copy, headings (off-black) |
| `--text-muted` | `#5A6272` | Body secondary, labels |
| `--text-dim` | `#8A93A3` | Captions, eyebrow text, placeholder |
| `--text-inverse` | `#FFFFFF` | Text on `--accent` fills |

### Lines

| Token | Hex | Usage |
|-------|-----|-------|
| `--border` | `#E5E8EE` | Default 1 px hairline |
| `--border-strong` | `#CBD1DB` | Hovered borders, dividers needing weight |

### Accent (single colour, used sparingly)

| Token | Hex | Usage |
|-------|-----|-------|
| `--accent` | `#5B5BF0` | Primary action, active nav, focus ring core |
| `--accent-soft` | `#EEF0FE` | Pill backgrounds, focus halo, accent badges |
| `--accent-strong` | `#4040D4` | Primary button hover/active, accent text on soft fill |

### Semantic (only these three)

| Role | Token | Hex | Soft Hex |
|------|-------|-----|----------|
| Success | `--ok` / `--ok-soft` | `#10B981` | `#DFF6EC` |
| Warning | `--warn` / `--warn-soft` | `#F59E0B` | `#FEF2DC` |
| Error | `--err` / `--err-soft` | `#EF4444` | `#FEE4E4` |

> "Info" is **not** a separate semantic — informational state uses `--accent` /
> `--accent-soft`. Adding a fourth semantic colour requires an explicit design
> review.

### Werte palette (scoped)

Used **only** inside the values radar and Werte chips. Do not reuse for general
UI: `--werte-achtsamkeit #2EA07A`, `--werte-anerkennung #8369D0`,
`--werte-aufmerksamkeit #D97747`, `--werte-entwicklung #C99216`,
`--werte-sinn #5275D6`.

---

## 3. Typography

```css
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
```

- **Sans (UI):** `Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif`
- **Mono (code, IDs, metrics):** `'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace`
- **OpenType features:** `font-feature-settings: 'cv02', 'cv03', 'cv11'` (Inter
  alternates — already set on `body`).

### Scale

| Token | Size | Typical use |
|-------|------|-------------|
| `xs` | 10 px | Eyebrows, badges, dot labels |
| `sm` | 11 – 12 px | Pills, table cells, captions |
| `base` | 13 – 14 px | Body, buttons, nav items (body default = 14 px) |
| `md` | 15 – 16 px | Sidebar brand, primary inputs, sub-headings |
| `lg` | 20 px | Section headings |
| `xl` | 24 – 28 px | View titles, card stat figures |
| `2xl` | 32 px | Empty-state headings |
| `3xl` | 44 px | Hero metric ("big number") |
| `4xl` | 56 px+ | Reserve for marketing / mission control hero |

### Weights & line-heights

- **400** body, **500** medium UI emphasis, **600** buttons / labels,
  **700** headings & big numbers.
- **Line-heights:** 1.25 (headings & big numbers), 1.4 (UI / pills),
  1.5 (body — set on `body`), 1.6 (long-form prose, rare).

---

## 4. Spacing & radius

### Spacing — 4 px grid

| Token | Value | Usage |
|-------|-------|-------|
| `space-0` | 0 | Reset |
| `space-1` | 4 px | Tight inline gaps (icon ↔ label) |
| `space-2` | 8 px | Pill padding, chip gaps |
| `space-3` | 12 px | Default control padding y |
| `space-4` | 16 px | Card padding, default block gap |
| `space-5` | 20 px | Card padding (comfortable) |
| `space-6` | 24 px | Section gaps inside a view |
| `space-8` | 32 px | Page gutters |
| `space-10` | 40 px | View top padding |
| `space-12` | 48 px | Empty-state spacing |
| `space-16` | 64 px | Hero padding |

> Tailwind is **not** in use; spacing is applied via inline `style` or the
> hand-rolled class system in `styles.css`. Always quantise to multiples of 4 px.

### Radius

| Token | Value | Usage |
|-------|-------|-------|
| `--radius-button` | `10px` | Buttons, segmented controls |
| `--radius-input` | `10px` | Inputs, selects, textarea |
| `--radius-card` | `14px` | Cards, modals, panels |
| `--radius-chip` | `999px` | Pills, dots, status chips, avatars |

---

## 5. Elevation

Three subtle levels — never a dramatic drop shadow on a light surface.

| Token | Value | Usage |
|-------|-------|-------|
| `--shadow-card` | `0 1px 2px rgba(15, 17, 23, 0.05)` | Resting cards, sticky bars |
| `--shadow-raised` | `0 6px 20px rgba(15, 17, 23, 0.08)` | Hovered cards, popovers, focused inputs |
| `--shadow-dialog` | `0 20px 60px rgba(15, 17, 23, 0.18)` | Modals, command palette, full-screen overlays |

Dark mode swaps the alpha to a near-black: `0.4 / 0.5 / 0.65`. Never combine
two shadows; pick one tier.

---

## 6. Components (specs)

All transitions: `120ms var(--ease)` for colour/background, `160ms var(--ease)`
for transform / shadow. `--ease` = `cubic-bezier(0.22, 0.8, 0.36, 1)`.

### Button

| Variant | Background | Text | Border | Hover |
|---------|------------|------|--------|-------|
| **Primary** | `--accent` | `--text-inverse` | none | `background: --accent-strong` |
| **Secondary** | `--surface` | `--text` | `1px --border` | `border-color: --border-strong` |
| **Ghost** | transparent | `--text-muted` | none | `background: --surface-sunken; color: --text` |

- Padding: `8px 14px` (default), `10px 16px` (large), `6px 10px` (compact).
- Radius: `--radius-button`. Font-weight 600. `cursor: pointer` always.
- Focus: 3 – 4 px halo `--accent-soft` + 1 px `--accent` border.

### Card

```
background: var(--surface);
border: 1px solid var(--border);
border-radius: var(--radius-card);
box-shadow: var(--shadow-card);
padding: 16 – 20 px;
```

`.card--raised` lifts to `--shadow-raised`. Hover (when interactive):
`border-color: --border-strong; box-shadow: --shadow-raised; transform:
translateY(-1px);`. Never scale-transform on hover.

### Input

```
background: var(--surface);
border: 1px solid var(--border);
border-radius: var(--radius-input);
padding: 8px 12px;
font-size: 14px;
color: var(--text);
```

Focus: `border-color: --accent; box-shadow: 0 0 0 4px var(--accent-soft)`.
Placeholder: `color: var(--text-dim)`. Disabled: `background: --surface-sunken;
color: --text-dim`.

### Modal

- Overlay: `background: rgba(15, 17, 23, 0.45); backdrop-filter: blur(4px)`.
- Dialog: `background: --surface-elevated; border-radius: --radius-card;
  box-shadow: --shadow-dialog; padding: 24 – 32 px; max-width: 560 px`.
- Close affordance top-right (ghost button, 24 × 24 px hit target).

### Toast

- Width 320 – 420 px, `--radius-card`, `--shadow-raised`.
- Default: `background: --surface; border: 1px solid --border`.
- Variants: `.toast--ok` / `.toast--warn` / `.toast--err` use the matching
  `--*-soft` background, `color: var(--*)`, and `border-color:
  color-mix(in srgb, var(--*) 24%, transparent)`.
- Lifetime 4 s default, dismissible. Stack from bottom-right with 12 px gap.

### Tab / segmented control

- Container: `background: --surface-sunken; padding: 4 px; border-radius:
  --radius-button`.
- Item: `padding: 6px 12px; color: --text-muted; font-weight: 500`.
- Active: `background: --surface; color: --text; box-shadow: --shadow-card`.
- Optional accent variant (filter chip): `.pill--accent` →
  `background: --accent-soft; color: --accent-strong`.

---

## 7. Motion

| Token | Value | Usage |
|-------|-------|-------|
| `duration-fast` | `120ms` | Colour, background, border swaps |
| `duration-normal` | `160ms` | Transform, shadow, layout-bound props |
| `duration-slow` | `240ms` | Modal enter, drawer, large surface change |
| `--ease` | `cubic-bezier(0.22, 0.8, 0.36, 1)` | Standard ease-out for everything |

- Honour `prefers-reduced-motion: reduce` — disable transform/opacity
  transitions and snap to end state.
- No bouncy / spring easings. No infinite-loop animations except the dedicated
  pulse on live-status dots.

---

## 8. Do / Don't

1. **Don't** use pure black (`#000`) on pure white (`#FFF`). Use `--text` on
   `--bg` / `--surface`.
2. **Don't** introduce a second accent colour. One indigo, period. Reach for
   `--ok` / `--warn` / `--err` only when state demands it.
3. **Don't** stack shadows. Pick one elevation tier per surface.
4. **Don't** ship emoji as functional icons — use SVG (Lucide / Heroicons).
   Emoji is allowed only inside user-generated content.
5. **Don't** use scale or layout-shifting transforms on hover. A 1 px
   `translateY` plus a shadow change is the maximum.

---

## 9. Pre-delivery checklist

- [ ] All colour, radius, shadow, easing values come from CSS variables — no
      ad-hoc hex in `.tsx`.
- [ ] Body text contrast ≥ 4.5:1 against its surface.
- [ ] Every interactive element has a visible `:focus-visible` style.
- [ ] `cursor: pointer` on all clickable non-link elements.
- [ ] Light-mode reviewed first; dark-mode (`[data-theme='dark']`) checked
      after.
- [ ] Responsive sanity at 375 / 768 / 1024 / 1440 px.
- [ ] No horizontal scroll on mobile; nothing hidden behind the fixed header
      (56 px) or sidebar (240 px).
- [ ] `prefers-reduced-motion` respected.
