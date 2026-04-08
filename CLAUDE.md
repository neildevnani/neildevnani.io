# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a static personal website for Neil Devnani — no build system, no package manager, no framework. All pages are plain HTML files sharing a single `style.css`.

## Structure

- `index.html` — homepage with intro and photo
- `resume.html` — embeds `Neil Devnani Final Resume copy.pdf` via `<iframe>`
- `DS_Projects.html` — data science projects page; embeds Jupyter notebook HTML exports and links to `nn_visualizer/`
- `soccer.html`, `socials.html` — other personal pages
- `style.css` — shared stylesheet used by all top-level pages
- `nn_visualizer/` — self-contained interactive neural network visualizer (XOR problem); has its own `index.html`, `style.css`, and `script.js`
- `baby_model.html`, `solar_activity_model (1) (2).html` — Jupyter notebook HTML exports embedded via `<iframe>` in DS_Projects.html

## Development

Open any `.html` file directly in a browser — no server needed for most pages. To preview locally with a dev server (useful for iframe content):

```bash
python3 -m http.server 8080
```

## Styling Conventions

All top-level pages link to `style.css` which defines CSS custom properties at `:root`:
- Colors: `--bg1`, `--bg2`, `--card`, `--text`, `--muted`, `--accent` (blue `#60a5fa`), `--accent2` (violet `#a78bfa`)
- Max width: `--maxw: 960px`
- Reusable classes: `.card`, `.btn`, `.grid`, `.hero`, `.video-container`

The `nn_visualizer/` subdirectory has its own scoped `style.css` and does not inherit from the root stylesheet.

## Adding New Pages

Copy the nav/header/footer pattern from any existing page. Link the new page in the `<nav>` of all other pages.
