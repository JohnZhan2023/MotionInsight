# MotionInsight research demo

Static research showcase for the `demo` branch. Repository code and the original README are preserved.

## Local preview

From this checkout:

```sh
python3 -m http.server 4173 --bind 127.0.0.1
```

Open http://127.0.0.1:4173/. No installation or build is required.

## Content and provenance

- `index.html`, `styles.css`, and `app.js` provide an English academic project page, with English video prompts. The simple centered layout follows the user's PerpetualWonder reference; no code or research content from that project is copied.
- Author links shared with the DuoMatching reference use its destinations. Xuanyu Zhang and Jingqi Tong link to verified personal homepages; Junlin Li links to the PKU VILLA academic author page; Li Zhang links to the Google Scholar profile cited by Qunliang Xing’s personal homepage.
- The logo and method figure are the existing repository assets.
- Research figures and DPO procedure are from Appendix F / Table 9 of https://arxiv.org/abs/2609.37030.
- The withdrawn `video_comparison_questionnaire_20260915_v7.zip` material is not included.
- The 16 MP4s in `media/comparisons/` are derived from the replacement `truemotion_questionnaire_20260518/generated/videos/` folder. The source files remain unchanged. Each output has a fixed left-to-right order: Wan, DPO baseline, MotionInsight-DPO (Ours). The 832 × 480 panels are cropped and restacked according to the annotated manifest. Reordered clips use H.264 high-profile, yuv420p, CRF 16 and fast-start; q001 already has the requested order and is copied unchanged. All outputs retain 2496 × 480 resolution, 61 frames at 12 fps, original durations of about 5.08 seconds, and no audio. Full-frame comparison against the expected rearrangement verifies SSIM above 0.99 for every clip.
- `comparison-data.js` retains the new source's exact English prompts; Chinese prompts have been removed at the user's request. Display titles are editorial scene labels. Posters are regenerated from the reordered clips at one second. Scene thumbnails use the middle (DPO baseline) panel at that timestamp.
- The latest `truemotion_questionnaire_20260518(1).zip` adds `generated/manifests/source_mapping.json`. All 16 source videos were verified byte-for-byte against that archive before rearrangement. Its per-example mappings determine the crop order. `methods` in `comparison-data.js` now stores the fixed displayed order, `["wan", "dpo", "ours"]`, for every video. Raw source folder/file paths are not included in the website.
- Display names: `wan` = Wan; `dpo` = DPO baseline; `ours` = MotionInsight-DPO (Ours), using the user's established project context. The archive does not identify the full DPO baseline name or a Wan version; do not silently equate its `dpo` tag with VideoPhy2-DPO. Only the model names are displayed above the video; anonymous A/B/C markers are omitted.
- The video viewer provides 16 thumbnail choices and native playback/seeking/fullscreen controls. The custom previous/next buttons and playback-speed selector are removed. Only the selected video is loaded.
- `citation.bib` uses the user-supplied arXiv `@misc` citation, with the URL normalized to valid BibTeX syntax. The copy control has a manual-selection fallback, and the file is directly downloadable.
- Fonts use Google Fonts with local system-font fallbacks. No analytics, login, or server-side storage is used.

## Publication

The user approved publication on 2026-10-04. GitHub Pages serves the project site at https://johnzhan2023.github.io/MotionInsight/.

Publish from branch `demo`, directory `/` (root). `.nojekyll` is included, and asset URLs are relative so the page works under `/MotionInsight/`.

Before publishing, retain the fixed model order and the corresponding rearranged media together. Confirm the formal DPO baseline name and Wan version before expanding their display names. The published human-study values must remain attributed to the paper and must not be presented as statistics computed from the demo videos.
