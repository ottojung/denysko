# Canonical glyph source — provenance & pin

Denysko's canonical glyph input is a **vendored, SHA-pinned** font artifact so the
rasterized geometry is byte-identical on every machine and in CI. No system-installed
font or fontconfig lookup is ever used.

## File

- `fonts/ttf/CormorantUpright-SemiBold.ttf`

## Why this font

Cormorant describes its *Upright* style as an upright script version of the Italic, so
it is much closer to the desired calligraphic look than the previous sans-serif face.
It is open source under the SIL Open Font License 1.1, and the family covers Latin and
Cyrillic. SemiBold is preferred over Light/Regular because Denysko rasterizes to a fixed
512×512 mask and skeletonizes the filled strokes; a somewhat heavier cut is less fragile
than a very thin calligraphic face.

## License

- **SIL Open Font License 1.1**
- License text: `fonts/ttf/CormorantUpright-OFL.txt`
- Copyright © 2015, Christian Thalmann and the Cormorant Project Authors
  (github.com/CatharsisFonts/Cormorant)

## Upstream source

- Project: `CatharsisFonts/Cormorant`
  (https://github.com/CatharsisFonts/Cormorant)
- Canonical relative path in the upstream tree:
  `fonts/ttf/CormorantUpright-SemiBold.ttf`
- Mirrored (and used for the byte-pinned download):
  `https://github.com/google/fonts/raw/main/ofl/cormorantupright/CormorantUpright-SemiBold.ttf`

## Pin

The artifact is pinned by SHA-256. The recorded digest below is the single source of
truth; `src/topology.py::_verify_font_pin` aborts at import time if the bytes on disk
do not match. A legitimate font update must change the artifact, the digest in
`src/topology.py::_PINNED_SHA256`, **and** this file together.

```
sha256sum fonts/ttf/CormorantUpright-SemiBold.ttf
585e9106c433f1b4cc5d023103305123d92741526a7e27e9ff8a1f5befcc90e6  fonts/ttf/CormorantUpright-SemiBold.ttf
```

## Version

The Cormorant project does not publish semantic version tags; the
authoritative version of the artifact is its content, fixed by the
SHA-256 pin above. Treat the digest (and the matching
`src/topology.py::_PINNED_SHA256`) as the version of record. Any change
to the vendored bytes is, by definition, a new version and must update
the artifact, the code constant, and this file together.

## Follow-up calibration note

After the switch, stroke thickness at the 512 raster scale should be inspected. If
SemiBold still produces sub-raster fragile strokes for some letters, prefer the Bold
cut *globally* (replace the vendored artifact and update the pin) rather than introducing
per-letter dilation.
