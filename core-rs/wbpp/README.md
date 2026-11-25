# WBPP integration

This directory contains the PixInsight WBPP-side adapter for the Rust backend.
It does not modify the Python application.

## Scope

WBPP calibrates the light frames and can debayer them into separate color
channels. The Pipeline Builder exports each supported group from
`processed[DEBAYER]` or `processed[CALIBRATION]`. Rust measures frame quality
and star positions, registers the frames, and integrates their pixels with
PWS and Drizzle. PixInsight saves the result as XISF.

For a DWT-only integration, enable the Pipeline Builder and turn off WBPP
Image Registration, Local Normalization, Image Integration, and Drizzle
Integration. Rust self-registration is active by default. The project can
reuse existing XDRZ sidecars as registration seeds with `"useXdrzSeeds": true`
in its `DWT/dwt-backend.json`. WBPP Image Registration can produce those
sidecars when you choose to use them.

## Build

```powershell
cargo build --manifest-path core-rs/Cargo.toml --release --bin dwt-backend
```

Install into the PixInsight script directory:

```text
PixInsight/src/scripts/DWT-Drizzle/bin/dwt-backend.exe
```

Build the Rust executable, then run the installer for the current PixInsight
script directory:

```powershell
core-rs/wbpp/install.ps1 -PixInsightScripts "C:/Program Files/PixInsight/src/scripts"
```

Then select `DWT-Drizzle/pipeline-builder.js` in:

```text
WBPP > Pipeline > Pipeline scripting > Pipeline builder
```

The Pipeline operation list and execution monitor show `DWT-Drizzle (Rust)`.
The Process Console logs the builder script path when the operation starts.
The console and the operation monitor report frame measurement, registration,
reference construction, Drizzle row progress, and output writing.

Optionally select `DWT-Drizzle/event-script.js` in the Event Script field.
The script reads `dwt-backend.json` next to the selected Pipeline Builder,
then applies an optional project override at
`<WBPP output>/DWT/dwt-backend.json`. See the [configuration example](dwt-backend.json.example).
`fluxPower` and `detailSnr` default to zero.

| Setting | Default | Effect |
| --- | --- | --- |
| `drizzleScale` | `1.0` | Output pixels per input pixel. Set `2.0` for a 2× Drizzle canvas. |
| `drizzlePixfrac` | `1.0` | Drop width in input pixels. |
| `referenceFrames` | `0` | Number of sharp registered frames used for the median reference. Zero uses up to eight. |
| `rFieldMode` | `"max"` | Local peak envelope. `"band"` weights band-pass structures. `"ctn"` uses contrast relative to noise. |
| `rFieldWindow` | `0.0` | Band-pass window in output pixels. Zero sizes the window from the reference FWHM. |
| `rFieldContrast` | `3.0` | Signal-to-noise level for full detail weight in `"ctn"` mode. |

## Handoff contract

The pipeline-builder exports each active regular light group's calibrated or
debayered frames to:

```text
<WBPP output>/DWT/runs/<group>/<unique-frame-id>.f32le
```

Each raster is row-major float32 little-endian. The builder then runs:

```text
dwt-backend.exe wbpp-pws <request.json> <result.json>
```

The Pipeline Builder request uses the `dwt-wbpp-command/1` schema. It records
the calibrated frame paths, float32 rasters, exposure times, and available WBPP
quality measurements. XDRZ 1.0 matrices map reference to source.

```text
source = M * reference
reference = inverse(M) * source
```

When XDRZ seeds are enabled, Rust inverts the matrices and converts them to the
chosen reference frame's coordinates. It refines the initial transforms with
stars from the calibrated frames. Invalid files produce a warning and the
affected frames use Rust self-registration.

## Current behavior

The WBPP pipeline-builder currently:

1. Runs the configured WBPP light-frame preprocessing.
2. Exports each supported group as single-channel float32 rasters.
3. Runs `dwt-backend.exe wbpp-pws`. The backend normalizes exposure, measures
   noise and FWHM, verifies each transform against the star catalog, and
   retries failed matches with triangle and neighbor-frame registration.
   When neighbor-frame recovery succeeds, the backend can refine those
   transforms with weighted least squares. It builds the R field from the
   median of the sharpest registered frames, then applies PWS with
   Winsorized rejection on the Drizzle grid.
4. Loads the Rust output raster and saves
   `<WBPP output>/DWT/runs/<group>/master_dwt_<group>.xisf`.

Current support covers regular single-channel groups and separate debayered
channels. Raw CFA Drizzle, wgpu, radial distortion, and adaptive star-support
dilation are open implementation areas. The Rust result JSON records the
registered-frame count, fallback counts, and the number of sharp reference frames.
Calibrated frames remain in memory. The backend checks available RAM before
loading a group and again before allocating the Drizzle canvas. Reduce the
group size or Drizzle scale if the estimated allocation exceeds the budget.
