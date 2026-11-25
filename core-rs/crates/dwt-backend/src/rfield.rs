//! Per-pixel PWS region field.
//!
//! Luminance-normalized and band-pass envelopes for the integration grid.

use crate::error::{BackendError, BackendResult};
use crate::frame::Frame;
use crate::pws::select_median;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RFieldMode {
    #[default]
    Max,
    Band,
    Ctn,
}

/// Parameters for the region-field construction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RFieldConfig {
    pub sigma_psf: f32,
    pub window_multiplier: f32,
    pub noise_epsilon: f32,
    pub shape_power: f32,
    pub floor: f32,
}

impl Default for RFieldConfig {
    fn default() -> Self {
        Self {
            sigma_psf: 3.0,
            window_multiplier: 2.0,
            noise_epsilon: 16.0,
            shape_power: 1.0,
            floor: 0.0,
        }
    }
}

impl RFieldConfig {
    fn validate(self) -> BackendResult<()> {
        for (name, value) in [
            ("sigma_psf", self.sigma_psf),
            ("window_multiplier", self.window_multiplier),
            ("noise_epsilon", self.noise_epsilon),
            ("shape_power", self.shape_power),
            ("floor", self.floor),
        ] {
            if !value.is_finite() {
                return Err(BackendError::InvalidValue { name, value });
            }
        }
        if self.sigma_psf <= 0.0 {
            return Err(BackendError::InvalidValue {
                name: "sigma_psf",
                value: self.sigma_psf,
            });
        }
        if self.window_multiplier <= 0.0 {
            return Err(BackendError::InvalidValue {
                name: "window_multiplier",
                value: self.window_multiplier,
            });
        }
        if self.noise_epsilon < 0.0 || self.shape_power <= 0.0 {
            return Err(BackendError::Message(
                "R field requires nonnegative noise epsilon and positive shape power".into(),
            ));
        }
        if !(0.0..=0.95).contains(&self.floor) {
            return Err(BackendError::InvalidValue {
                name: "floor",
                value: self.floor,
            });
        }
        Ok(())
    }
}

/// Builds a region field from a reference image.
pub fn build_r_field(reference: &Frame, config: RFieldConfig) -> BackendResult<Frame> {
    build_r_field_with_mode(reference, config, RFieldMode::Max, 0.0, 3.0)
}

pub(crate) fn build_r_field_with_mode(
    reference: &Frame,
    config: RFieldConfig,
    mode: RFieldMode,
    band_window: f32,
    contrast: f32,
) -> BackendResult<Frame> {
    config.validate()?;
    let width = reference.width();
    let height = reference.height();
    let pixels = reference.pixels();
    let (sky, sigma) = sky_and_noise(pixels)?;
    let mut signal: Vec<_> = pixels
        .iter()
        .map(|&value| {
            if value.is_finite() {
                (value - sky).max(0.0)
            } else {
                0.0
            }
        })
        .collect();
    let radius = (config.window_multiplier * config.sigma_psf.max(0.5))
        .round_ties_even()
        .max(1.0) as usize;
    let band_width = if band_window > 0.0 {
        band_window
    } else {
        (4.0 * config.sigma_psf.max(0.5)).max(9.0)
    };
    let band_radius = ((band_width.round_ties_even() as usize).max(3) | 1) / 2;
    if mode != RFieldMode::Max {
        let smooth = box_mean(&signal, width, height, band_radius);
        for (value, local) in signal.iter_mut().zip(smooth) {
            *value -= local;
        }
    }
    if mode == RFieldMode::Ctn {
        let (_, noise) = sky_and_noise(&signal)?;
        let low = 0.8 * noise;
        let scale = ((contrast.max(1.05) - 0.8) * noise).max(1.0e-30);
        for value in &mut signal {
            *value = (*value - low) / scale;
        }
    } else {
        let radius = if mode == RFieldMode::Band {
            band_radius
        } else {
            radius
        };
        let peaks = local_maximum(&signal, width, height, radius);
        let epsilon = f64::from(config.noise_epsilon * sigma);
        for (value, peak) in signal.iter_mut().zip(peaks) {
            let denominator = peak + epsilon;
            *value = if denominator > 0.0 {
                (f64::from(*value) / denominator) as f32
            } else {
                0.0
            };
        }
    }
    let output = pixels
        .iter()
        .zip(signal)
        .map(|(source, value)| {
            if source.is_finite() {
                let envelope = value.clamp(0.0, 1.0).powf(config.shape_power);
                config.floor + (1.0 - config.floor) * envelope
            } else {
                0.0
            }
        })
        .collect();
    Frame::new(width, height, output)
}

fn sky_and_noise(pixels: &[f32]) -> BackendResult<(f32, f32)> {
    let mut finite: Vec<_> = pixels
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    let sky = select_median(&mut finite).ok_or(BackendError::EmptyPixels)?;
    for value in &mut finite {
        *value = (*value - sky).abs();
    }
    let noise = select_median(&mut finite).unwrap_or(0.0);
    Ok((sky, (1.4826 * noise).max(1.0e-30)))
}

fn box_mean(pixels: &[f32], width: usize, height: usize, radius: usize) -> Vec<f32> {
    let mean = |line: &[f32]| {
        let n = line.len();
        let mut sums = Vec::with_capacity(n + 1);
        sums.push(0.0_f64);
        let mut running = 0.0;
        for &value in line {
            running += f64::from(value);
            sums.push(running);
        }
        let radius = radius as f64;
        (0..n)
            .map(|i| {
                let start = i.saturating_sub(radius as usize);
                let end = i.saturating_add(radius as usize).saturating_add(1).min(n);
                let left = (radius - i as f64).max(0.0) * f64::from(line[0]);
                let right = (radius - (n - 1 - i) as f64).max(0.0) * f64::from(line[n - 1]);
                ((sums[end] - sums[start] + left + right) / (2.0 * radius + 1.0)) as f32
            })
            .collect::<Vec<_>>()
    };
    let horizontal: Vec<f32> = pixels.chunks_exact(width).flat_map(mean).collect();
    let mut output = vec![0.0; pixels.len()];
    let mut column = vec![0.0; height];
    for x in 0..width {
        for y in 0..height {
            column[y] = horizontal[y * width + x];
        }
        for (y, value) in mean(&column).into_iter().enumerate() {
            output[y * width + x] = value;
        }
    }
    output
}

fn local_maximum(pixels: &[f32], width: usize, height: usize, radius: usize) -> Vec<f64> {
    let horizontal: Vec<f64> = pixels
        .chunks_exact(width)
        .flat_map(|row| sliding_window_maximum(row, radius))
        .collect();
    let mut vertical = vec![0.0_f64; pixels.len()];
    let mut column = vec![0.0_f64; height];
    for x in 0..width {
        for y in 0..height {
            column[y] = horizontal[y * width + x];
        }
        for (y, value) in sliding_window_maximum(&column, radius)
            .into_iter()
            .enumerate()
        {
            vertical[y * width + x] = value;
        }
    }
    vertical
}

fn sliding_window_maximum<T>(values: &[T], half_window: usize) -> Vec<f64>
where
    T: Copy + Into<f64>,
{
    let length = values.len();
    let mut output = vec![0.0_f64; length];
    if length == 0 {
        return output;
    }
    let half_window = half_window.min(length - 1);
    let mut deque = std::collections::VecDeque::new();
    for index in 0..length + half_window {
        if index < length {
            let value = to_f64(values[index]).max(0.0);
            while deque
                .back()
                .is_some_and(|&back| to_f64(values[back]).max(0.0) <= value)
            {
                deque.pop_back();
            }
            deque.push_back(index);
        }
        if index < half_window {
            continue;
        }
        let center = index - half_window;
        let left = center.saturating_sub(half_window);
        while deque.front().is_some_and(|&front| front < left) {
            deque.pop_front();
        }
        output[center] = deque
            .front()
            .and_then(|&front| values.get(front))
            .copied()
            .map(|value| to_f64(value).max(0.0))
            .unwrap_or(0.0);
    }
    output
}

fn to_f64<T>(value: T) -> f64
where
    T: Copy + Into<f64>,
{
    value.into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_reference_builds_zero_region_field() {
        let reference = Frame::filled(8, 8, 10.0).expect("reference");
        let field = build_r_field(&reference, RFieldConfig::default()).expect("field");
        assert!(field.pixels().iter().all(|value| *value == 0.0));
    }

    #[test]
    fn bright_center_builds_positive_region_field() {
        let mut pixels = vec![0.0_f32; 9 * 9];
        for y in 3..6 {
            for x in 3..6 {
                pixels[y * 9 + x] = 1.0;
            }
        }
        let reference = Frame::new(9, 9, pixels).expect("reference");
        let field = build_r_field(
            &reference,
            RFieldConfig {
                sigma_psf: 1.0,
                noise_epsilon: 0.1,
                ..RFieldConfig::default()
            },
        )
        .expect("field");
        assert!(field.pixel(4, 4).expect("center") > 0.0);
    }

    #[test]
    fn all_envelopes_match_python_reference_samples() -> BackendResult<()> {
        let pixels = (0..9)
            .flat_map(|y| {
                (0..11).map(move |x| {
                    let background = 0.4 + f64::from((x * 17 + y * 29) % 13) * 0.002;
                    let distance = (x - 5) * (x - 5) + (y - 4) * (y - 4);
                    (background + 0.09 * (-f64::from(distance) / 2.0).exp()) as f32
                })
            })
            .collect();
        let reference = Frame::new(11, 9, pixels)?;
        let config = RFieldConfig {
            sigma_psf: 1.0,
            noise_epsilon: 3.0,
            ..RFieldConfig::default()
        };
        for (mode, expected) in [
            (
                RFieldMode::Max,
                [0.01019456, 0.37813818, 0.715247, 0.15445338],
            ),
            (RFieldMode::Band, [0.0, 0.262639, 0.6577093, 0.0]),
            (RFieldMode::Ctn, [0.0, 1.0, 1.0, 0.0]),
        ] {
            let field = build_r_field_with_mode(&reference, config, mode, 5.0, 3.0)?;
            for ((x, y), value) in [(6, 2), (5, 3), (5, 4), (6, 5)].into_iter().zip(expected) {
                let actual = field.pixels()[y * 11 + x];
                assert!(
                    (actual - value).abs() < 5.0e-5,
                    "{mode:?} at {x},{y}: {actual} != {value}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn sliding_window_matches_direct_scan() {
        let values = [0.0_f32, 3.0, 1.0, 4.0, 2.0, 5.0, 0.5, 2.5, 1.5];
        for half_window in 1_usize..4 {
            let actual = sliding_window_maximum(&values, half_window);
            let expected: Vec<f64> = (0..values.len())
                .map(|index| {
                    let start = index.saturating_sub(half_window);
                    let end = (index + half_window).min(values.len() - 1);
                    values[start..=end]
                        .iter()
                        .copied()
                        .map(f64::from)
                        .fold(0.0_f64, f64::max)
                })
                .collect();
            assert_eq!(actual, expected, "half_window={half_window}");
        }
    }
}
