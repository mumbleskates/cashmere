use std::cmp::Ordering::Equal;
use std::f32::consts::FRAC_1_SQRT_2;
use std::fs::File;
use std::io::BufWriter;
use std::ops::{Add, Div};
use std::path::PathBuf;
use std::str::FromStr;

use clap::{Parser, ValueEnum};
use coarsetime::{Duration, Instant};
use indicatif::{ProgressBar, ProgressFinish, ProgressStyle};
use num_traits::MulAddAssign;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use sha2::{Digest, Sha512};

use color::{LABColor, LUVColor, RGBColor, SRGBColor, XYZColor};

mod color;

/// Get a prefix of a slice as a reference to a fixed-size array type, if it is long enough.
fn try_prefix<T, const N: usize, U: AsRef<[T]>>(from: &U) -> Option<&[T; N]> {
    from.as_ref().get(..N).map(|s| s.try_into().unwrap())
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
struct Position(i32, i32);

impl Add for Position {
    type Output = Position;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0, self.1 + rhs.1)
    }
}

#[derive(Copy, Clone, PartialOrd, PartialEq, Debug)]
pub struct Color(pub [f32; 3]);

impl AsRef<[f32; 3]> for Color {
    #[inline]
    fn as_ref(&self) -> &[f32; 3] {
        &self.0
    }
}

impl From<LABColor<f32>> for Color {
    fn from(value: LABColor<f32>) -> Self {
        Self([value.l, value.a, value.b])
    }
}

impl From<LUVColor<f32>> for Color {
    fn from(value: LUVColor<f32>) -> Self {
        Self([value.l, value.u, value.v])
    }
}

impl From<XYZColor<f32>> for Color {
    fn from(value: XYZColor<f32>) -> Self {
        Self([value.x, value.y, value.z])
    }
}

impl From<RGBColor<f32>> for Color {
    fn from(value: RGBColor<f32>) -> Self {
        Self([value.red, value.green, value.blue])
    }
}

impl From<SRGBColor<f32>> for Color {
    fn from(value: SRGBColor<f32>) -> Self {
        Self([value.red, value.green, value.blue])
    }
}

impl MulAddAssign<&Self, f32> for Color {
    fn mul_add_assign(&mut self, a: &Self, b: f32) {
        self.0
            .iter_mut()
            .zip(a.0)
            .for_each(|(lhs, rhs)| *lhs += rhs * b);
    }
}

impl Div<f32> for Color {
    type Output = Self;

    fn div(self, rhs: f32) -> Self::Output {
        Self(self.0.map(|x| x / rhs))
    }
}

struct ImageField {
    width: u32,
    height: u32,
}

impl ImageField {
    fn len(&self) -> usize {
        self.width as usize * self.height as usize
    }

    fn position_to_index(&self, p: Position) -> usize {
        p.0 as usize + self.height as usize * p.1 as usize
    }

    fn index_to_position(&self, idx: usize) -> Position {
        Position(
            (idx % self.width as usize) as i32,
            (idx / self.width as usize) as i32,
        )
    }

    fn contains(&self, p: Position) -> bool {
        (0..self.width).contains(&(p.0 as u32)) && (0..self.height).contains(&(p.1 as u32))
    }
}

/// Offset, sampling weight, and whether it is a placeable neighbor
const SAMPLE_WEIGHTS: &[(Position, f32, bool)] = &[
    (Position(-1, 0), 1.0, true),
    (Position(-1, -1), FRAC_1_SQRT_2, false),
    (Position(0, -1), 1.0, true),
    (Position(1, -1), FRAC_1_SQRT_2, false),
    (Position(1, 0), 1.0, true),
    (Position(1, 1), FRAC_1_SQRT_2, false),
    (Position(0, 1), 1.0, true),
    (Position(-1, 1), FRAC_1_SQRT_2, false),
];

pub trait ColorMap {
    fn new() -> Self;
    fn len(&self) -> usize;
    fn insert(&mut self, key: u32, value: Color);
    fn nearest(&self, value: &Color) -> u32;
    fn remove(&mut self, key: u32);
    #[cfg(feature = "full_validation")]
    fn fully_validate(&mut self) {}
}

#[derive(Parser)]
#[command(name = "colordeposit")]
struct Cli {
    /// File the output PNG image will be written to
    output: Option<PathBuf>,
    /// The color metric used to determine similar colors.
    #[arg(long)]
    color_metric: Option<ColorMetric>,
    /// The order in which colors are added to the image. Possible values are 'shuffle' for random,
    /// or some variation of +R+G+B', '-G+B-R', etc. for a specific ordering from highest to lowest
    /// denomination. Components for specific orderings include R, G, B (SRGB red, green, and blue
    /// channels), l, u, v (CIE LUV color space coordinates), x, y, and z (CIE XYZ color space) and
    /// are each preceded either by '+' (ascending) or '-' (descending). There can be up to three
    /// such components; for example, 'ordered:-R+B' would be the equivalent of all colors shuffled,
    /// then ordered by ascending blue value, then finally ordered by descending red value (with
    /// green left randomized).
    #[arg(long)]
    ordering: Option<Ordering>,
    /// Shuffling seed (any string value).
    #[arg(long)]
    seed: Option<String>,
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
enum ColorMetric {
    /// CIELAB
    Lab,
    /// CIELUV
    Luv,
    /// CIE-XYZ
    Xyz,
    /// Linear RGB
    Rgb,
    /// SRGB
    Srgb,
}

use ColorMetric::*;

#[derive(Copy, Clone)]
enum Direction {
    Asc,
    Desc,
}

use Direction::*;

#[derive(Copy, Clone)]
struct SortIndex {
    idx: usize,
    dir: Direction,
}

#[derive(Clone)]
struct Ordering {
    shuffle_first: bool,
    ordering_metric: ColorMetric,
    sort_indices: Vec<SortIndex>, // 0-3 distinct indexes into the color's fields
}

impl FromStr for Ordering {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.is_empty() {
            return Err("ordering must not be empty; for a randomized ordering, use 'shuffle'");
        }
        if s == "shuffle" {
            return Ok(Self {
                shuffle_first: true,
                ordering_metric: Srgb, // doesn't matter
                sort_indices: vec![],
            });
        }
        let arg: Vec<char> = s.chars().collect();
        if arg.len() > 6 {
            return Err("ordering can't have more than three sort terms");
        }
        if arg.len() % 2 != 0 {
            return Err("ordering must alternate between +/- and some color channel");
        }
        let mut terms = Vec::<(char, Direction)>::new();
        for term in arg.chunks_exact(2) {
            if terms.iter().any(|previous_term| previous_term.0 == term[0]) {
                return Err("ordering expression has multiple terms with the same channel");
            }
            terms.push((
                term[1],
                match term[0] {
                    '+' => Asc,
                    '-' => Desc,
                    _ => {
                        return Err(
                            "each ordering term must start with '+' (ascending) or '-' (descending)",
                        );
                    }
                },
            ))
        }
        let channel_names = [(Lab, "lab"), (Luv, "luv"), (Xyz, "xyz"), (Rgb, "RGB")];
        let (ordering_metric, channel_letters) = 'finding_metric: {
            for (metric, letters) in channel_names {
                if terms.iter().all(|term| letters.contains(term.0)) {
                    break 'finding_metric (metric, letters);
                }
            }
            return Err(
                "each ordering term must come from the same metric (lab, luv, xyz, or RGB)",
            );
        };
        let sort_indices: Vec<SortIndex> = terms
            .into_iter()
            .map(|term| SortIndex {
                idx: channel_letters.find(term.0).unwrap(),
                dir: term.1,
            })
            .collect();

        Ok(Self {
            shuffle_first: sort_indices.len() < 3,
            ordering_metric,
            sort_indices,
        })
    }
}

fn color_converter(metric: ColorMetric) -> fn(Vec<(u32, Color)>) -> Vec<(u32, Color)> {
    fn do_convert(colors: Vec<(u32, Color)>, conv: impl Fn(u32) -> Color) -> Vec<(u32, Color)> {
        colors
            .into_iter()
            .map(|(argb, _)| (argb, conv(argb)))
            .collect()
    }
    fn convert_to_lab(colors: Vec<(u32, Color)>) -> Vec<(u32, Color)> {
        do_convert(colors, |argb| {
            let lab: LABColor<f32> = SRGBColor::from_argb(argb).into();
            lab.into()
        })
    }
    fn convert_to_luv(colors: Vec<(u32, Color)>) -> Vec<(u32, Color)> {
        do_convert(colors, |argb| {
            let luv: LUVColor<f32> = SRGBColor::from_argb(argb).into();
            luv.into()
        })
    }
    fn convert_to_xyz(colors: Vec<(u32, Color)>) -> Vec<(u32, Color)> {
        do_convert(colors, |argb| {
            let xyz: XYZColor<f32> = SRGBColor::from_argb(argb).into();
            xyz.into()
        })
    }
    fn convert_to_rgb(colors: Vec<(u32, Color)>) -> Vec<(u32, Color)> {
        do_convert(colors, |argb| {
            let rgb: RGBColor<f32> = SRGBColor::from_argb(argb).into();
            rgb.into()
        })
    }
    fn convert_to_srgb(colors: Vec<(u32, Color)>) -> Vec<(u32, Color)> {
        do_convert(colors, |argb| SRGBColor::from_argb(argb).into())
    }
    match metric {
        Lab => convert_to_lab,
        Luv => convert_to_luv,
        Xyz => convert_to_xyz,
        Rgb => convert_to_rgb,
        Srgb => convert_to_srgb,
    }
}

pub fn colordeposit_main<Frontier: ColorMap>()
where
    LABColor<f32>: Into<Color>,
{
    let args = Cli::parse();

    const ZERO_COLOR: Color = Color([0.0; 3]);
    let color_metric = args.color_metric.unwrap_or(Lab);
    let ordering = args
        .ordering
        .unwrap_or(Ordering::from_str("shuffle").unwrap());
    let rng = &mut match args.seed {
        Some(seed) => StdRng::from_seed({
            let mut h = Sha512::new();
            h.update(seed.as_bytes());
            *try_prefix(&h.finalize()).unwrap()
        }),
        None => StdRng::from_entropy(),
    };

    println!("initializing");
    let mut colors: Vec<u32> = (0..(1 << 24)).collect();
    if ordering.shuffle_first {
        println!("shuffling");
        colors.shuffle(rng);
    }
    let mut color_values = colors.into_iter().map(|argb| (argb, ZERO_COLOR)).collect();
    if !ordering.sort_indices.is_empty() {
        println!("converting colors for sort");
        color_values = color_converter(ordering.ordering_metric)(color_values);
        println!("sorting");
        color_values.sort_by(|a, b| {
            for term in ordering.sort_indices.iter() {
                let cmp = (&a.1 .0[term.idx])
                    .partial_cmp(&b.1 .0[term.idx])
                    .unwrap_or(Equal);
                let cmp = match term.dir {
                    Asc => cmp,
                    Desc => cmp.reverse(),
                };
                if cmp.is_ne() {
                    return cmp;
                }
            }
            Equal
        })
    }
    if ordering.sort_indices.is_empty() || color_metric != ordering.ordering_metric {
        println!("converting colors for similarity");
        color_values = color_converter(color_metric)(color_values);
    }

    let field = ImageField {
        width: 4096,
        height: 4096,
    };
    assert_eq!(field.len(), color_values.len());
    let mut frontier = Frontier::new();
    frontier.insert(
        field.position_to_index(Position(2048, 2048)) as u32,
        ZERO_COLOR,
    );

    const UNPLACED: u32 = u32::MAX;
    const FRONTIER: u32 = UNPLACED - 1;
    let mut output_buffer = vec![UNPLACED; field.len()];

    let start_time = Instant::now();
    let log_frequency = Duration::from_millis(100);
    let mut last_log = start_time;
    let mut last_log_progress = 0usize;
    let mut next_log = last_log + log_frequency;
    let bar = ProgressBar::new(color_values.len() as u64)
        .with_style(
            ProgressStyle::with_template(
                "{prefix} {wide_bar} {elapsed_precise}/{eta_precise} | {percent_precise}% \
                | {msg}",
            )
            .unwrap(),
        )
        .with_prefix("placing")
        .with_finish(ProgressFinish::AndLeave);
    let mut tick_bar = |now: Instant, i, frontier_size| {
        let seconds_since_last_log = (now - last_log).as_nanos() as f32 / 1e9;
        let rate = (i - last_log_progress) as f32 / seconds_since_last_log;
        bar.set_position(i as u64);
        bar.set_message(format!("{rate:.0} px/sec, frontier size {frontier_size}"));
        bar.tick();
        last_log = now;
        last_log_progress = i;
    };
    for (i, (_, placing_color)) in color_values.iter().enumerate() {
        let now = Instant::now();
        if now >= next_log {
            tick_bar(now, i, frontier.len());
            next_log = now + log_frequency;
        }
        let best_frontier_key = frontier.nearest(placing_color);

        // Placing a pixel here, so it is removed from the frontier
        frontier.remove(best_frontier_key);
        #[cfg(feature = "full_validation")]
        frontier.fully_validate();
        output_buffer[best_frontier_key as usize] = i as u32;
        let placed_position = field.index_to_position(best_frontier_key as usize);
        for (neighbor_offset, _, placeable) in SAMPLE_WEIGHTS {
            let neighbor_pos = placed_position + *neighbor_offset;
            // `neighbor_pos` must be in the image
            if !field.contains(neighbor_pos) {
                continue;
            }
            let neighbor_index = field.position_to_index(neighbor_pos);
            // `neighbor_pos` must be an unplaced pixel
            match output_buffer[neighbor_index] {
                UNPLACED => {
                    if !*placeable {
                        continue;
                    }
                    output_buffer[neighbor_index] = FRONTIER;
                }
                FRONTIER => {}
                _ => continue, // already placed
            }
            let neighbor_key = neighbor_index as u32;
            // Average the color values for the placed pixels in sampling range of neighbor_pos
            let mut weight = 0f32;
            let mut summed = ZERO_COLOR;
            for (sample_offset, sample_weight, _) in SAMPLE_WEIGHTS {
                let sample_pos = neighbor_pos + *sample_offset;
                // `sample_pos` must be in the image
                if !field.contains(sample_pos) {
                    continue;
                }
                // `sample_pos` must be an already-placed pixel
                let sampled_color_idx = output_buffer[field.position_to_index(sample_pos)];
                if sampled_color_idx == UNPLACED || sampled_color_idx == FRONTIER {
                    continue;
                }
                let (_srgb, sampled_color) = &color_values[sampled_color_idx as usize];
                weight += sample_weight;
                summed.mul_add_assign(sampled_color, *sample_weight);
            }
            let new_frontier_color = summed / weight;
            // Update the frontier color at that position
            frontier.insert(neighbor_key, new_frontier_color);
            #[cfg(feature = "full_validation")]
            frontier.fully_validate();
        }
    }
    let finish_time = Instant::now();
    bar.set_position(color_values.len() as u64);
    let total_seconds = (finish_time - start_time).as_nanos() as f32 / 1e9;
    let rate = color_values.len() as f32 / total_seconds;
    bar.set_message(format!("average {rate:.0} px/sec"));
    bar.finish();

    // Remap output buffer back to srgb colors
    println!("saving...");
    let path = args.output.unwrap_or(PathBuf::from("output.png"));
    let file = File::create(path).unwrap();
    let w = &mut BufWriter::new(file);
    let mut encoder = png::Encoder::new(w, field.width, field.height);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_srgb(png::SrgbRenderingIntent::RelativeColorimetric);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.set_compression(png::Compression::Best);
    encoder.set_adaptive_filter(png::AdaptiveFilterType::Adaptive);
    let mut png_writer = encoder.write_header().unwrap();
    png_writer
        .write_image_data(
            output_buffer
                .drain(..)
                .flat_map(|i| match i {
                    UNPLACED | FRONTIER => [0u8; 3],
                    _ => {
                        let (srgb, _) = color_values[i as usize];
                        [
                            ((srgb & 0xff0000) >> 16) as u8,
                            ((srgb & 0x00ff00) >> 8) as u8,
                            (srgb & 0x0000ff) as u8,
                        ]
                    }
                })
                .collect::<Vec<u8>>()
                .as_mut_slice(),
        )
        .unwrap();
    png_writer.finish().unwrap();
    println!("done!");
}

// TODO(widders): bench cashmere against the kiddo tests and those + others against color deposit
// TODO(widders): make sure it's benched against kdtree, fnntw, nabo, idk check around for others
