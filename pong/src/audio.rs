use std::io::Read;
use std::sync::{Arc, Mutex};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::Sample as _;
use itertools::Itertools;
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};
pub struct AudioSystem {
    stream: cpal::Stream,
    sounds: Arc<Mutex<Vec<Sound>>>,
    sample_rate: f64,
}

#[derive(Debug, Clone)]
pub struct Sound {
    data: Vec<f32>,
    channels: u16,
    sample_rate: u32,
    i: usize,
    repeat: bool,
    playing: bool,
}

impl AudioSystem {
    pub fn new() -> eyre::Result<Self> {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .expect("no output device available");
        println!("Output device: {}", device.name()?);
        let config = device.default_output_config().unwrap();
        println!("Default output config: {:?}", config);

        let sample_rate = config.config().sample_rate.0 as f64;

        let sounds = Arc::new(Mutex::new(Vec::<Sound>::new()));
        let sounds_stream = sounds.clone();

        let stream = device.build_output_stream(
            &config.config(),
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                let mut sounds_lock = sounds_stream.lock().unwrap();
                for sample in data.iter_mut() {
                    *sample = 0.0;
                }
                for sound in sounds_lock.iter_mut().filter(|sound| sound.playing) {
                    for sample in data.iter_mut() {
                        *sample += sound.data[sound.i];

                        sound.i += 1;
                        if sound.i >= sound.data.len() {
                            sound.i = 0;
                            if !sound.repeat {
                                sound.playing = false;
                                break;
                            }
                        }
                    }
                }
            },
            move |err| eprintln!("an error occurred on stream: {}", err),
            None,
        )?;
        stream.play()?;

        Ok(Self {
            stream,
            sounds,
            sample_rate,
        })
    }

    pub fn play_sound(&self, sound: Sound) {
        // Play sound
        self.sounds.lock().unwrap().push(sound);
    }

    pub fn load_sound<R: Read>(&self, reader: R, repeat: bool) -> eyre::Result<Sound> {
        let reader = hound::WavReader::new(reader)?;

        let spec = reader.spec();
        let samples = match spec.sample_format {
            hound::SampleFormat::Int => match spec.bits_per_sample {
                8 => reader
                    .into_samples::<i8>()
                    .map(|s| s.map(|s| s as f32 / f32::powi(2., 7)))
                    .collect::<Result<Vec<_>, _>>()?,
                16 => reader
                    .into_samples::<i16>()
                    .map(|s| s.map(|s| s as f32 / f32::powi(2., 15)))
                    .collect::<Result<Vec<_>, _>>()?,
                24 => reader
                    .into_samples::<i32>()
                    .map(|s| s.map(|s| s as f32 / f32::powi(2., 23)))
                    .collect::<Result<Vec<_>, _>>()?,
                32 => reader
                    .into_samples::<i32>()
                    .map(|s| s.map(|s| s as f32 / f32::powi(2., 31)))
                    .collect::<Result<Vec<_>, _>>()?,
                _ => return Err(eyre::eyre!("unsupported bits per sample")),
            },
            hound::SampleFormat::Float => {
                if spec.bits_per_sample == 32 {
                    reader
                        .into_samples::<f32>()
                        .collect::<Result<Vec<_>, _>>()?
                } else {
                    return Err(eyre::eyre!("unsupported bits per sample"));
                }
            }
        };

        let mut samples_channels: Vec<Vec<f32>> = (0..spec.channels)
            .map(|c| {
                samples
                    .iter()
                    .skip(c as usize)
                    .step_by(spec.channels as usize)
                    .copied()
                    .collect::<Vec<f32>>()
            })
            .collect();
        samples_channels
            .iter()
            .for_each(|c| println!("Channel length: {}", c.len()));

        if samples_channels.len() == 1 {
            // mono sound
            samples_channels.push(samples_channels[0].clone());
        }

        let params = SincInterpolationParameters {
            sinc_len: 256,
            f_cutoff: 0.95,
            interpolation: SincInterpolationType::Linear,
            oversampling_factor: 256,
            window: WindowFunction::BlackmanHarris2,
        };
        let mut resampler = SincFixedIn::<f32>::new(
            self.sample_rate / spec.sample_rate as f64,
            2.0,
            params,
            samples_channels[0].len(),
            2,
        )
        .unwrap();

        let samples_channels = resampler.process(&samples_channels, None).unwrap();

        let samples = {
            let mut samples: Vec<f32> = Vec::with_capacity(samples_channels[0].len() * samples_channels.len());
            for i in 0..samples_channels[0].len() {
                for c in 0..samples_channels.len() {
                    samples.push(samples_channels[c as usize][i]);
                }
            }
            samples
        };

        println!("Samples length: {}", samples.len());

        // Load sound
        Ok(Sound {
            data: samples,
            channels: spec.channels,
            sample_rate: spec.sample_rate,
            i: 0,
            repeat,
            playing: true,
        })
    }
}

fn run<T>() {}
